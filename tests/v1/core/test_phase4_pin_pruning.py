import json
import time
from collections import defaultdict, deque
from types import SimpleNamespace

from vllm.v1.core.sched.output import CompactReplayData
from vllm.v1.outputs import ManagedContextTransferOutput
from vllm.v1.core.sched.scheduler import (
    CompactReplayFullRefillState,
    ManagedContextDeferredRestore,
    ManagedContextHotGPUStats,
    ManagedContextSpan,
    Phase4Pin,
    SchedulingPolicy,
    Scheduler,
)
from vllm.v1.request import RequestStatus
from vllm.v1.utils import ConstantList


class _FakeRequestQueue(deque):
    def prepend_request(self, request) -> None:
        self.appendleft(request)

    def prepend_requests(self, requests) -> None:
        for request in reversed(list(requests)):
            self.appendleft(request)

    def add_request(self, request) -> None:
        self.append(request)

    def peek_request(self):
        return self[0]

    def remove_request(self, request) -> None:
        self.remove(request)

    def remove_requests(self, requests) -> None:
        for request in requests:
            try:
                self.remove(request)
            except ValueError:
                pass


class _FakeBlockPool:
    def __init__(self, new_block_ids: list[int] | None = None) -> None:
        self.freed_blocks = []
        self.evicted_ids = []
        self.touched_blocks = []
        self._new_block_ids = deque(new_block_ids or [])

    def free_blocks(self, blocks) -> None:
        self.freed_blocks.extend(list(blocks))

    def touch(self, blocks) -> None:
        self.touched_blocks.extend(list(blocks))

    def evict_blocks(self, block_ids) -> None:
        self.evicted_ids.extend(sorted(block_ids))

    def get_num_free_blocks(self) -> int:
        return len(self._new_block_ids)

    def get_new_blocks(self, count: int):
        if count > len(self._new_block_ids):
            raise ValueError("not enough free blocks")
        return [
            SimpleNamespace(
                block_id=self._new_block_ids.popleft(),
                logical_start=-1,
                block_hash=None,
                is_null=False,
            )
            for _ in range(count)
        ]


def _scheduler_with_requests(request_ids: set[str]) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request_id: object() for request_id in request_ids}
    scheduler.running = []
    scheduler.waiting = _FakeRequestQueue()
    scheduler.skipped_waiting = _FakeRequestQueue()
    scheduler._phase4_pinned_blocks = {}
    scheduler._phase4_pin_order = deque()
    scheduler._phase4_pin_store_event_to_trace_id = {}
    scheduler._phase4_pin_load_event_to_trace_id = {}
    scheduler._managed_context_active_restores = {}
    scheduler._managed_context_deferred_restores = {}
    scheduler._managed_context_pending_loads = {}
    scheduler._managed_context_restore_reservations = {}
    scheduler._managed_context_archive = {}
    scheduler._managed_context_archive_order = deque()
    scheduler._managed_context_hot_gpu_order = deque()
    scheduler._managed_context_hot_gpu_stats = ManagedContextHotGPUStats(
        budget_blocks=0
    )
    scheduler._managed_context_store_events_to_submit = {}
    scheduler._managed_context_load_events_to_submit = {}
    scheduler._managed_context_store_event_to_span = {}
    scheduler._managed_context_load_event_to_request_id = {}
    scheduler._request_kv_swap_store_event_to_request_id = {}
    scheduler._request_kv_swap_load_event_to_request_id = {}
    scheduler._managed_context_cpu_free_block_ids = deque()
    scheduler._managed_context_cpu_max_blocks = 0
    scheduler._managed_context_cpu_offload_immediate = False
    scheduler._managed_context_cpu_evict_on_capacity = False
    scheduler._managed_context_cpu_max_pending_store_events = 0
    scheduler._managed_context_cpu_max_pending_load_events = 0
    scheduler._managed_context_next_transfer_event_id = 0
    scheduler._pending_admission_compaction_ids = set()
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler.num_cumulative_preemption = 0
    scheduler.log_stats = False
    scheduler._compaction_enabled = True
    scheduler._compaction_max_turns = 1
    scheduler._managed_context_enabled = True
    scheduler._managed_context_cpu_archive_enabled = False
    scheduler._managed_context_recall_max_spans = 2
    scheduler._managed_context_recall_max_kv_blocks = None
    scheduler.cache_config = SimpleNamespace(enable_prefix_caching=True)
    return scheduler


def _request(
    trace_id: str,
    expected_cached_tokens: int | None = 64,
    *,
    request_id: str = "reader",
) -> SimpleNamespace:
    extra_args = {"kve_phase4_trace_id": trace_id}
    if expected_cached_tokens is not None:
        extra_args["kve_phase4_expected_cached_tokens"] = (
            expected_cached_tokens
        )
    return SimpleNamespace(
        request_id=request_id,
        status=RequestStatus.WAITING,
        num_prompt_tokens=512,
        prompt_token_ids=[1] * 512,
        num_computed_tokens=0,
        num_external_computed_tokens=0,
        num_cached_tokens=0,
        position_offset=0,
        skip_reading_prefix_cache=False,
        needs_rebuild=False,
        _kve_reprefill_after_flush=False,
        _kve_phase4_reprefill_after_pin_release=False,
        sampling_params=SimpleNamespace(
            extra_args=extra_args,
        ),
    )


def _pin(
    *,
    token_count: int = 0,
    consumed_by_request_id: str | None = None,
    consumed_at: float | None = None,
    entries=None,
    block_count: int | None = None,
    last_loaded_at: float | None = None,
) -> Phase4Pin:
    if block_count is None:
        block_count = sum(len(blocks) for _manager, blocks in entries or [])
    return Phase4Pin(
        entries=[] if entries is None else entries,
        token_count=token_count,
        block_count=block_count,
        request_id="writer",
        call_idx=1,
        created_at=time.monotonic() - 100.0,
        consumed_by_request_id=consumed_by_request_id,
        consumed_at=consumed_at,
        last_loaded_at=last_loaded_at,
    )


def _gpu_blocks(start: int, count: int):
    return [
        SimpleNamespace(
            block_id=start + idx,
            logical_start=(start + idx) * 16,
            block_hash=f"h{start + idx}",
            is_null=False,
        )
        for idx in range(count)
    ]


def _managed_span(
    span_id: str,
    trace_id: str,
    manager,
    blocks,
    *,
    status: str = "gpu_pinned",
) -> ManagedContextSpan:
    return ManagedContextSpan(
        span_id=span_id,
        trace_id=trace_id,
        request_id="writer",
        absolute_turn_start=0,
        absolute_turn_end=1,
        token_ids=[1] * (len(blocks) * 16),
        entries=[(manager, blocks)],
        kv_block_count=len(blocks),
        logical_start_by_group=[
            [int(block.logical_start) for block in blocks]
        ],
        position_offset_frame=0,
        evict_start=0,
        evict_end=len(blocks) * 16,
        created_at=time.monotonic(),
        status=status,
    )


def test_phase4_consumed_pin_is_kept_while_consumer_is_live() -> None:
    scheduler = _scheduler_with_requests({"reader"})
    pin = _pin(consumed_by_request_id="reader", consumed_at=time.monotonic() - 60.0)

    assert not scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=1800.0
    )


def test_phase4_consumed_pin_survives_after_consumer_grace(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    pin = _pin(consumed_by_request_id="reader", consumed_at=time.monotonic() - 2.0)

    assert not scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=1800.0
    )


def test_phase4_consumed_pin_prunes_after_regular_ttl(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    pin = _pin(
        consumed_by_request_id="reader",
        consumed_at=time.monotonic() - 60.0,
    )
    pin.created_at = time.monotonic() - 60.0

    assert scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=30.0
    )


def test_phase4_consumed_pin_keeps_short_grace_for_retry(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "5")
    scheduler = _scheduler_with_requests(set())
    pin = _pin(consumed_by_request_id="reader", consumed_at=time.monotonic() - 2.0)

    assert not scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=1800.0
    )


def test_phase4_consumed_pin_keeps_queued_successor_after_grace(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler.waiting.append(_request("trace"))
    pin = _pin(
        token_count=128,
        consumed_by_request_id="reader",
        consumed_at=time.monotonic() - 2.0,
    )

    assert not scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=1800.0
    )


def test_phase4_consumed_pin_keeps_queued_successor_after_ttl(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler.waiting.append(_request("trace", expected_cached_tokens=128))
    pin = _pin(
        token_count=128,
        consumed_by_request_id="reader",
        consumed_at=time.monotonic() - 60.0,
    )
    pin.created_at = time.monotonic() - 60.0

    assert not scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=30.0
    )


def test_phase4_consumed_pin_without_expected_tokens_prunes_after_ttl(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler.waiting.append(_request("trace", expected_cached_tokens=None))
    pin = _pin(
        token_count=128,
        consumed_by_request_id="reader",
        consumed_at=time.monotonic() - 2.0,
    )
    pin.created_at = time.monotonic() - 60.0

    assert scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=30.0
    )


def test_phase4_consumed_pin_unsatisfied_successor_prunes_after_ttl(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler.waiting.append(_request("trace", expected_cached_tokens=256))
    pin = _pin(
        token_count=128,
        consumed_by_request_id="reader",
        consumed_at=time.monotonic() - 2.0,
    )
    pin.created_at = time.monotonic() - 60.0

    assert scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=30.0
    )


def test_phase4_consumed_pin_ttl_zero_disables_prune_after_grace(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    pin = _pin(
        consumed_by_request_id="reader",
        consumed_at=time.monotonic() - 2.0,
    )

    assert not scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=0.0
    )


def test_phase4_pin_prune_compacts_stale_order_entries() -> None:
    scheduler = _scheduler_with_requests(set())
    scheduler._phase4_pinned_blocks = {
        "trace-a": _pin(),
        "trace-b": _pin(),
    }
    scheduler._phase4_pin_order = deque(
        ["trace-a", "stale", "trace-b", "trace-a"]
    )

    scheduler._prune_phase4_pins()

    assert list(scheduler._phase4_pin_order) == ["trace-b", "trace-a"]


def test_phase4_pressure_release_skips_running_consumer() -> None:
    scheduler = _scheduler_with_requests({"reader"})
    scheduler.running = [SimpleNamespace(request_id="reader")]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            consumed_by_request_id="reader",
            consumed_at=time.monotonic() - 60.0,
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])
    released = []

    def fake_release_phase4_pins(trace_id, reason, *, evict_prefix=False):
        released.append((trace_id, reason, evict_prefix))

    scheduler._release_phase4_pins = fake_release_phase4_pins

    assert not scheduler._release_phase4_pressure_pin("pressure")
    assert released == []


def test_phase4_pressure_release_skips_queued_successor(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler.waiting.append(_request("trace", expected_cached_tokens=64))
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            token_count=128,
            consumed_by_request_id="reader",
            consumed_at=time.monotonic() - 2.0,
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])
    released = []

    def fake_release_phase4_pins(trace_id, reason, *, evict_prefix=False):
        released.append((trace_id, reason, evict_prefix))

    scheduler._release_phase4_pins = fake_release_phase4_pins

    assert not scheduler._release_phase4_pressure_pin("pressure")
    assert released == []


def test_phase4_running_pressure_keeps_unconsumed_pin_without_cpu() -> None:
    scheduler = _scheduler_with_requests(set())
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(token_count=128, consumed_by_request_id=None)
    }
    scheduler._phase4_pin_order = deque(["trace"])
    released = []

    def fake_release_phase4_pins(trace_id, reason, *, evict_prefix=False):
        released.append((trace_id, reason, evict_prefix))

    scheduler._release_phase4_pins = fake_release_phase4_pins

    assert not scheduler._release_phase4_pressure_pin("running-alloc-pressure")
    assert released == []
    assert "trace" in scheduler._phase4_pinned_blocks


def test_phase4_running_pressure_offloads_unconsumed_pin_async(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 16
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
        SimpleNamespace(block_id=12, logical_start=160, block_hash="c"),
        SimpleNamespace(block_id=13, logical_start=176, block_hash="d"),
    ]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            token_count=128,
            consumed_by_request_id=None,
            entries=[(manager, blocks)],
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])

    assert not scheduler._release_phase4_pressure_pin("running-alloc-pressure")

    pin = scheduler._phase4_pinned_blocks["trace"]
    assert pin.status == "store_pending"
    assert pin.store_event_id == 0
    assert pin.cpu_block_ids_by_group == ([0, 1, 2, 3],)
    assert scheduler._phase4_pin_store_event_to_trace_id == {0: "trace"}
    assert scheduler._managed_context_store_events_to_submit[0].gpu_block_ids == [
        10,
        11,
        12,
        13,
    ]
    assert block_pool.freed_blocks == []


def test_phase4_stall_pressure_offloads_unconsumed_pin_without_successor(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 16
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
        SimpleNamespace(block_id=12, logical_start=160, block_hash="c"),
        SimpleNamespace(block_id=13, logical_start=176, block_hash="d"),
    ]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            token_count=128,
            consumed_by_request_id=None,
            entries=[(manager, blocks)],
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])

    assert scheduler._release_phase4_pressure_pin("scheduler-stall-pressure")

    pin = scheduler._phase4_pinned_blocks["trace"]
    assert pin.status == "store_pending"
    assert pin.store_event_id == 0
    assert scheduler._managed_context_store_events_to_submit[0].gpu_block_ids == [
        10,
        11,
        12,
        13,
    ]
    assert block_pool.freed_blocks == []


def test_phase4_stall_pressure_offloads_queued_successor_pin(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "999999")
    scheduler = _scheduler_with_requests({"reader"})
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 16
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    request = _request("trace", expected_cached_tokens=64)
    scheduler.waiting.append(request)
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
        SimpleNamespace(block_id=12, logical_start=160, block_hash="c"),
        SimpleNamespace(block_id=13, logical_start=176, block_hash="d"),
    ]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            token_count=128,
            consumed_by_request_id="old-reader",
            consumed_at=time.monotonic(),
            entries=[(manager, blocks)],
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])

    assert scheduler._release_phase4_pressure_pin("scheduler-stall-pressure")

    pin = scheduler._phase4_pinned_blocks["trace"]
    assert pin.status == "store_pending"
    assert pin.cpu_block_ids_by_group == ([0, 1, 2, 3],)
    assert pin.logical_start_by_group == ([128, 144, 160, 176],)
    assert pin.store_event_id == 0
    assert scheduler._phase4_pin_store_event_to_trace_id == {0: "trace"}
    assert scheduler._managed_context_store_events_to_submit[0].gpu_block_ids == [
        10,
        11,
        12,
        13,
    ]
    assert scheduler._managed_context_store_events_to_submit[0].cpu_block_ids == [
        0,
        1,
        2,
        3,
    ]
    assert list(scheduler._managed_context_cpu_free_block_ids) == list(range(4, 16))
    assert block_pool.freed_blocks == []
    assert request.position_offset == 0
    assert request.num_computed_tokens == 0
    assert request.num_external_computed_tokens == 0
    assert request.num_cached_tokens == 0
    assert not request.skip_reading_prefix_cache
    assert not request.needs_rebuild
    assert not request._kve_reprefill_after_flush
    assert not request._kve_phase4_reprefill_after_pin_release
    assert scheduler._phase4_has_queued_successor(
        "trace",
        _pin(token_count=128),
    )


def test_phase4_pressure_replay_drops_pin_without_cpu_offload(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    monkeypatch.setenv("KVE_PHASE4_PRESSURE_REPLAY", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 16
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
        SimpleNamespace(block_id=12, logical_start=160, block_hash="c"),
        SimpleNamespace(block_id=13, logical_start=176, block_hash="d"),
    ]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            token_count=128,
            consumed_by_request_id=None,
            entries=[(manager, blocks)],
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])

    assert scheduler._release_phase4_pressure_pin("running-alloc-pressure")

    assert "trace" not in scheduler._phase4_pinned_blocks
    assert block_pool.freed_blocks == blocks
    assert block_pool.evicted_ids == []
    assert scheduler._phase4_pin_store_event_to_trace_id == {}
    assert scheduler._managed_context_store_events_to_submit == {}
    assert list(scheduler._managed_context_cpu_free_block_ids) == list(
        range(16)
    )


def test_phase4_pressure_replay_can_force_evict_prefix(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    monkeypatch.setenv("KVE_PHASE4_PRESSURE_REPLAY", "1")
    monkeypatch.setenv("KVE_PHASE4_REPLAY_DROP_EVICT_PREFIX", "1")
    scheduler = _scheduler_with_requests(set())
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
    ]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            token_count=128,
            consumed_by_request_id=None,
            entries=[(manager, blocks)],
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])

    assert scheduler._release_phase4_pressure_pin("running-alloc-pressure")

    assert block_pool.freed_blocks == blocks
    assert block_pool.evicted_ids == [10, 11]


def test_phase4_pressure_replay_requires_full_replay(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_PRESSURE_REPLAY", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 16
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
    ]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            token_count=128,
            consumed_by_request_id=None,
            entries=[(manager, blocks)],
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])

    assert not scheduler._release_phase4_pressure_pin(
        "running-alloc-pressure"
    )

    pin = scheduler._phase4_pinned_blocks["trace"]
    assert pin.status == "store_pending"
    assert scheduler._phase4_pin_store_event_to_trace_id == {0: "trace"}
    assert block_pool.freed_blocks == []


def test_phase4_pin_store_and_load_completion_round_trip(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "999999")
    scheduler = _scheduler_with_requests({"reader"})
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 16
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    request = _request("trace", expected_cached_tokens=64)
    scheduler.waiting.append(request)
    block_pool = _FakeBlockPool(new_block_ids=[20, 21, 22, 23])
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
        SimpleNamespace(block_id=12, logical_start=160, block_hash="c"),
        SimpleNamespace(block_id=13, logical_start=176, block_hash="d"),
    ]
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            token_count=128,
            consumed_by_request_id="old-reader",
            consumed_at=time.monotonic(),
            entries=[(manager, blocks)],
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])

    assert scheduler._release_phase4_pressure_pin("scheduler-stall-pressure")
    scheduler._drain_managed_context_transfer_metadata()
    scheduler._update_from_managed_context_transfer_finished(
        ManagedContextTransferOutput(completed_store_event_ids=[0])
    )

    pin = scheduler._phase4_pinned_blocks["trace"]
    assert pin.status == "cpu_offloaded"
    assert pin.entries == []
    assert block_pool.freed_blocks == blocks
    assert list(scheduler._managed_context_cpu_free_block_ids) == list(range(4, 16))

    error = scheduler._start_phase4_pin_cpu_load(
        "trace",
        pin,
        reason="test",
    )

    assert error is None
    assert pin.status == "load_pending"
    assert scheduler._phase4_pin_load_event_to_trace_id == {1: "trace"}
    scheduler._drain_managed_context_transfer_metadata()
    scheduler._update_from_managed_context_transfer_finished(
        ManagedContextTransferOutput(completed_load_event_ids=[1])
    )

    assert pin.status == "gpu_pinned"
    assert pin.last_loaded_at is not None
    assert pin.cpu_block_ids_by_group == ()
    assert [block.block_id for _manager, group in pin.entries for block in group] == [
        20,
        21,
        22,
        23,
    ]
    assert [block.logical_start for _manager, group in pin.entries for block in group] == [
        128,
        144,
        160,
        176,
    ]
    assert list(scheduler._managed_context_cpu_free_block_ids) == list(range(4, 16)) + [
        0,
        1,
        2,
        3,
    ]


def test_phase4_proactive_offload_moves_nonproductive_pin_to_cpu(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_CPU_OFFLOAD", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 16
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
    ]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(token_count=128, entries=[(manager, blocks)])
    }
    scheduler._phase4_pin_order = deque(["trace"])

    assert (
        scheduler._proactively_offload_nonproductive_phase4_pins("test")
        == 1
    )

    pin = scheduler._phase4_pinned_blocks["trace"]
    assert pin.status == "store_pending"
    assert pin.store_event_id == 0
    assert pin.cpu_block_ids_by_group == ([0, 1],)
    assert scheduler._managed_context_store_events_to_submit[0].gpu_block_ids == [
        10,
        11,
    ]
    assert block_pool.freed_blocks == []


def test_phase4_proactive_offload_keeps_running_trace_gpu(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_CPU_OFFLOAD", "1")
    scheduler = _scheduler_with_requests({"reader"})
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 16
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    request = _request("trace", request_id="reader")
    request.status = RequestStatus.RUNNING
    scheduler.requests = {"reader": request}
    scheduler.running = [request]
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
    ]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(token_count=128, entries=[(manager, blocks)])
    }
    scheduler._phase4_pin_order = deque(["trace"])

    assert (
        scheduler._proactively_offload_nonproductive_phase4_pins("test")
        == 0
    )

    assert scheduler._phase4_pinned_blocks["trace"].status == "gpu_pinned"
    assert scheduler._managed_context_store_events_to_submit == {}
    assert block_pool.freed_blocks == []


def test_phase4_proactive_wrapper_skips_below_start_watermark(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_CPU_OFFLOAD", "1")
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_OFFLOAD_START_USAGE", "0.90")
    scheduler = _scheduler_with_requests(set())
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 16
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    block_pool = _FakeBlockPool(new_block_ids=list(range(50)))
    block_pool.num_gpu_blocks = 100
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
    ]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(token_count=128, entries=[(manager, blocks)])
    }
    scheduler._phase4_pin_order = deque(["trace"])

    scheduler._proactively_offload_nonproductive_kv("test")

    assert scheduler._phase4_pinned_blocks["trace"].status == "gpu_pinned"
    assert scheduler._managed_context_store_events_to_submit == {}
    assert block_pool.freed_blocks == []


def test_phase4_proactive_wrapper_offloads_above_start_watermark(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_CPU_OFFLOAD", "1")
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_OFFLOAD_START_USAGE", "0.90")
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_OFFLOAD_STOP_USAGE", "0.78")
    scheduler = _scheduler_with_requests(set())
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 16
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    block_pool = _FakeBlockPool(new_block_ids=list(range(5)))
    block_pool.num_gpu_blocks = 100
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
    ]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(token_count=128, entries=[(manager, blocks)])
    }
    scheduler._phase4_pin_order = deque(["trace"])

    scheduler._proactively_offload_nonproductive_kv("test")

    pin = scheduler._phase4_pinned_blocks["trace"]
    assert pin.status == "store_pending"
    assert pin.store_event_id == 0
    assert scheduler._managed_context_store_events_to_submit[0].gpu_block_ids == [
        10,
        11,
    ]


def test_phase4_proactive_keeps_recent_loaded_queued_successor(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_CPU_OFFLOAD", "1")
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_PIN_LOAD_GRACE_SECONDS", "60")
    scheduler = _scheduler_with_requests({"reader"})
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 16
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    scheduler.waiting.append(_request("trace", request_id="reader"))
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
    ]
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            token_count=128,
            entries=[(manager, blocks)],
            last_loaded_at=time.monotonic(),
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])

    assert (
        scheduler._proactively_offload_nonproductive_phase4_pins("test")
        == 0
    )

    assert scheduler._phase4_pinned_blocks["trace"].status == "gpu_pinned"
    assert scheduler._managed_context_store_events_to_submit == {}
    assert block_pool.freed_blocks == []


def test_phase4_proactive_releases_nonproductive_hot_gpu_span(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_CPU_OFFLOAD", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler._managed_context_cpu_archive_enabled = True
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
        SimpleNamespace(block_id=11, logical_start=144, block_hash="b"),
    ]
    span = ManagedContextSpan(
        span_id="T0001",
        trace_id="trace",
        request_id="writer",
        absolute_turn_start=0,
        absolute_turn_end=1,
        token_ids=[1, 2],
        entries=[(manager, blocks)],
        kv_block_count=2,
        logical_start_by_group=[[128, 144]],
        position_offset_frame=0,
        evict_start=0,
        evict_end=2,
        created_at=time.monotonic(),
        status="cpu_hot",
        cpu_block_ids_by_group=([0, 1],),
    )
    scheduler._managed_context_archive = {("trace", "T0001"): span}
    scheduler._managed_context_hot_gpu_order = deque([("trace", "T0001")])
    scheduler._managed_context_hot_gpu_stats = ManagedContextHotGPUStats(
        budget_blocks=16,
        resident_blocks=2,
    )

    assert (
        scheduler._proactively_release_nonproductive_hot_gpu_spans("test")
        == 2
    )

    assert span.status == "cpu_offloaded"
    assert span.entries == []
    assert block_pool.freed_blocks == blocks
    assert scheduler._managed_context_hot_gpu_stats.resident_blocks == 0


def test_phase4_proactive_offload_auto_enabled_for_full_replay(
    monkeypatch,
) -> None:
    scheduler = _scheduler_with_requests(set())
    scheduler._managed_context_cpu_archive_enabled = True

    assert not scheduler._phase4_proactive_cpu_offload_enabled()

    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    assert not scheduler._phase4_proactive_cpu_offload_enabled()

    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_CPU_OFFLOAD", "0")
    assert not scheduler._phase4_proactive_cpu_offload_enabled()

    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_CPU_OFFLOAD", "1")
    assert scheduler._phase4_proactive_cpu_offload_enabled()


def test_phase4_proactive_release_does_not_protect_future_reservations(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_CPU_OFFLOAD", "1")
    scheduler = _scheduler_with_requests({"reader"})
    scheduler._managed_context_cpu_archive_enabled = True
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    blocks = [
        SimpleNamespace(block_id=10, logical_start=128, block_hash="a"),
    ]
    span = ManagedContextSpan(
        span_id="T0001",
        trace_id="trace",
        request_id="writer",
        absolute_turn_start=0,
        absolute_turn_end=1,
        token_ids=[1],
        entries=[(manager, blocks)],
        kv_block_count=1,
        logical_start_by_group=[[128]],
        position_offset_frame=0,
        evict_start=0,
        evict_end=1,
        created_at=time.monotonic(),
        status="cpu_hot",
        cpu_block_ids_by_group=([0],),
    )
    key = ("trace", "T0001")
    scheduler._managed_context_archive = {key: span}
    scheduler._managed_context_hot_gpu_order = deque([key])
    scheduler._managed_context_restore_reservations["reader"] = {key}
    scheduler._managed_context_hot_gpu_stats = ManagedContextHotGPUStats(
        budget_blocks=16,
        resident_blocks=1,
    )

    assert scheduler._proactively_release_nonproductive_hot_gpu_spans("test") == 1
    assert span.status == "cpu_offloaded"
    assert block_pool.freed_blocks == blocks


def test_phase4_productive_traces_do_not_include_all_restore_reservations(
    monkeypatch,
) -> None:
    scheduler = _scheduler_with_requests({"reader"})
    request = _request("trace", expected_cached_tokens=64)
    scheduler.requests["reader"] = request
    scheduler._managed_context_restore_reservations["reader"] = {
        ("trace", "T0001"),
    }
    scheduler._phase4_pinned_blocks["trace"] = _pin(
        token_count=128,
        block_count=8,
    )
    scheduler._phase4_pin_order = deque(["trace"])

    assert "trace" not in scheduler._phase4_productive_gpu_trace_ids()

    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_KEEP_QUEUED_SUCCESSORS", "1")
    scheduler.waiting.append(request)

    assert "trace" in scheduler._phase4_productive_gpu_trace_ids()


def test_phase4_proactive_replay_drop_runs_without_cpu_offload(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    monkeypatch.setenv("KVE_PHASE4_PRESSURE_REPLAY", "1")
    scheduler = _scheduler_with_requests({"active-reader"})
    active_request = _request(
        "active",
        expected_cached_tokens=64,
        request_id="active-reader",
    )
    active_request.status = RequestStatus.RUNNING
    scheduler.requests["active-reader"] = active_request
    scheduler.running = [active_request]

    block_pool = _FakeBlockPool(new_block_ids=list(range(5)))
    block_pool.num_gpu_blocks = 100
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    active_blocks = _gpu_blocks(100, 20)
    cold_big_blocks = _gpu_blocks(10, 12)
    cold_small_blocks = _gpu_blocks(30, 5)
    scheduler._phase4_pinned_blocks = {
        "active": _pin(
            token_count=320,
            entries=[(manager, active_blocks)],
        ),
        "cold-big": _pin(
            token_count=192,
            entries=[(manager, cold_big_blocks)],
        ),
        "cold-small": _pin(
            token_count=80,
            entries=[(manager, cold_small_blocks)],
        ),
    }
    scheduler._phase4_pin_order = deque(["active", "cold-small", "cold-big"])

    scheduler._proactively_offload_nonproductive_kv("test")

    assert "active" in scheduler._phase4_pinned_blocks
    assert "cold-big" not in scheduler._phase4_pinned_blocks
    assert "cold-small" not in scheduler._phase4_pinned_blocks
    assert block_pool.freed_blocks == cold_big_blocks + cold_small_blocks
    assert block_pool.evicted_ids == []
    assert scheduler._managed_context_store_events_to_submit == {}
    assert scheduler._phase4_pin_store_event_to_trace_id == {}


def test_phase4_proactive_replay_drop_can_be_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    monkeypatch.setenv("KVE_PHASE4_PRESSURE_REPLAY", "1")
    monkeypatch.setenv("KVE_PHASE4_PROACTIVE_REPLAY_DROP", "0")
    scheduler = _scheduler_with_requests(set())
    block_pool = _FakeBlockPool(new_block_ids=list(range(5)))
    block_pool.num_gpu_blocks = 100
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    cold_blocks = _gpu_blocks(10, 12)
    scheduler._phase4_pinned_blocks = {
        "cold": _pin(token_count=192, entries=[(manager, cold_blocks)])
    }
    scheduler._phase4_pin_order = deque(["cold"])

    scheduler._proactively_offload_nonproductive_kv("test")

    assert "cold" in scheduler._phase4_pinned_blocks
    assert block_pool.freed_blocks == []
    assert block_pool.evicted_ids == []


def test_managed_context_restore_admission_parks_on_request_budget(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_RESTORE_ADMISSION", "1")
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_ACTIVE_RESTORE_MAX_REQUESTS", "1")
    scheduler = _scheduler_with_requests({"active", "reader"})
    block_pool = _FakeBlockPool(new_block_ids=list(range(10)))
    block_pool.num_gpu_blocks = 100
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    active_blocks = _gpu_blocks(100, 2)
    scheduler._managed_context_active_restores = {
        "active": SimpleNamespace(entries=[(manager, active_blocks)])
    }
    request = _request("trace", request_id="reader")
    span = _managed_span("T0001", "trace", manager, _gpu_blocks(10, 2))

    error = scheduler._managed_context_restore_admission_error(
        request,
        [span],
    )

    assert error is not None
    assert "inflight restore request budget" in error


def test_managed_context_restore_admission_counts_pending_loads(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_RESTORE_ADMISSION", "1")
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_ACTIVE_RESTORE_MAX_REQUESTS", "1")
    scheduler = _scheduler_with_requests({"pending", "reader"})
    block_pool = _FakeBlockPool(new_block_ids=list(range(10)))
    block_pool.num_gpu_blocks = 100
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    scheduler._managed_context_pending_loads = {
        "pending": SimpleNamespace(
            restored_entries_by_span={
                "T0000": [(manager, _gpu_blocks(100, 2))]
            }
        )
    }
    request = _request("trace", request_id="reader")
    span = _managed_span("T0001", "trace", manager, _gpu_blocks(10, 2))

    error = scheduler._managed_context_restore_admission_error(
        request,
        [span],
    )

    assert error is not None
    assert "inflight restore request budget" in error


def test_managed_context_restore_admission_allows_first_restore_over_budget(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_RESTORE_ADMISSION", "1")
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_ACTIVE_RESTORE_MAX_REQUESTS", "1")
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_ACTIVE_RESTORE_MAX_BLOCKS", "4")
    scheduler = _scheduler_with_requests({"reader"})
    block_pool = _FakeBlockPool(new_block_ids=list(range(10)))
    block_pool.num_gpu_blocks = 100
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    request = _request("trace", request_id="reader")
    span = _managed_span("T0001", "trace", manager, _gpu_blocks(10, 8))

    assert (
        scheduler._managed_context_restore_admission_error(request, [span])
        is None
    )


def test_managed_context_restore_admission_parks_on_block_budget(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_RESTORE_ADMISSION", "1")
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_ACTIVE_RESTORE_MAX_BLOCKS", "10")
    scheduler = _scheduler_with_requests({"active", "reader"})
    block_pool = _FakeBlockPool(new_block_ids=list(range(10)))
    block_pool.num_gpu_blocks = 100
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    active_blocks = _gpu_blocks(100, 8)
    scheduler._managed_context_active_restores = {
        "active": SimpleNamespace(entries=[(manager, active_blocks)])
    }
    request = _request("trace", request_id="reader")
    span = _managed_span("T0001", "trace", manager, _gpu_blocks(10, 4))

    error = scheduler._managed_context_restore_admission_error(
        request,
        [span],
    )

    assert error is not None
    assert "hidden KV block budget" in error


def test_deferred_restore_admission_keeps_request_parked(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_RESTORE_ADMISSION", "1")
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_ACTIVE_RESTORE_MAX_REQUESTS", "1")
    scheduler = _scheduler_with_requests({"active", "reader"})
    block_pool = _FakeBlockPool(new_block_ids=list(range(10)))
    block_pool.num_gpu_blocks = 100
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    scheduler._managed_context_active_restores = {
        "active": SimpleNamespace(entries=[(manager, _gpu_blocks(100, 2))])
    }
    request = _request("trace", request_id="reader")
    request.num_prompt_tokens = 8
    request.num_computed_tokens = 7
    request.sampling_params.extra_args = {
        "kve_phase4_trace_id": "trace",
        "kve_restore_span_ids": ["T0001"],
        "kve_restore_defer_until_prefill": True,
    }
    span = _managed_span("T0001", "trace", manager, _gpu_blocks(10, 2))
    deferred = ManagedContextDeferredRestore(
        span_ids=["T0001"],
        spans=[span],
        restored_entries_by_span={},
        skip_hot_hit_span_ids=set(),
        created_at=time.monotonic(),
    )
    scheduler._managed_context_deferred_restores["reader"] = deferred

    assert not scheduler._activate_deferred_managed_context_restore_if_ready(
        request
    )

    assert scheduler._managed_context_deferred_restores["reader"] is deferred
    assert "reader" not in scheduler._managed_context_active_restores
    assert scheduler._managed_context_restore_admission_is_deferred(request)
    assert scheduler._managed_context_deferred_restore_blocked_by_admission(
        request
    )


def test_restore_admission_deferred_skipped_request_yields_to_waiting(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_RESTORE_ADMISSION", "1")
    scheduler = _scheduler_with_requests({"waiting", "restore"})
    scheduler.policy = SchedulingPolicy.FCFS
    waiting_request = _request("waiting-trace", request_id="waiting")
    restore_request = _request("restore-trace", request_id="restore")
    scheduler.waiting.append(waiting_request)
    scheduler.skipped_waiting.append(restore_request)
    scheduler._set_managed_context_restore_admission_deferred(
        restore_request,
        "restore budget",
    )

    assert scheduler._select_waiting_queue_for_scheduling() is scheduler.waiting


def test_phase4_pin_load_watermark_parks_projected_high_usage(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_PIN_LOAD_TARGET_USAGE", "0.50")
    scheduler = _scheduler_with_requests({"reader"})
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_free_block_ids = deque(range(16))
    block_pool = _FakeBlockPool(new_block_ids=[20, 21, 22, 23, 24])
    block_pool.num_gpu_blocks = 10
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    pin = _pin(token_count=128, block_count=2)
    pin.status = "cpu_offloaded"
    pin.cpu_block_ids_by_group = ([0, 1],)
    pin.logical_start_by_group = ([128, 144],)

    error = scheduler._start_phase4_pin_cpu_load("trace", pin, reason="test")

    assert error is not None
    assert error.startswith("Phase4 pin load is parked by GPU watermark")
    assert Scheduler._phase4_pin_recovery_defers(error)
    assert pin.status == "cpu_offloaded"
    assert scheduler._managed_context_load_events_to_submit == {}


def test_phase4_prefix_miss_reprefill_requires_explicit_opt_in(
    monkeypatch,
) -> None:
    scheduler = _scheduler_with_requests({"reader"})
    request = _request("trace", expected_cached_tokens=64)

    assert not scheduler._phase4_requeue_prefix_miss_for_reprefill(
        request,
        "prefix miss",
    )
    assert not request._kve_reprefill_after_flush

    monkeypatch.setenv("KVE_PHASE4_PREFIX_MISS_REPREFILL", "1")

    assert scheduler._phase4_requeue_prefix_miss_for_reprefill(
        request,
        "prefix miss",
    )
    assert request._kve_reprefill_after_flush


def test_phase4_stall_pressure_does_not_replay_when_pin_offload_unavailable() -> None:
    scheduler = _scheduler_with_requests(set())
    scheduler.waiting.append(_request("trace", expected_cached_tokens=64))
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(token_count=128)
    }
    scheduler._phase4_pin_order = deque(["trace"])
    released = []

    def fake_release_phase4_pins(trace_id, reason, *, evict_prefix=False):
        released.append((trace_id, reason, evict_prefix))

    scheduler._release_phase4_pins = fake_release_phase4_pins

    assert not scheduler._release_phase4_pressure_pin(
        "scheduler-stall-pressure"
    )
    assert released == []
    assert not scheduler.waiting[0]._kve_reprefill_after_flush


def test_phase4_stall_pressure_does_not_report_pending_pin_progress() -> None:
    scheduler = _scheduler_with_requests(set())
    pin = _pin(token_count=128, consumed_by_request_id=None)
    pin.status = "store_pending"
    scheduler._phase4_pinned_blocks = {"trace": pin}
    scheduler._phase4_pin_order = deque(["trace"])
    released = []

    def fake_release_phase4_pins(trace_id, reason, *, evict_prefix=False):
        released.append((trace_id, reason, evict_prefix))

    scheduler._release_phase4_pins = fake_release_phase4_pins

    assert not scheduler._release_phase4_pressure_pin("scheduler-stall-pressure")
    assert released == []
    assert scheduler._phase4_pinned_blocks["trace"].status == "store_pending"


def test_phase4_error_request_does_not_replace_pin() -> None:
    scheduler = _scheduler_with_requests(set())
    scheduler._phase4_pinned_blocks = {"trace": _pin(token_count=128)}
    scheduler._phase4_pin_order = deque(["trace"])
    request = _request("trace", expected_cached_tokens=128)
    request.status = RequestStatus.FINISHED_ERROR
    released = []

    def fake_release_phase4_pins(trace_id, reason, *, evict_prefix=False):
        released.append((trace_id, reason, evict_prefix))

    scheduler._release_phase4_pins = fake_release_phase4_pins

    scheduler._pin_phase4_request_blocks(request)

    assert released == []
    assert scheduler._phase4_pinned_blocks["trace"].token_count == 128


def test_phase4_replay_cold_publish_soft_releases_published_pin(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    monkeypatch.setenv("KVE_PHASE4_REPLAY_COLD_PUBLISH", "1")
    scheduler = _scheduler_with_requests(set())
    block_pool = _FakeBlockPool()
    blocks = [
        SimpleNamespace(
            block_id=10 + idx,
            logical_start=idx * 16,
            block_hash=f"h{idx}",
            is_null=False,
        )
        for idx in range(4)
    ]
    manager = SimpleNamespace(
        block_size=16,
        block_pool=block_pool,
        req_to_blocks={"writer-full": blocks},
    )
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    request = _request(
        "trace",
        expected_cached_tokens=64,
        request_id="writer-full",
    )
    request.status = RequestStatus.FINISHED_STOPPED
    request.num_tokens = 64
    request.sampling_params.extra_args["kve_phase4_call_idx"] = 2

    trace_id = scheduler._pin_phase4_request_blocks(request)

    assert trace_id == "trace"
    assert "trace" in scheduler._phase4_pinned_blocks
    assert scheduler._maybe_cold_release_phase4_published_pin(trace_id)
    assert "trace" not in scheduler._phase4_pinned_blocks
    assert block_pool.touched_blocks == blocks
    assert block_pool.freed_blocks == blocks
    assert block_pool.evicted_ids == []


def test_phase4_replay_cold_publish_requires_full_replay(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_REPLAY_COLD_PUBLISH", "1")
    scheduler = _scheduler_with_requests(set())
    block_pool = _FakeBlockPool()
    blocks = [
        SimpleNamespace(
            block_id=10,
            logical_start=0,
            block_hash="h0",
            is_null=False,
        )
    ]
    manager = SimpleNamespace(
        block_size=16,
        block_pool=block_pool,
        req_to_blocks={"writer-full": blocks},
    )
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    request = _request(
        "trace",
        expected_cached_tokens=16,
        request_id="writer-full",
    )
    request.status = RequestStatus.FINISHED_STOPPED
    request.num_tokens = 16

    trace_id = scheduler._pin_phase4_request_blocks(request)

    assert trace_id == "trace"
    assert not scheduler._maybe_cold_release_phase4_published_pin(trace_id)
    assert "trace" in scheduler._phase4_pinned_blocks
    assert block_pool.freed_blocks == []


def test_phase4_replay_cold_publish_can_evict_prefix(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    monkeypatch.setenv("KVE_PHASE4_REPLAY_COLD_PUBLISH", "1")
    monkeypatch.setenv("KVE_PHASE4_REPLAY_DROP_EVICT_PREFIX", "1")
    scheduler = _scheduler_with_requests(set())
    block_pool = _FakeBlockPool()
    blocks = [
        SimpleNamespace(
            block_id=10 + idx,
            logical_start=idx * 16,
            block_hash=f"h{idx}",
            is_null=False,
        )
        for idx in range(2)
    ]
    manager = SimpleNamespace(
        block_size=16,
        block_pool=block_pool,
        req_to_blocks={"writer-full": blocks},
    )
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    request = _request(
        "trace",
        expected_cached_tokens=32,
        request_id="writer-full",
    )
    request.status = RequestStatus.FINISHED_STOPPED
    request.num_tokens = 32

    trace_id = scheduler._pin_phase4_request_blocks(request)

    assert scheduler._maybe_cold_release_phase4_published_pin(trace_id)
    assert block_pool.freed_blocks == blocks
    assert block_pool.evicted_ids == [10, 11]


def test_phase4_replay_cold_publish_hot_pin_limit_keeps_newest(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    monkeypatch.setenv("KVE_PHASE4_REPLAY_COLD_PUBLISH", "1")
    monkeypatch.setenv("KVE_PHASE4_REPLAY_HOT_PIN_LIMIT", "2")
    scheduler = _scheduler_with_requests(set())
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    old_blocks = _gpu_blocks(10, 1)
    mid_blocks = _gpu_blocks(20, 1)
    new_blocks = _gpu_blocks(30, 1)
    scheduler._phase4_pinned_blocks = {
        "old": _pin(token_count=16, entries=[(manager, old_blocks)]),
        "mid": _pin(token_count=16, entries=[(manager, mid_blocks)]),
        "new": _pin(token_count=16, entries=[(manager, new_blocks)]),
    }
    scheduler._phase4_pin_order = deque(["old", "mid", "new"])

    released = scheduler._enforce_phase4_replay_hot_pin_budget("test-budget")

    assert released == 1
    assert set(scheduler._phase4_pinned_blocks) == {"mid", "new"}
    assert list(scheduler._phase4_pin_order) == ["mid", "new"]
    assert block_pool.freed_blocks == old_blocks
    assert block_pool.evicted_ids == []


def test_phase4_replay_cold_publish_hot_block_limit_keeps_within_budget(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    monkeypatch.setenv("KVE_PHASE4_REPLAY_COLD_PUBLISH", "1")
    monkeypatch.setenv("KVE_PHASE4_REPLAY_HOT_PIN_BLOCK_LIMIT", "3")
    scheduler = _scheduler_with_requests(set())
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    old_blocks = _gpu_blocks(10, 2)
    mid_blocks = _gpu_blocks(20, 2)
    new_blocks = _gpu_blocks(30, 1)
    scheduler._phase4_pinned_blocks = {
        "old": _pin(token_count=32, entries=[(manager, old_blocks)]),
        "mid": _pin(token_count=32, entries=[(manager, mid_blocks)]),
        "new": _pin(token_count=16, entries=[(manager, new_blocks)]),
    }
    scheduler._phase4_pin_order = deque(["old", "mid", "new"])

    released = scheduler._enforce_phase4_replay_hot_pin_budget("test-budget")

    assert released == 1
    assert set(scheduler._phase4_pinned_blocks) == {"mid", "new"}
    assert list(scheduler._phase4_pin_order) == ["mid", "new"]
    assert block_pool.freed_blocks == old_blocks


def test_phase4_stall_pressure_releases_multiple_pins(monkeypatch) -> None:
    monkeypatch.setenv("KVE_PHASE4_STALL_PRESSURE_RELEASE_MAX", "3")
    scheduler = _scheduler_with_requests(set())
    scheduler.waiting.append(_request("trace"))
    scheduler._kve_gpu_block_pool_diag_summary = lambda: []
    scheduler._kve_managed_context_diag_summary = lambda: {}
    scheduler._managed_context_release_hot_gpu_pressure = lambda reason: 0
    release_results = [True, True, False]
    release_calls = []

    def fake_release_phase4_pressure_pin(reason):
        release_calls.append(reason)
        return release_results.pop(0)

    scheduler._release_phase4_pressure_pin = fake_release_phase4_pressure_pin

    assert scheduler._maybe_release_phase4_stall_pressure_pin(
        total_num_scheduled_tokens=0
    )
    assert release_calls == [
        "scheduler-stall-pressure",
        "scheduler-stall-pressure",
        "scheduler-stall-pressure",
    ]


def test_phase4_stall_pressure_drains_pins_before_hot_spans(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_STALL_PRESSURE_RELEASE_MAX", "3")
    scheduler = _scheduler_with_requests(set())
    scheduler.waiting.append(_request("trace"))
    scheduler._kve_gpu_block_pool_diag_summary = lambda: []
    scheduler._kve_managed_context_diag_summary = lambda: {}
    calls = []

    def fake_release_phase4_pins_for_blocks(needed_blocks, *, reason):
        calls.append(("pins", needed_blocks, reason))
        return 2

    def fake_release_hot(reason):
        calls.append(("hot", reason))
        return 64

    scheduler._release_phase4_pressure_pins_for_blocks = (
        fake_release_phase4_pins_for_blocks
    )
    scheduler._managed_context_release_hot_gpu_pressure = fake_release_hot

    assert scheduler._maybe_release_phase4_stall_pressure_pin(
        total_num_scheduled_tokens=0
    )
    assert calls == [
        ("pins", 0, "scheduler-stall-pressure"),
        ("hot", "scheduler-stall-pressure"),
    ]


def test_phase4_stall_pressure_does_not_abandon_remote_kv_waiter(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "999999")
    scheduler = _scheduler_with_requests({"reader"})
    request = _request("trace", expected_cached_tokens=64)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.skipped_waiting.append(request)
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            token_count=128,
            consumed_by_request_id="old-reader",
            consumed_at=time.monotonic(),
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])
    released = []

    def fake_release_phase4_pins(trace_id, reason, *, evict_prefix=False):
        released.append((trace_id, reason, evict_prefix))

    scheduler._release_phase4_pins = fake_release_phase4_pins

    assert not scheduler._release_phase4_pressure_pin(
        "scheduler-stall-pressure"
    )
    assert released == []
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert not request._kve_reprefill_after_flush


def test_phase4_queued_successor_uses_active_reprefill_marker() -> None:
    scheduler = _scheduler_with_requests({"reader"})
    request = _request("trace", expected_cached_tokens=64)
    request._kve_phase4_reprefill_after_pin_release = True
    scheduler.waiting.append(request)

    assert scheduler._phase4_has_queued_successor(
        "trace",
        _pin(token_count=128),
    )

    request._kve_reprefill_after_flush = True
    assert not scheduler._phase4_has_queued_successor(
        "trace",
        _pin(token_count=128),
    )


def test_managed_context_restore_ids_are_reserved_before_schedule() -> None:
    scheduler = _scheduler_with_requests({"reader"})
    request = _request("trace", expected_cached_tokens=64)
    request.sampling_params.extra_args["kve_restore_span_ids"] = [
        "T0001",
        "T0002",
    ]

    scheduler._reserve_managed_context_restore_request(request)

    assert scheduler._managed_context_restore_reservations["reader"] == {
        ("trace", "T0001"),
        ("trace", "T0002"),
    }


def test_managed_context_unavailable_restore_drop_is_opt_in(
    monkeypatch,
) -> None:
    scheduler = _scheduler_with_requests({"reader"})
    request = _request("trace", expected_cached_tokens=64)
    request.sampling_params.extra_args["kve_restore_span_ids"] = ["T0001"]

    assert not scheduler._drop_unavailable_managed_context_restore(
        request,
        "span 'T0001' is not available for this trace",
    )
    assert request.sampling_params.extra_args["kve_restore_span_ids"] == [
        "T0001",
    ]

    monkeypatch.setenv("KVE_MANAGED_CONTEXT_DROP_UNAVAILABLE_RESTORE", "1")
    assert scheduler._drop_unavailable_managed_context_restore(
        request,
        "span 'T0001' is not available for this trace",
    )
    assert "kve_restore_span_ids" not in request.sampling_params.extra_args
    assert (
        request.sampling_params.extra_args["kve_phase4_trace_id"]
        == "trace"
    )


def test_phase4_pressure_release_keeps_stale_prefix_without_cpu(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler._phase4_pinned_blocks = {
        "trace": _pin(
            token_count=128,
            consumed_by_request_id="reader",
            consumed_at=time.monotonic() - 2.0,
        )
    }
    scheduler._phase4_pin_order = deque(["trace"])
    released = []

    def fake_release_phase4_pins(trace_id, reason, *, evict_prefix=False):
        released.append((trace_id, reason, evict_prefix))
        scheduler._phase4_pinned_blocks.pop(trace_id, None)

    scheduler._release_phase4_pins = fake_release_phase4_pins

    assert not scheduler._release_phase4_pressure_pin("pressure")
    assert released == []
    assert "trace" in scheduler._phase4_pinned_blocks


def test_phase4_stall_pressure_release_requires_no_progress(monkeypatch) -> None:
    monkeypatch.setenv("KVE_PHASE4_STALL_PRESSURE_RELEASE_MAX", "1")
    scheduler = _scheduler_with_requests(set())
    scheduler.waiting.append(_request("trace"))
    calls = []

    def fake_release_phase4_pressure_pin(reason):
        calls.append(reason)
        return True

    scheduler._release_phase4_pressure_pin = fake_release_phase4_pressure_pin
    scheduler._kve_gpu_block_pool_diag_summary = lambda: "gpu"
    scheduler._kve_managed_context_diag_summary = lambda: "managed"

    assert not scheduler._maybe_release_phase4_stall_pressure_pin(
        total_num_scheduled_tokens=1
    )
    assert calls == []

    assert scheduler._maybe_release_phase4_stall_pressure_pin(
        total_num_scheduled_tokens=0
    )
    assert calls == ["scheduler-stall-pressure"]


def _replayable_request(
    request_id: str,
    *,
    position_offset: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        status=RequestStatus.RUNNING,
        position_offset=position_offset,
        padding_pending=False,
        num_output_placeholders=0,
        prompt_token_ids=[1, 2, 3, 4],
        num_prompt_tokens=4,
        num_tokens=8,
        num_output_tokens=4,
        num_computed_tokens=8,
        num_external_computed_tokens=0,
        num_cached_tokens=8,
        num_preemptions=0,
        spec_token_ids=[99],
        is_prefill_chunk=False,
        needs_rebuild=False,
        skip_reading_prefix_cache=False,
        _kve_reprefill_after_flush=False,
        _kve_phase4_reprefill_after_pin_release=False,
        sampling_params=SimpleNamespace(extra_args={}),
        compaction_events=[object()],
    )


def test_compacted_preempt_flushes_and_requeues_for_reprefill() -> None:
    scheduler = _scheduler_with_requests({"compact"})
    request = _replayable_request("compact", position_offset=128)
    request.sampling_params.extra_args = {
        "kve_phase4_trace_id": "trace",
        "kve_restore_span_ids": ["T0001"],
    }
    scheduler.requests = {"compact": request}
    freed_visible = []
    freed_encoder = []
    released = []

    scheduler.kv_cache_manager = SimpleNamespace(
        free=lambda req: freed_visible.append(req.request_id)
    )
    scheduler.encoder_cache_manager = SimpleNamespace(
        free=lambda req: freed_encoder.append(req.request_id)
    )
    scheduler._release_managed_context_active_restore = (
        lambda req_id, reason: released.append(("active", req_id, reason))
    )
    scheduler._release_managed_context_deferred_restore = (
        lambda req_id, reason: released.append(("deferred", req_id, reason))
    )
    scheduler._release_managed_context_restore_reservation = (
        lambda req_id, reason: released.append(("reserve", req_id, reason))
    )

    scheduler._preempt_request(request, time.monotonic())

    assert request.status == RequestStatus.PREEMPTED
    assert request.position_offset == 128
    assert request.num_computed_tokens == 0
    assert request.num_cached_tokens == -1
    assert request.num_preemptions == 1
    assert request.spec_token_ids == []
    assert request.needs_rebuild
    assert request.skip_reading_prefix_cache
    assert request._kve_reprefill_after_flush
    assert list(scheduler.waiting) == [request]
    assert freed_visible == ["compact"]
    assert freed_encoder == ["compact"]
    assert scheduler._managed_context_restore_reservations["compact"] == {
        ("trace", "T0001"),
    }
    assert released == [
        ("active", "compact", "reprefill-preempt"),
        ("deferred", "compact", "reprefill-preempt"),
        ("reserve", "compact", "reprefill-preempt"),
    ]


def test_compacted_preempt_defers_padding_pending_request() -> None:
    scheduler = _scheduler_with_requests({"compact"})
    request = _replayable_request("compact", position_offset=128)
    request.padding_pending = True

    result = scheduler._preempt_request(request, time.monotonic())

    assert result.kind == "deferred"
    assert request.status == RequestStatus.RUNNING
    assert request.padding_pending
    assert list(scheduler.waiting) == []


def test_compacted_preempt_uses_request_kv_swap_before_reprefill() -> None:
    scheduler = _scheduler_with_requests({"compact"})
    request = _replayable_request("compact", position_offset=128)
    scheduler._request_kv_swaps = {
        "compact": SimpleNamespace(kv_block_count=2),
    }
    swap_calls = []

    def start_request_kv_swap_out(request, reason):
        swap_calls.append((request.request_id, reason))
        return None

    def fail_reprefill(*args, **kwargs):
        raise AssertionError("re-prefill should not run after swap succeeds")

    scheduler._start_request_kv_swap_out = start_request_kv_swap_out
    scheduler._preempt_request_for_reprefill = fail_reprefill

    scheduler._preempt_request(request, time.monotonic())

    assert swap_calls == [("compact", "compacted")]
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_preemptions == 1
    assert request.spec_token_ids == []
    assert not request._kve_reprefill_after_flush
    assert list(scheduler.waiting) == [request]


def test_reprefill_restamp_keeps_protected_prefix_in_zero_frame() -> None:
    scheduler = _scheduler_with_requests({"compact"})
    request = _replayable_request("compact", position_offset=128)
    request._kve_reprefill_after_flush = True
    blocks = [
        SimpleNamespace(is_null=False, logical_start=128),
        SimpleNamespace(is_null=False, logical_start=144),
        SimpleNamespace(is_null=False, logical_start=160),
    ]
    manager = SimpleNamespace(
        block_size=16,
        req_to_blocks={"compact": blocks},
    )
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    scheduler._worker_protected_prefix_len = lambda req: 32

    scheduler._restamp_reprefill_logical_starts(
        request,
        reason="test",
    )

    assert [block.logical_start for block in blocks] == [0, 16, 160]


def test_reprefill_marker_clears_after_visible_prompt_ready() -> None:
    scheduler = _scheduler_with_requests({"compact"})
    request = _replayable_request("compact", position_offset=128)
    request._kve_reprefill_after_flush = True
    request.num_prompt_tokens = 4
    request.num_computed_tokens = 3

    scheduler._clear_reprefill_after_flush_if_prompt_ready(
        request,
        reason="test-partial",
    )

    assert request._kve_reprefill_after_flush

    request.num_computed_tokens = 4
    scheduler._clear_reprefill_after_flush_if_prompt_ready(
        request,
        reason="test-ready",
    )

    assert not request._kve_reprefill_after_flush


def test_compact_replay_reprefill_rearms_cpu_offload_after_prompt_ready() -> None:
    scheduler = _scheduler_with_requests({"compact"})
    request = _replayable_request("compact", position_offset=128)
    replay_snapshot = SimpleNamespace(
        evictions=1,
        final_writer_len=8,
        live_writer_indices=(0, 3, 4, 5, 6, 7),
    )
    request.compact_replay_snapshot = lambda: replay_snapshot

    assert scheduler._mark_request_for_full_reprefill(
        request,
        "test-compact-replay",
    )
    assert request._kve_compact_replay_refill_snapshot is replay_snapshot
    assert request._kve_compact_replay_refill_active

    offload_calls = []
    scheduler._retry_managed_context_gpu_pinned_offloads = (
        lambda reason: offload_calls.append(("managed", reason)) or 0
    )
    scheduler._proactively_offload_nonproductive_kv = (
        lambda reason: offload_calls.append(("phase4", reason))
    )

    request.num_prompt_tokens = 4
    request.num_computed_tokens = 4
    scheduler._clear_reprefill_after_flush_if_prompt_ready(
        request,
        reason="test-ready",
    )

    assert not request._kve_reprefill_after_flush
    assert not request._kve_compact_replay_refill_active
    assert offload_calls == [
        ("managed", "compact-replay-refill-done"),
        ("phase4", "compact-replay-refill-done"),
    ]


def test_compact_replay_snapshot_is_sent_on_rebuild() -> None:
    scheduler = _scheduler_with_requests({"compact"})
    request = _replayable_request("compact", position_offset=128)
    request.all_token_ids = [1, 3, 4, 5, 6, 7]
    request.num_output_tokens = 2
    replay_snapshot = SimpleNamespace(
        token_ids=(1, 2, 3, 4, 5, 6, 7),
        death_indices=(7, 4, 7, 7, 7, 7, 7),
        live_writer_indices=(0, 2, 3, 4, 5, 6),
        evictions=1,
        final_writer_len=7,
    )
    request.compact_replay_snapshot = lambda: replay_snapshot

    assert scheduler._mark_request_for_full_reprefill(
        request,
        "test-compact-replay",
    )

    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            get_blocks=lambda _req_id: ([SimpleNamespace(block_id=10)],),
            single_type_managers=[SimpleNamespace(block_size=16)],
        )
    )
    scheduler._managed_context_active_hidden_kv = lambda _req_id: ((), 0, [])
    scheduler._worker_protected_prefix_len = lambda _req: 0
    scheduler.use_pp = False
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=False)

    cached = scheduler._make_cached_request_data(
        running_reqs=[request],
        resumed_reqs=[],
        num_scheduled_tokens={"compact": 4},
        spec_decode_tokens={},
        req_to_new_blocks={},
    )

    replay_data = cached.compact_replay_data["compact"]
    assert replay_data == CompactReplayData.from_snapshot(replay_snapshot)
    assert cached.rebuild_req_ids == {"compact"}
    assert cached.all_token_ids["compact"] == [1, 3, 4, 5, 6, 7]


def _make_full_replay_request() -> SimpleNamespace:
    request = _replayable_request("compact", position_offset=128)
    request.prompt_token_ids = [1, 3, 4, 5]
    request.num_prompt_tokens = 4
    request._output_token_ids = [6, 7]
    request.output_token_ids = ConstantList(request._output_token_ids)
    request._all_token_ids = [1, 3, 4, 5, 6, 7]
    request.all_token_ids = ConstantList(request._all_token_ids)
    request.block_hashes = []
    request.update_block_hashes = lambda: None
    return request


def test_compact_replay_full_refill_activation_swaps_to_writer_tokens(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    scheduler = _scheduler_with_requests({"compact"})
    scheduler._compact_replay_compaction_manager = lambda: SimpleNamespace(
        block_size=2
    )
    request = _make_full_replay_request()
    replay_snapshot = SimpleNamespace(
        token_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        death_indices=(8, 8, 4, 4, 8, 8, 8, 8),
        live_writer_indices=(0, 1, 4, 5, 6, 7),
        evictions=1,
        final_writer_len=8,
    )

    assert scheduler._activate_compact_replay_full_refill(
        request,
        replay_snapshot,
        reason="test",
    )

    state = request._kve_compact_replay_full_refill_state
    assert isinstance(state, CompactReplayFullRefillState)
    assert request.prompt_token_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert request._all_token_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert request._output_token_ids == []
    assert request.num_prompt_tokens == 8
    assert request.position_offset == 0
    assert request._kve_compact_replay_full_refill_active
    assert state.dead_ranges == [(2, 4)]


def test_compact_replay_full_refill_consumes_restore_xargs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    scheduler = _scheduler_with_requests({"compact"})
    scheduler._compact_replay_compaction_manager = lambda: SimpleNamespace(
        block_size=2
    )
    request = _make_full_replay_request()
    request.sampling_params.extra_args = {
        "kve_phase4_trace_id": "trace",
        "kve_restore_span_ids": ["T0001"],
        "kve_restore_defer_until_prefill": True,
        "kve_restore_after_visible_tokens": 3,
        "kve_compact_replay_spans": "[]",
    }
    scheduler._managed_context_restore_reservations["compact"] = {
        ("trace", "T0001")
    }
    replay_snapshot = SimpleNamespace(
        token_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        death_indices=(8, 8, 4, 4, 8, 8, 8, 8),
        live_writer_indices=(0, 1, 4, 5, 6, 7),
        evictions=1,
        final_writer_len=8,
    )

    assert scheduler._activate_compact_replay_full_refill(
        request,
        replay_snapshot,
        reason="test",
    )

    assert "kve_restore_span_ids" not in request.sampling_params.extra_args
    assert "kve_restore_defer_until_prefill" not in request.sampling_params.extra_args
    assert "kve_restore_after_visible_tokens" not in request.sampling_params.extra_args
    assert request.sampling_params.extra_args["kve_compact_replay_spans"] == "[]"
    assert not request.managed_context_defer_restore_until_prefill
    assert scheduler._managed_context_restore_reservations == {}


def test_compact_replay_full_refill_completion_restores_live_state(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    scheduler = _scheduler_with_requests({"compact"})
    compact_calls = []

    def compact_request(_request_id, _prompt_tokens, *, explicit_block_range):
        compact_calls.append(explicit_block_range)
        start, end = explicit_block_range
        return (end - start) * 2

    scheduler._compact_replay_compaction_manager = lambda: SimpleNamespace(
        block_size=2,
        compact_request=compact_request,
    )
    request = _make_full_replay_request()
    request.turn_end_positions = [1, 3, 4]
    request.last_turn_scan_pos = 6
    request.num_turns_evicted = 2
    replay_snapshot = SimpleNamespace(
        token_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        death_indices=(8, 8, 4, 4, 8, 8, 8, 8),
        live_writer_indices=(0, 1, 4, 5, 6, 7),
        evictions=1,
        final_writer_len=8,
    )

    assert scheduler._activate_compact_replay_full_refill(
        request,
        replay_snapshot,
        reason="test",
    )
    request.num_computed_tokens = 8
    request.turn_end_positions = [1, 2, 4, 6, 8]
    request.last_turn_scan_pos = 8
    request.num_turns_evicted = 99

    assert scheduler._complete_compact_replay_full_refill(
        request,
        reason="test",
    )

    assert compact_calls == [(1, 2)]
    assert request.prompt_token_ids == [1, 3, 4, 5]
    assert request._all_token_ids == [1, 3, 4, 5, 6, 7]
    assert request._output_token_ids == [6, 7]
    assert request.num_prompt_tokens == 4
    assert request.num_computed_tokens == 6
    assert request.num_cached_tokens == 0
    assert request.position_offset == 128
    assert request.turn_end_positions == [1, 3, 4]
    assert request.last_turn_scan_pos == 6
    assert request.num_turns_evicted == 2
    assert request.needs_rebuild
    assert not request._kve_reprefill_after_flush
    assert not request._kve_compact_replay_full_refill_active


def test_compact_replay_segmented_refill_compacts_at_boundary(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "segmented")
    scheduler = _scheduler_with_requests({"compact"})
    scheduler._compaction_max_turns = 0
    scheduler._worker_protected_prefix_len = lambda _req: 2
    scheduler.cache_config = SimpleNamespace(enable_prefix_caching=False)
    compact_calls = []

    def compact_request(_request_id, _prompt_tokens, *, explicit_block_range):
        compact_calls.append(explicit_block_range)
        start, end = explicit_block_range
        return (end - start) * 2

    scheduler._compact_replay_compaction_manager = lambda: SimpleNamespace(
        block_size=2,
        compact_request=compact_request,
    )
    request = _make_full_replay_request()
    replay_snapshot = SimpleNamespace(
        token_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        death_indices=(8, 8, 4, 4, 8, 8, 8, 8),
        live_writer_indices=(0, 1, 4, 5, 6, 7),
        evictions=1,
        final_writer_len=8,
    )

    assert scheduler._activate_compact_replay_segmented_refill(
        request,
        replay_snapshot,
        reason="test",
    )
    assert request.prompt_token_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert request._kve_compact_replay_segmented_deletions == [
        (4, 2, 4, 4)
    ]
    assert scheduler._cap_compact_replay_segmented_prefill_tokens(
        request,
        num_computed_tokens=0,
        num_new_tokens=8,
    ) == 4

    request.num_computed_tokens = 4
    assert scheduler._advance_compact_replay_segmented_refill(
        request,
        reason="test-boundary",
    )

    assert compact_calls == [(1, 2)]
    assert request.prompt_token_ids == [1, 2, 5, 6, 7, 8]
    assert request._all_token_ids == [1, 2, 5, 6, 7, 8]
    assert request.num_prompt_tokens == 6
    assert request.num_computed_tokens == 2
    assert request.position_offset == 2
    assert request.needs_rebuild
    assert request._kve_compact_replay_segmented_deletions == []

    request.num_computed_tokens = 6
    assert scheduler._complete_compact_replay_full_refill(
        request,
        reason="test-complete",
    )
    assert compact_calls == [(1, 2)]
    assert request.prompt_token_ids == [1, 3, 4, 5]
    assert request._all_token_ids == [1, 3, 4, 5, 6, 7]
    assert request._output_token_ids == [6, 7]
    assert request.num_prompt_tokens == 4
    assert request.num_computed_tokens == 6
    assert request.position_offset == 128
    assert not request._kve_compact_replay_segmented_refill_active


def test_compact_replay_segmented_refill_compacts_at_final_boundary(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "segmented")
    scheduler = _scheduler_with_requests({"compact"})
    scheduler._compaction_max_turns = 0
    scheduler._worker_protected_prefix_len = lambda _req: 2
    scheduler.cache_config = SimpleNamespace(enable_prefix_caching=False)
    compact_calls = []

    def compact_request(_request_id, _prompt_tokens, *, explicit_block_range):
        compact_calls.append(explicit_block_range)
        start, end = explicit_block_range
        return (end - start) * 2

    scheduler._compact_replay_compaction_manager = lambda: SimpleNamespace(
        block_size=2,
        compact_request=compact_request,
    )
    request = _make_full_replay_request()
    replay_snapshot = SimpleNamespace(
        token_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        death_indices=(8, 8, 8, 8, 8, 8, 8, 8),
        live_writer_indices=(0, 1, 4, 5, 6, 7),
        evictions=1,
        final_writer_len=8,
    )

    assert scheduler._activate_compact_replay_segmented_refill(
        request,
        replay_snapshot,
        reason="test",
    )
    assert request._kve_compact_replay_segmented_deletions == [
        (8, 2, 4, 8)
    ]
    assert scheduler._cap_compact_replay_segmented_prefill_tokens(
        request,
        num_computed_tokens=0,
        num_new_tokens=8,
    ) == 8

    request.num_computed_tokens = 8
    assert scheduler._advance_compact_replay_segmented_refill(
        request,
        reason="test-final-boundary",
    )

    assert compact_calls == [(1, 2)]
    assert request.prompt_token_ids == [1, 2, 5, 6, 7, 8]
    assert request.num_computed_tokens == 6
    assert request._kve_compact_replay_segmented_deletions == []


def test_compact_replay_segmented_refill_compacts_nested_spans(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "segmented")
    scheduler = _scheduler_with_requests({"compact"})
    scheduler._compaction_max_turns = 0
    scheduler._worker_protected_prefix_len = lambda _req: 2
    scheduler.cache_config = SimpleNamespace(enable_prefix_caching=False)
    compact_calls = []

    def compact_request(_request_id, _prompt_tokens, *, explicit_block_range):
        compact_calls.append(explicit_block_range)
        start, end = explicit_block_range
        return (end - start) * 2

    scheduler._compact_replay_compaction_manager = lambda: SimpleNamespace(
        block_size=2,
        compact_request=compact_request,
    )
    request = _make_full_replay_request()
    replay_snapshot = SimpleNamespace(
        token_ids=tuple(range(12)),
        death_indices=(12, 12, 12, 12, 12, 12, 10, 10, 12, 12, 12, 12),
        live_writer_indices=(0, 1, 8, 9, 10, 11),
        evictions=2,
        final_writer_len=12,
    )

    assert scheduler._activate_compact_replay_segmented_refill(
        request,
        replay_snapshot,
        reason="test",
    )
    assert request._kve_compact_replay_segmented_deletions == [
        (10, 6, 8, 10),
        (10, 2, 6, 12),
    ]

    request.num_computed_tokens = 10
    assert scheduler._advance_compact_replay_segmented_refill(
        request,
        reason="test-nested-first",
    )
    assert compact_calls == [(3, 4)]
    assert request.num_computed_tokens == 8
    assert request.prompt_token_ids == [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]

    request.num_computed_tokens = 10
    assert scheduler._advance_compact_replay_segmented_refill(
        request,
        reason="test-nested-second",
    )
    assert compact_calls == [(3, 4), (1, 3)]
    assert request.num_computed_tokens == 6
    assert request.prompt_token_ids == [0, 1, 8, 9, 10, 11]
    assert request._kve_compact_replay_segmented_deletions == []


def test_compact_replay_segmented_rebuild_does_not_send_flex_mask(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "segmented")
    scheduler = _scheduler_with_requests({"compact"})
    scheduler._compaction_max_turns = 0
    scheduler._worker_protected_prefix_len = lambda _req: 2
    scheduler.cache_config = SimpleNamespace(enable_prefix_caching=False)
    scheduler._compact_replay_compaction_manager = lambda: SimpleNamespace(
        block_size=2,
        compact_request=lambda *_args, **_kwargs: 0,
    )
    request = _make_full_replay_request()
    replay_snapshot = SimpleNamespace(
        token_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        death_indices=(8, 8, 4, 4, 8, 8, 8, 8),
        live_writer_indices=(0, 1, 4, 5, 6, 7),
        evictions=1,
        final_writer_len=8,
    )
    assert scheduler._activate_compact_replay_segmented_refill(
        request,
        replay_snapshot,
        reason="test",
    )
    request.needs_rebuild = True

    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            get_blocks=lambda _req_id: ([SimpleNamespace(block_id=10)],),
            single_type_managers=[SimpleNamespace(block_size=2)],
        )
    )
    scheduler._managed_context_active_hidden_kv = lambda _req_id: ((), 0, [])
    scheduler.use_pp = False
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=False)

    cached = scheduler._make_cached_request_data(
        running_reqs=[request],
        resumed_reqs=[],
        num_scheduled_tokens={"compact": 4},
        spec_decode_tokens={},
        req_to_new_blocks={},
    )

    assert cached.rebuild_req_ids == {"compact"}
    assert cached.compact_replay_data == {}


def test_compact_replay_segmented_mode_does_not_fall_back_to_full(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_PREFIX_MISS_REPREFILL", "1")
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "segmented")
    scheduler = _scheduler_with_requests({"compact"})
    scheduler._compaction_max_turns = 0
    scheduler._worker_protected_prefix_len = lambda _req: 2
    scheduler._compact_replay_compaction_manager = lambda: SimpleNamespace(
        block_size=2,
    )
    freed_visible = []
    freed_encoder = []
    scheduler.kv_cache_manager = SimpleNamespace(
        free=lambda req: freed_visible.append(req.request_id),
    )
    scheduler.encoder_cache_manager = SimpleNamespace(
        free=lambda req: freed_encoder.append(req.request_id),
    )
    request = _make_full_replay_request()
    request.compact_replay_snapshot = lambda: SimpleNamespace(
        token_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        death_indices=(8, 8, 5, 5, 8, 8, 8, 8),
        live_writer_indices=(0, 1, 4, 5, 6, 7),
        evictions=1,
        final_writer_len=8,
    )

    assert not scheduler._phase4_requeue_prefix_miss_for_reprefill(
        request,
        "prefix miss",
    )

    assert freed_visible == []
    assert freed_encoder == []
    assert request.prompt_token_ids == [1, 3, 4, 5]
    assert request.num_computed_tokens == 8
    assert not getattr(
        request,
        "_kve_compact_replay_full_refill_active",
        False,
    )
    assert not hasattr(request, "_kve_compact_replay_full_refill_state")


def test_phase4_pin_release_reprefill_skips_prefix_expectation() -> None:
    request = _request("trace", expected_cached_tokens=64)

    assert not Scheduler._phase4_pin_release_reprefill_active(request)

    request._kve_phase4_reprefill_after_pin_release = True
    assert not Scheduler._phase4_pin_release_reprefill_active(request)

    request._kve_reprefill_after_flush = True
    assert Scheduler._phase4_pin_release_reprefill_active(request)


def test_phase4_prefix_miss_requeues_for_reprefill_when_explicitly_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_PREFIX_MISS_REPREFILL", "1")
    scheduler = _scheduler_with_requests({"reader"})
    request = _request("trace", expected_cached_tokens=64)
    request.prompt_token_ids = [1] * 512
    request.sampling_params.extra_args["kve_restore_span_ids"] = ["T0001"]
    request.spec_token_ids = [99]
    request.is_prefill_chunk = True
    request.num_computed_tokens = 256
    request.num_external_computed_tokens = 3
    request.num_cached_tokens = 256
    request.skip_reading_prefix_cache = False
    request.needs_rebuild = False

    scheduler._managed_context_restore_reservations["reader"] = {
        ("trace", "T0001"),
    }
    scheduler._pending_admission_compaction_ids.add("reader")
    scheduler.prev_step_scheduled_req_ids.add("reader")

    assert scheduler._phase4_requeue_prefix_miss_for_reprefill(
        request,
        "prefix miss",
    )
    assert request.num_computed_tokens == 0
    assert request.num_external_computed_tokens == 0
    assert request.num_cached_tokens == -1
    assert request.spec_token_ids == []
    assert not request.is_prefill_chunk
    assert request.needs_rebuild
    assert request.skip_reading_prefix_cache
    assert request._kve_reprefill_after_flush
    assert request._kve_phase4_reprefill_after_pin_release
    assert scheduler._managed_context_restore_reservations["reader"] == {
        ("trace", "T0001"),
    }
    assert "reader" not in scheduler._pending_admission_compaction_ids
    assert "reader" not in scheduler.prev_step_scheduled_req_ids


def test_phase4_prefix_miss_requeues_for_full_replay_when_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    scheduler = _scheduler_with_requests({"compact"})
    scheduler._compact_replay_compaction_manager = lambda: SimpleNamespace(
        block_size=2
    )
    request = _make_full_replay_request()
    replay_snapshot = SimpleNamespace(
        token_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        death_indices=(8, 8, 4, 4, 8, 8, 8, 8),
        live_writer_indices=(0, 1, 4, 5, 6, 7),
        evictions=1,
        final_writer_len=8,
    )
    request.compact_replay_snapshot = lambda: replay_snapshot
    freed_visible = []
    freed_encoder = []
    scheduler.kv_cache_manager = SimpleNamespace(
        free=lambda req: freed_visible.append(req.request_id)
    )
    scheduler.encoder_cache_manager = SimpleNamespace(
        free=lambda req: freed_encoder.append(req.request_id)
    )

    assert scheduler._phase4_requeue_prefix_miss_for_reprefill(
        request,
        "prefix miss",
    )
    assert request._kve_reprefill_after_flush
    assert request._kve_phase4_reprefill_after_pin_release
    assert request._kve_compact_replay_refill_active
    assert request._kve_compact_replay_full_refill_active
    assert request.num_prompt_tokens == 8
    assert request.position_offset == 0
    assert freed_visible == ["compact"]
    assert freed_encoder == ["compact"]


def test_phase4_prefix_miss_full_replay_uses_restore_span_xargs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    scheduler = _scheduler_with_requests({"compact"})
    scheduler._compact_replay_compaction_manager = lambda: SimpleNamespace(
        block_size=2
    )
    request = _make_full_replay_request()
    request.prompt_token_ids = [1, 2, 7, 8, 9]
    request.num_prompt_tokens = 5
    request._all_token_ids = [1, 2, 7, 8, 9]
    request.all_token_ids = ConstantList(request._all_token_ids)
    request._output_token_ids = []
    request.output_token_ids = ConstantList(request._output_token_ids)
    request.compact_replay_snapshot = lambda: None
    request.sampling_params.extra_args["kve_compact_replay_spans"] = json.dumps([
        {
            "span_id": "T0002",
            "evict_start": 2,
            "tokens_evicted": 2,
            "evicted_token_ids": [5, 6],
            "writer_len_at_compaction": 9,
            "original_turn_start": 2,
        },
        {
            "span_id": "T0001",
            "evict_start": 2,
            "tokens_evicted": 2,
            "evicted_token_ids": [3, 4],
            "writer_len_at_compaction": 7,
            "original_turn_start": 0,
        },
    ])
    freed_visible = []
    freed_encoder = []
    scheduler.kv_cache_manager = SimpleNamespace(
        free=lambda req: freed_visible.append(req.request_id)
    )
    scheduler.encoder_cache_manager = SimpleNamespace(
        free=lambda req: freed_encoder.append(req.request_id)
    )

    assert scheduler._phase4_requeue_prefix_miss_for_reprefill(
        request,
        "prefix miss",
    )

    snapshot = request._kve_compact_replay_refill_snapshot
    assert snapshot.token_ids == (1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert snapshot.death_indices == (9, 9, 7, 7, 9, 9, 9, 9, 9)
    assert snapshot.live_writer_indices == (0, 1, 6, 7, 8)
    assert snapshot.evictions == 2
    assert request._kve_compact_replay_refill_active
    assert request._kve_compact_replay_full_refill_active
    assert request.prompt_token_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert request.num_prompt_tokens == 9
    assert request.position_offset == 0
    assert freed_visible == ["compact"]
    assert freed_encoder == ["compact"]


def test_compact_replay_xargs_clamps_future_death_to_replay_prefix(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    scheduler = _scheduler_with_requests({"compact"})
    request = _make_full_replay_request()
    request.prompt_token_ids = [1, 2, 5, 6]
    request.num_prompt_tokens = 4
    request.sampling_params.extra_args["kve_compact_replay_spans"] = json.dumps([
        {
            "span_id": "T0001",
            "evict_start": 2,
            "tokens_evicted": 2,
            "evicted_token_ids": [3, 4],
            "writer_len_at_compaction": 10,
            "original_turn_start": 0,
        },
    ])

    snapshot = scheduler._compact_replay_snapshot_from_xargs(request)

    assert snapshot is not None
    assert snapshot.token_ids == (1, 2, 3, 4, 5, 6)
    assert snapshot.death_indices == (6, 6, 6, 6, 6, 6)
    assert snapshot.live_writer_indices == (0, 1, 4, 5)


def test_compact_replay_no_sample_completion_forces_decode() -> None:
    scheduler = _scheduler_with_requests({"compact"})
    request = _make_full_replay_request()
    request.num_tokens = len(request._all_token_ids)
    request.num_computed_tokens = request.num_tokens
    request.num_cached_tokens = request.num_tokens
    request.needs_rebuild = False
    request.skip_reading_prefix_cache = False
    scheduler.prev_step_scheduled_req_ids.add("compact")

    assert scheduler._force_compact_replay_decode_if_fully_computed(request)

    assert request.num_computed_tokens == request.num_tokens - 1
    assert request.num_cached_tokens == request.num_computed_tokens
    assert request.needs_rebuild
    assert request.skip_reading_prefix_cache
    assert "compact" not in scheduler.prev_step_scheduled_req_ids


def test_cpu_offloaded_restore_can_requeue_for_compact_replay(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    scheduler = _scheduler_with_requests({"compact"})
    block_pool = _FakeBlockPool(new_block_ids=list(range(10)))
    block_pool.num_gpu_blocks = 100
    manager = SimpleNamespace(block_size=2, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    request = _make_full_replay_request()
    request.prompt_token_ids = [1, 2, 7, 8, 9]
    request.num_prompt_tokens = 5
    request.sampling_params.extra_args["kve_compact_replay_spans"] = json.dumps([
        {
            "span_id": "T0001",
            "evict_start": 2,
            "tokens_evicted": 2,
            "evicted_token_ids": [3, 4],
            "writer_len_at_compaction": 7,
            "original_turn_start": 0,
        },
    ])
    span = _managed_span(
        "T0001",
        "trace",
        manager,
        _gpu_blocks(10, 2),
        status="cpu_offloaded",
    )
    span.cpu_block_ids_by_group = ([0, 1],)

    reason = scheduler._managed_context_restore_replay_prefill_reason(
        request,
        [span],
    )

    assert reason is not None
    assert "managed-context restore replay prefill" in reason


def test_managed_context_replay_only_archive_skips_kv_copy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    monkeypatch.setenv("KVE_MANAGED_CONTEXT_REPLAY_ONLY_ARCHIVE", "1")
    scheduler = _scheduler_with_requests({"writer"})
    scheduler._managed_context_next_span_by_trace = defaultdict(int)
    scheduler._managed_context_archive_ttl_seconds = 1800.0
    scheduler._managed_context_archive_max_blocks = None
    block_pool = _FakeBlockPool()
    manager = SimpleNamespace(block_size=16, block_pool=block_pool)
    request = _request("trace", request_id="writer")
    request._all_token_ids = list(range(128))
    request.position_offset = 64

    span_ids = scheduler._archive_managed_context_span(
        request,
        compaction_mgr=manager,
        evict_start=16,
        evict_end=48,
        explicit_block_range=(1, 3),
        last_turn_evicted=3,
        stride_used=2,
    )

    assert span_ids == ["T0001"]
    span = scheduler._managed_context_archive[("trace", "T0001")]
    assert span.status == "replay_only"
    assert span.entries == []
    assert span.kv_block_count == 0
    assert span.token_ids == list(range(16, 48))
    assert block_pool.touched_blocks == []
    assert scheduler._managed_context_store_events_to_submit == {}


def test_replay_only_restore_uses_compact_replay_xargs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    scheduler = _scheduler_with_requests({"compact"})
    scheduler.max_model_len = 16384
    scheduler._managed_context_archive_ttl_seconds = 1800.0
    scheduler._managed_context_archive_max_blocks = None
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[object()])
    )
    request = _make_full_replay_request()
    request.sampling_params.extra_args["kve_phase4_trace_id"] = "compact"
    request.sampling_params.extra_args["kve_restore_span_ids"] = ["T0001"]
    request.sampling_params.extra_args["kve_compact_replay_spans"] = json.dumps(
        [
            {
                "span_id": "T0001",
                "evict_start": 1,
                "tokens_evicted": 1,
                "evicted_token_ids": [2],
                "writer_len_at_compaction": 5,
                "original_turn_start": 0,
            }
        ]
    )
    span = ManagedContextSpan(
        span_id="T0001",
        trace_id="compact",
        request_id="writer",
        absolute_turn_start=0,
        absolute_turn_end=0,
        token_ids=[2],
        entries=[],
        kv_block_count=0,
        logical_start_by_group=[],
        position_offset_frame=0,
        evict_start=1,
        evict_end=2,
        created_at=time.monotonic(),
        status="replay_only",
    )
    scheduler._managed_context_archive = {("compact", "T0001"): span}
    scheduler._managed_context_archive_order = deque([("compact", "T0001")])

    spans, error = scheduler._validate_managed_context_restore_request(request)

    assert error is None
    assert spans == [span]
    assert scheduler._managed_context_restore_needs_cpu_load(spans)
    reason = scheduler._managed_context_restore_replay_prefill_reason(
        request,
        spans,
    )
    assert reason is not None
    assert "managed-context restore replay prefill" in reason


def test_replay_only_restore_requires_compact_replay_xargs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    scheduler = _scheduler_with_requests({"compact"})
    scheduler.max_model_len = 16384
    scheduler._managed_context_archive_ttl_seconds = 1800.0
    scheduler._managed_context_archive_max_blocks = None
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[object()])
    )
    request = _make_full_replay_request()
    request.sampling_params.extra_args["kve_phase4_trace_id"] = "compact"
    request.sampling_params.extra_args["kve_restore_span_ids"] = ["T0001"]
    span = ManagedContextSpan(
        span_id="T0001",
        trace_id="compact",
        request_id="writer",
        absolute_turn_start=0,
        absolute_turn_end=0,
        token_ids=[2],
        entries=[],
        kv_block_count=0,
        logical_start_by_group=[],
        position_offset_frame=0,
        evict_start=1,
        evict_end=2,
        created_at=time.monotonic(),
        status="replay_only",
    )
    scheduler._managed_context_archive = {("compact", "T0001"): span}
    scheduler._managed_context_archive_order = deque([("compact", "T0001")])

    spans, error = scheduler._validate_managed_context_restore_request(request)

    assert spans == []
    assert error is not None
    assert "compact replay metadata is missing" in error


def test_gpu_hot_restore_does_not_force_compact_replay(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_COMPACT_REPLAY_REFILL_MODE", "full")
    scheduler = _scheduler_with_requests({"compact"})
    block_pool = _FakeBlockPool(new_block_ids=list(range(10)))
    block_pool.num_gpu_blocks = 100
    manager = SimpleNamespace(block_size=2, block_pool=block_pool)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    request = _make_full_replay_request()
    request.sampling_params.extra_args["kve_compact_replay_spans"] = json.dumps([
        {
            "span_id": "T0001",
            "evict_start": 2,
            "tokens_evicted": 2,
            "evicted_token_ids": [3, 4],
            "writer_len_at_compaction": 7,
            "original_turn_start": 0,
        },
    ])
    span = _managed_span(
        "T0001",
        "trace",
        manager,
        _gpu_blocks(10, 2),
        status="cpu_hot",
    )

    assert (
        scheduler._managed_context_restore_replay_prefill_reason(
            request,
            [span],
        )
        is None
    )


def test_phase4_prefix_miss_reprefill_is_disabled_by_default() -> None:
    scheduler = _scheduler_with_requests({"reader"})
    request = _request("trace", expected_cached_tokens=64)
    request.prompt_token_ids = [1] * 512

    assert not scheduler._phase4_requeue_prefix_miss_for_reprefill(
        request,
        "prefix miss",
    )
    assert not request._kve_reprefill_after_flush


def test_managed_context_restore_preempt_releases_hidden_state_for_reprefill() -> None:
    scheduler = _scheduler_with_requests({"restore"})
    request = _replayable_request("restore", position_offset=0)
    scheduler.requests = {"restore": request}
    scheduler._managed_context_active_restores = {"restore": object()}
    scheduler._managed_context_deferred_restores = {}
    freed_visible = []
    freed_encoder = []
    released = []

    scheduler.kv_cache_manager = SimpleNamespace(
        free=lambda req: freed_visible.append(req.request_id)
    )
    scheduler.encoder_cache_manager = SimpleNamespace(
        free=lambda req: freed_encoder.append(req.request_id)
    )
    scheduler._release_managed_context_active_restore = (
        lambda req_id, reason: released.append(("active", req_id, reason))
    )
    scheduler._release_managed_context_deferred_restore = (
        lambda req_id, reason: released.append(("deferred", req_id, reason))
    )
    scheduler._release_managed_context_restore_reservation = (
        lambda req_id, reason: released.append(("reserve", req_id, reason))
    )

    scheduler._preempt_request(request, time.monotonic())

    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    assert request.needs_rebuild
    assert request.skip_reading_prefix_cache
    assert request._kve_reprefill_after_flush
    assert list(scheduler.waiting) == [request]
    assert "restore" in scheduler.requests
    assert freed_visible == ["restore"]
    assert freed_encoder == ["restore"]
    assert released == [
        ("active", "restore", "reprefill-preempt"),
        ("deferred", "restore", "reprefill-preempt"),
        ("reserve", "restore", "reprefill-preempt"),
    ]
