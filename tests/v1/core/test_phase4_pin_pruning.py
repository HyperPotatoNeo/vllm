import time
from collections import deque
from types import SimpleNamespace

from vllm.v1.outputs import ManagedContextTransferOutput
from vllm.v1.core.sched.scheduler import Phase4Pin, Scheduler
from vllm.v1.request import RequestStatus


class _FakeRequestQueue(deque):
    def prepend_request(self, request) -> None:
        self.appendleft(request)


class _FakeBlockPool:
    def __init__(self, new_block_ids: list[int] | None = None) -> None:
        self.freed_blocks = []
        self.evicted_ids = []
        self._new_block_ids = deque(new_block_ids or [])

    def free_blocks(self, blocks) -> None:
        self.freed_blocks.extend(list(blocks))

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
) -> Phase4Pin:
    return Phase4Pin(
        entries=[] if entries is None else entries,
        token_count=token_count,
        block_count=0,
        request_id="writer",
        call_idx=1,
        created_at=time.monotonic() - 100.0,
        consumed_by_request_id=consumed_by_request_id,
        consumed_at=consumed_at,
    )


def test_phase4_consumed_pin_is_kept_while_consumer_is_live() -> None:
    scheduler = _scheduler_with_requests({"reader"})
    pin = _pin(consumed_by_request_id="reader", consumed_at=time.monotonic() - 60.0)

    assert not scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=1800.0
    )


def test_phase4_consumed_pin_prunes_after_consumer_grace(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    pin = _pin(consumed_by_request_id="reader", consumed_at=time.monotonic() - 2.0)

    assert scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=1800.0
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


def test_phase4_consumed_pin_ignores_queued_request_without_expected_tokens(
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

    assert scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=1800.0
    )


def test_phase4_consumed_pin_ignores_unsatisfied_queued_successor(
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

    assert scheduler._phase4_pin_is_prunable(
        "trace", pin, time.monotonic(), ttl_seconds=1800.0
    )


def test_phase4_consumed_pin_ttl_zero_still_prunes_after_grace(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS", "1")
    scheduler = _scheduler_with_requests(set())
    pin = _pin(
        consumed_by_request_id="reader",
        consumed_at=time.monotonic() - 2.0,
    )

    assert scheduler._phase4_pin_is_prunable(
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
