# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from types import SimpleNamespace

import pytest

import vllm.v1.core.sched.scheduler as scheduler_mod
from vllm.v1.outputs import ManagedContextTransferOutput
from vllm.v1.core.sched.scheduler import (
    ManagedContextHotGPUStats,
    ManagedContextSpan,
    Scheduler,
    _managed_context_cpu_reload_block_demand,
    _managed_context_gpu_reload_wait_reason,
    _pop_contiguous_managed_context_cpu_blocks,
)
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def test_pop_contiguous_managed_context_cpu_blocks_prefers_lowest_run() -> None:
    free = deque([5, 2, 3, 4, 9, 10])

    assert _pop_contiguous_managed_context_cpu_blocks(free, 3) == [2, 3, 4]
    assert list(free) == [5, 9, 10]


def test_pop_contiguous_managed_context_cpu_blocks_removes_non_adjacent_run() -> None:
    free = deque([8, 1, 6, 2, 3])

    assert _pop_contiguous_managed_context_cpu_blocks(free, 3) == [1, 2, 3]
    assert list(free) == [8, 6]


def test_pop_contiguous_managed_context_cpu_blocks_returns_none_if_fragmented() -> None:
    free = deque([0, 2, 4, 6])

    assert _pop_contiguous_managed_context_cpu_blocks(free, 2) is None
    assert list(free) == [0, 2, 4, 6]


def test_pop_contiguous_managed_context_cpu_blocks_rejects_invalid_size() -> None:
    free = deque([0, 1, 2])

    assert _pop_contiguous_managed_context_cpu_blocks(free, 0) is None
    assert _pop_contiguous_managed_context_cpu_blocks(free, 4) is None
    assert list(free) == [0, 1, 2]


def _span(span_id: str, status: str, blocks: int) -> ManagedContextSpan:
    return ManagedContextSpan(
        span_id=span_id,
        trace_id="trace",
        request_id="request",
        absolute_turn_start=0,
        absolute_turn_end=0,
        token_ids=[],
        entries=[],
        kv_block_count=blocks,
        logical_start_by_group=[],
        position_offset_frame=0,
        evict_start=0,
        evict_end=0,
        created_at=0.0,
        status=status,
    )


def test_managed_context_cpu_reload_block_demand_counts_only_cold_spans() -> None:
    spans = [
        _span("T0001", "cpu_offloaded", 3),
        _span("T0002", "gpu_pinned", 7),
        _span("T0003", "cpu_hot", 5),
        _span("T0004", "cpu_offloaded", 2),
    ]

    assert _managed_context_cpu_reload_block_demand(spans) == 5


def test_managed_context_gpu_reload_wait_reason_accounts_extra_blocks() -> None:
    assert (
        _managed_context_gpu_reload_wait_reason(
            reload_blocks=4,
            extra_blocks=3,
            free_blocks=7,
        )
        is None
    )

    reason = _managed_context_gpu_reload_wait_reason(
        reload_blocks=4,
        extra_blocks=3,
        free_blocks=6,
    )

    assert reason is not None
    assert "required=7" in reason
    assert "free=6" in reason


class _Request(SimpleNamespace):
    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)


class _FakeBlockPool:
    def __init__(self, *, free_blocks: int, new_block_ids: list[int]) -> None:
        self._free_blocks = free_blocks
        self._new_block_ids = deque(new_block_ids)
        self.freed_block_ids: list[int] = []

    def get_num_free_blocks(self) -> int:
        return self._free_blocks

    def get_new_blocks(self, count: int) -> list[SimpleNamespace]:
        if count > self._free_blocks:
            raise ValueError("not enough free blocks")
        self._free_blocks -= count
        return [
            SimpleNamespace(
                block_id=self._new_block_ids.popleft(),
                logical_start=-1,
                is_null=False,
            )
            for _ in range(count)
        ]

    def free_blocks(self, blocks) -> None:
        block_list = list(blocks)
        self.freed_block_ids.extend(int(block.block_id) for block in block_list)
        self._free_blocks += len(block_list)


class _FakeRequestQueue(deque):
    def prepend_request(self, request) -> None:
        self.appendleft(request)

    def remove_requests(self, requests) -> None:
        for request in requests:
            try:
                self.remove(request)
            except ValueError:
                pass


def _request_kv_swap_test_scheduler(
    monkeypatch,
    *,
    gpu_free_blocks: int = 4,
) -> tuple[Scheduler, SimpleNamespace, _Request]:
    monkeypatch.setenv("KVE_REQUEST_KV_SWAP", "1")
    scheduler = object.__new__(Scheduler)
    block_pool = _FakeBlockPool(
        free_blocks=gpu_free_blocks,
        new_block_ids=[20, 21, 22, 23],
    )
    blocks = [
        SimpleNamespace(block_id=10, logical_start=0, is_null=False),
        SimpleNamespace(block_id=11, logical_start=16, is_null=False),
    ]
    manager = SimpleNamespace(
        block_size=16,
        block_pool=block_pool,
        req_to_blocks={"req": blocks},
        num_cached_block={"req": 1},
    )
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager])
    )
    scheduler._managed_context_enabled = False
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_offload_immediate = False
    scheduler._managed_context_cpu_evict_on_capacity = False
    scheduler._managed_context_cpu_free_block_ids = deque([0, 1, 2, 3])
    scheduler._managed_context_cpu_max_blocks = 4
    scheduler._managed_context_archive = {}
    scheduler._managed_context_archive_order = deque()
    scheduler._managed_context_active_restores = {}
    scheduler._managed_context_deferred_restores = {}
    scheduler._managed_context_restore_reservations = {}
    scheduler._managed_context_next_transfer_event_id = 0
    scheduler._managed_context_store_events_to_submit = {}
    scheduler._managed_context_load_events_to_submit = {}
    scheduler._managed_context_store_event_to_span = {}
    scheduler._managed_context_load_event_to_request_id = {}
    scheduler._managed_context_pending_loads = {}
    scheduler._managed_context_finished_load_req_ids = set()
    scheduler._managed_context_cpu_max_pending_store_events = 0
    scheduler._managed_context_cpu_max_pending_load_events = 0
    scheduler.policy = scheduler_mod.SchedulingPolicy.FCFS
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_model_len = 4096
    scheduler.num_lookahead_tokens = 0
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler._request_kv_swaps = {}
    scheduler._request_kv_swap_ready_queue = deque()
    scheduler._request_kv_swap_store_event_to_request_id = {}
    scheduler._request_kv_swap_load_event_to_request_id = {}
    scheduler._request_kv_swap_finished_load_req_ids = set()
    scheduler._request_kv_swap_suspended = False
    scheduler.requests = {}
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.running = []
    scheduler.waiting = _FakeRequestQueue()
    scheduler.skipped_waiting = _FakeRequestQueue()
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.connector = None
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda request: None)
    scheduler._connector_finished = lambda request: (False, None)
    scheduler.log_stats = False
    request = _Request(
        request_id="req",
        client_index=0,
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        padding_pending=False,
        num_output_placeholders=0,
        prompt_token_ids=[1] * 64,
        num_prompt_tokens=64,
        num_tokens=80,
        num_computed_tokens=64,
        num_external_computed_tokens=0,
        num_cached_tokens=0,
        position_offset=128,
        needs_rebuild=False,
        compaction_events=[SimpleNamespace()],
        is_prefill_chunk=False,
        skip_reading_prefix_cache=False,
        _kve_reprefill_after_flush=False,
        _kve_phase4_reprefill_after_pin_release=False,
        spec_token_ids=[],
        num_preemptions=0,
    )
    scheduler.requests[request.request_id] = request
    return scheduler, manager, request


def _make_request_running_for_preempt_test(request: _Request) -> None:
    request.status = RequestStatus.RUNNING
    request.num_cached_tokens = 64
    request.spec_token_ids = [99]


def test_compacted_preempt_request_kv_swap_store_queue_full_defers_without_reprefill(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_REQUEST_KV_SWAP_MAX_PENDING_STORES", "1")
    scheduler, manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    _make_request_running_for_preempt_test(request)
    scheduler._request_kv_swaps["other"] = SimpleNamespace(status="store_pending")

    def fail_reprefill(*args, **kwargs):
        raise AssertionError("compact re-prefill must not run")

    scheduler._preempt_request_for_reprefill = fail_reprefill

    result = scheduler._preempt_request(request, 123.0)

    assert result.kind == "deferred"
    assert result.error == "request KV swap store queue is full"
    assert request.status == RequestStatus.RUNNING
    assert request.num_preemptions == 0
    assert request.num_computed_tokens == 64
    assert request.num_cached_tokens == 64
    assert request.position_offset == 128
    assert request.spec_token_ids == [99]
    assert not request.needs_rebuild
    assert not request.skip_reading_prefix_cache
    assert not request._kve_reprefill_after_flush
    assert list(scheduler.waiting) == []
    assert set(scheduler._request_kv_swaps) == {"other"}
    assert scheduler._managed_context_store_events_to_submit == {}
    assert [block.block_id for block in manager.req_to_blocks["req"]] == [10, 11]
    assert manager.num_cached_block == {"req": 1}
    assert manager.block_pool.freed_block_ids == []
    assert list(scheduler._managed_context_cpu_free_block_ids) == [0, 1, 2, 3]


def test_compacted_preempt_request_kv_swap_cpu_capacity_defers_without_reprefill(
    monkeypatch,
) -> None:
    scheduler, manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    _make_request_running_for_preempt_test(request)
    scheduler._managed_context_cpu_free_block_ids = deque([0])
    scheduler._managed_context_cpu_max_blocks = 1

    def fail_reprefill(*args, **kwargs):
        raise AssertionError("compact re-prefill must not run")

    scheduler._preempt_request_for_reprefill = fail_reprefill

    result = scheduler._preempt_request(request, 123.0)

    assert result.kind == "deferred"
    assert result.error == "request KV swap needs 2 CPU blocks, available=1 max=1"
    assert request.status == RequestStatus.RUNNING
    assert request.num_preemptions == 0
    assert request.num_computed_tokens == 64
    assert request.num_cached_tokens == 64
    assert request.position_offset == 128
    assert request.spec_token_ids == [99]
    assert list(scheduler.waiting) == []
    assert scheduler._request_kv_swaps == {}
    assert scheduler._managed_context_store_events_to_submit == {}
    assert [block.block_id for block in manager.req_to_blocks["req"]] == [10, 11]
    assert manager.num_cached_block == {"req": 1}
    assert manager.block_pool.freed_block_ids == []
    assert list(scheduler._managed_context_cpu_free_block_ids) == [0]


def test_compacted_preempt_current_step_request_defers_without_async_swap(
    monkeypatch,
) -> None:
    scheduler, manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    _make_request_running_for_preempt_test(request)

    def fail_reprefill(*args, **kwargs):
        raise AssertionError("compact re-prefill must not run")

    scheduler._preempt_request_for_reprefill = fail_reprefill

    result = scheduler._preempt_request(
        request,
        123.0,
        allow_async_kv_swap=False,
    )

    assert result.kind == "deferred"
    assert result.error == "request was scheduled in the current scheduler step"
    assert request.status == RequestStatus.RUNNING
    assert request.num_preemptions == 0
    assert request.spec_token_ids == [99]
    assert list(scheduler.waiting) == []
    assert scheduler._request_kv_swaps == {}
    assert scheduler._managed_context_store_events_to_submit == {}
    assert [block.block_id for block in manager.req_to_blocks["req"]] == [10, 11]
    assert manager.block_pool.freed_block_ids == []
    assert list(scheduler._managed_context_cpu_free_block_ids) == [0, 1, 2, 3]


def test_compacted_preempt_request_kv_swap_success_is_async_pending(
    monkeypatch,
) -> None:
    scheduler, manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    _make_request_running_for_preempt_test(request)

    result = scheduler._preempt_request(request, 123.0)

    assert result.kind == "async_pending"
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_preemptions == 1
    assert request.spec_token_ids == []
    assert not request._kve_reprefill_after_flush
    assert list(scheduler.waiting) == [request]

    swap = scheduler._request_kv_swaps["req"]
    assert swap.status == "store_pending"
    assert swap.store_event_id == 0
    assert swap.kv_block_count == 2
    assert swap.num_computed_tokens == 64
    assert swap.position_offset == 128
    assert swap.cpu_block_ids_by_group == ([0, 1],)
    assert swap.logical_start_by_group == ([0, 16],)

    assert [block.block_id for block in manager.req_to_blocks["req"]] == [10, 11]
    assert manager.num_cached_block == {"req": 1}
    assert manager.block_pool.freed_block_ids == []
    assert list(scheduler._managed_context_cpu_free_block_ids) == [2, 3]

    metadata = scheduler._drain_managed_context_transfer_metadata()
    assert metadata is not None
    assert len(metadata.store_events) == 1
    assert metadata.store_events[0].event_id == 0
    assert metadata.store_events[0].gpu_block_ids == [10, 11]
    assert metadata.store_events[0].cpu_block_ids == [0, 1]


def test_request_kv_swap_store_completion_frees_gpu_and_queues_load(
    monkeypatch,
) -> None:
    scheduler, manager, request = _request_kv_swap_test_scheduler(monkeypatch)

    assert scheduler._start_request_kv_swap_out(request, "test") is None
    assert manager.req_to_blocks["req"][0].block_id == 10
    metadata = scheduler._drain_managed_context_transfer_metadata()

    assert metadata is not None
    assert [event.event_id for event in metadata.store_events] == [0]

    scheduler._update_from_managed_context_transfer_finished(
        ManagedContextTransferOutput(completed_store_event_ids=[0])
    )

    swap = scheduler._request_kv_swaps["req"]
    assert swap.status == "swapped"
    assert list(scheduler._request_kv_swap_ready_queue) == ["req"]
    assert "req" not in manager.req_to_blocks
    assert "req" not in manager.num_cached_block
    assert manager.block_pool.freed_block_ids == [11, 10]
    assert list(scheduler._managed_context_cpu_free_block_ids) == [2, 3]


def test_request_kv_swap_load_respects_min_free_gpu_blocks(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_REQUEST_KV_SWAP_MIN_FREE_GPU_BLOCKS", "1")
    scheduler, manager, request = _request_kv_swap_test_scheduler(
        monkeypatch, gpu_free_blocks=0
    )
    assert scheduler._start_request_kv_swap_out(request, "test") is None
    scheduler._drain_managed_context_transfer_metadata()
    scheduler._update_from_managed_context_transfer_finished(
        ManagedContextTransferOutput(completed_store_event_ids=[0])
    )

    error = scheduler._start_request_kv_swap_load(
        request,
        extra_required_gpu_blocks=0,
    )

    assert error is not None
    assert "reload_blocks=2" in error
    assert "min_free=1" in error
    assert "required=3" in error
    assert scheduler._request_kv_swaps["req"].status == "swapped"
    assert list(scheduler._request_kv_swap_ready_queue) == ["req"]
    assert "req" not in manager.req_to_blocks
    assert scheduler._managed_context_load_events_to_submit == {}
    assert manager.block_pool.get_num_free_blocks() == 2


def test_request_kv_swap_load_accounts_resumed_allocation_blocks(
    monkeypatch,
) -> None:
    scheduler, manager, request = _request_kv_swap_test_scheduler(
        monkeypatch, gpu_free_blocks=0
    )
    assert scheduler._start_request_kv_swap_out(request, "test") is None
    scheduler._drain_managed_context_transfer_metadata()
    scheduler._update_from_managed_context_transfer_finished(
        ManagedContextTransferOutput(completed_store_event_ids=[0])
    )

    error = scheduler._start_request_kv_swap_load(
        request,
        extra_required_gpu_blocks=1,
    )

    assert error is not None
    assert "extra_blocks=1" in error
    assert "required=3" in error
    assert scheduler._request_kv_swaps["req"].status == "swapped"
    assert "req" not in manager.req_to_blocks
    assert scheduler._managed_context_load_events_to_submit == {}

    manager.block_pool._free_blocks = 3
    assert (
        scheduler._start_request_kv_swap_load(
            request,
            extra_required_gpu_blocks=1,
        )
        is None
    )

    assert scheduler._request_kv_swaps["req"].status == "load_pending"
    assert list(scheduler._request_kv_swap_ready_queue) == []
    assert [block.block_id for block in manager.req_to_blocks["req"]] == [20, 21]
    assert manager.block_pool.get_num_free_blocks() == 1
    assert list(scheduler._managed_context_load_events_to_submit) == [1]


def test_request_kv_swap_load_completion_restores_request_and_frees_cpu(
    monkeypatch,
) -> None:
    scheduler, manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    assert scheduler._start_request_kv_swap_out(request, "test") is None
    scheduler._drain_managed_context_transfer_metadata()
    scheduler._update_from_managed_context_transfer_finished(
        ManagedContextTransferOutput(completed_store_event_ids=[0])
    )

    assert scheduler._try_progress_request_kv_swap(request) is False
    swap = scheduler._request_kv_swaps["req"]
    assert swap.status == "load_pending"
    assert [block.block_id for block in manager.req_to_blocks["req"]] == [20, 21]
    assert [block.logical_start for block in manager.req_to_blocks["req"]] == [
        0,
        16,
    ]
    metadata = scheduler._drain_managed_context_transfer_metadata()

    assert metadata is not None
    assert [event.event_id for event in metadata.load_events] == [1]

    scheduler._update_from_managed_context_transfer_finished(
        ManagedContextTransferOutput(completed_load_event_ids=[1])
    )

    assert scheduler._try_progress_request_kv_swap(request) is True
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 64
    assert request.position_offset == 128
    assert request.num_cached_tokens == 64
    assert request.needs_rebuild
    assert "req" not in scheduler._request_kv_swaps
    assert list(scheduler._managed_context_cpu_free_block_ids) == [2, 3, 0, 1]


def test_request_kv_swap_load_waits_for_pending_load_budget(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_REQUEST_KV_SWAP_MAX_PENDING_LOADS", "1")
    scheduler, _manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    assert scheduler._start_request_kv_swap_out(request, "test") is None
    scheduler._drain_managed_context_transfer_metadata()
    scheduler._update_from_managed_context_transfer_finished(
        ManagedContextTransferOutput(completed_store_event_ids=[0])
    )
    scheduler._request_kv_swaps["other"] = SimpleNamespace(status="load_pending")

    assert scheduler._try_progress_request_kv_swap(request) is False
    assert scheduler._request_kv_swaps["req"].status == "swapped"
    assert scheduler._managed_context_load_events_to_submit == {}


def test_request_kv_swap_load_respects_shared_managed_load_budget(
    monkeypatch,
) -> None:
    scheduler, _manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    assert scheduler._start_request_kv_swap_out(request, "test") is None
    scheduler._drain_managed_context_transfer_metadata()
    scheduler._update_from_managed_context_transfer_finished(
        ManagedContextTransferOutput(completed_store_event_ids=[0])
    )
    scheduler._managed_context_cpu_max_pending_load_events = 1
    scheduler._managed_context_load_event_to_request_id = {99: "restore"}

    assert scheduler._try_progress_request_kv_swap(request) is False
    assert scheduler._request_kv_swaps["req"].status == "swapped"
    assert scheduler._managed_context_load_events_to_submit == {}


def test_request_kv_swap_pressure_swaps_running_request(monkeypatch) -> None:
    monkeypatch.setenv("KVE_REQUEST_KV_SWAP_GPU_PRESSURE_BLOCKS", "1")
    scheduler, manager, request = _request_kv_swap_test_scheduler(
        monkeypatch, gpu_free_blocks=0
    )
    _make_request_running_for_preempt_test(request)
    scheduler.running = [request]

    preempted = scheduler._preempt_request_for_kv_swap_pressure(
        123.0,
        reason="test-pressure",
        protected_request_ids=set(),
    )

    assert preempted is request
    assert scheduler.running == []
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_preemptions == 1
    assert list(scheduler.waiting) == [request]
    assert scheduler._request_kv_swaps["req"].status == "store_pending"
    assert [block.block_id for block in manager.req_to_blocks["req"]] == [10, 11]
    assert list(scheduler._managed_context_store_events_to_submit) == [0]


def test_request_kv_swap_pressure_skips_protected_request(monkeypatch) -> None:
    monkeypatch.setenv("KVE_REQUEST_KV_SWAP_GPU_PRESSURE_BLOCKS", "1")
    scheduler, manager, request = _request_kv_swap_test_scheduler(
        monkeypatch, gpu_free_blocks=0
    )
    _make_request_running_for_preempt_test(request)
    scheduler.running = [request]

    preempted = scheduler._preempt_request_for_kv_swap_pressure(
        123.0,
        reason="test-pressure",
        protected_request_ids={"req"},
    )

    assert preempted is None
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING
    assert scheduler._request_kv_swaps == {}
    assert scheduler._managed_context_store_events_to_submit == {}
    assert [block.block_id for block in manager.req_to_blocks["req"]] == [10, 11]


def test_request_kv_swap_ready_queue_discards_stale_ids(monkeypatch) -> None:
    scheduler, _manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    scheduler._request_kv_swap_ready_queue.extend(["missing", "req"])
    scheduler._request_kv_swaps["req"] = SimpleNamespace(status="swapped")

    assert scheduler._request_kv_swap_ready_head() == "req"
    assert list(scheduler._request_kv_swap_ready_queue) == ["req"]

    request.status = RequestStatus.FINISHED_ABORTED
    assert scheduler._request_kv_swap_ready_head() is None
    assert list(scheduler._request_kv_swap_ready_queue) == []


def test_request_kv_swap_ready_waiting_is_prioritized(monkeypatch) -> None:
    scheduler, _manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    other = _Request(request_id="other", status=RequestStatus.WAITING)
    scheduler.requests["other"] = other
    scheduler.waiting = scheduler_mod.create_request_queue(
        scheduler_mod.SchedulingPolicy.FCFS
    )
    scheduler.skipped_waiting = scheduler_mod.create_request_queue(
        scheduler_mod.SchedulingPolicy.FCFS
    )
    scheduler.skipped_waiting.add_request(other)
    scheduler.waiting.add_request(request)
    scheduler._request_kv_swap_ready_queue.append("req")
    scheduler._request_kv_swaps["req"] = SimpleNamespace(status="swapped")

    scheduler._request_kv_swap_prioritize_ready_waiting()

    assert scheduler.skipped_waiting.peek_request() is request
    assert list(scheduler.skipped_waiting)[1:] == [other]
    assert list(scheduler.waiting) == []


def test_request_kv_swap_resident_first_prefers_ordinary_waiting(
    monkeypatch,
) -> None:
    scheduler, _manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    other = _Request(request_id="other", status=RequestStatus.WAITING)
    scheduler.requests["other"] = other
    scheduler.waiting = scheduler_mod.create_request_queue(
        scheduler_mod.SchedulingPolicy.FCFS
    )
    scheduler.skipped_waiting = scheduler_mod.create_request_queue(
        scheduler_mod.SchedulingPolicy.FCFS
    )
    scheduler.skipped_waiting.add_request(request)
    scheduler.waiting.add_request(other)
    scheduler._request_kv_swaps["req"] = SimpleNamespace(
        status="swapped",
        created_at=time.monotonic(),
        kv_block_count=2,
        logical_start_by_group=([0, 16],),
        num_computed_tokens=request.num_computed_tokens,
        last_error=None,
    )

    assert scheduler._select_waiting_queue_for_scheduling() is scheduler.waiting

    monkeypatch.setenv("KVE_REQUEST_KV_SWAP_RESIDENT_FIRST", "0")
    assert (
        scheduler._select_waiting_queue_for_scheduling()
        is scheduler.skipped_waiting
    )


def test_request_kv_swap_resident_first_parks_while_running(
    monkeypatch,
) -> None:
    scheduler, _manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    scheduler.running = [_Request(request_id="running", status=RequestStatus.RUNNING)]
    scheduler._request_kv_swaps["req"] = SimpleNamespace(
        status="swapped",
        created_at=time.monotonic(),
        kv_block_count=2,
        logical_start_by_group=([0, 16],),
        num_computed_tokens=request.num_computed_tokens,
        last_error=None,
    )

    assert scheduler._request_kv_swap_should_park_waiting_request(
        request,
        token_budget=128,
    )
    assert "parked behind GPU-resident work" in (
        scheduler._request_kv_swaps["req"].last_error
    )


def test_request_kv_swap_resident_first_starvation_allows_reload(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_REQUEST_KV_SWAP_RELOAD_STARVATION_SECONDS", "0")
    scheduler, _manager, request = _request_kv_swap_test_scheduler(
        monkeypatch,
        gpu_free_blocks=8,
    )
    scheduler.running = [_Request(request_id="running", status=RequestStatus.RUNNING)]
    scheduler._request_kv_swaps["req"] = SimpleNamespace(
        status="swapped",
        created_at=time.monotonic(),
        kv_block_count=2,
        logical_start_by_group=([0, 16],),
        num_computed_tokens=request.num_computed_tokens,
        last_error=None,
    )

    assert not scheduler._request_kv_swap_should_park_waiting_request(
        request,
        token_budget=128,
    )


def test_request_kv_swap_resident_first_starvation_waits_for_gpu_room(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KVE_REQUEST_KV_SWAP_RELOAD_STARVATION_SECONDS", "0")
    monkeypatch.setenv("KVE_REQUEST_KV_SWAP_GPU_HEADROOM_BLOCKS", "4")
    scheduler, _manager, request = _request_kv_swap_test_scheduler(
        monkeypatch,
        gpu_free_blocks=2,
    )
    scheduler.running = [_Request(request_id="running", status=RequestStatus.RUNNING)]
    scheduler._request_kv_swaps["req"] = SimpleNamespace(
        status="swapped",
        created_at=time.monotonic(),
        kv_block_count=2,
        logical_start_by_group=([0, 16],),
        num_computed_tokens=request.num_computed_tokens,
        last_error=None,
    )

    assert scheduler._request_kv_swap_should_park_waiting_request(
        request,
        token_budget=128,
    )


def test_finish_request_kv_swap_in_flight_store_releases_on_completion(
    monkeypatch,
) -> None:
    scheduler, manager, request = _request_kv_swap_test_scheduler(monkeypatch)
    assert scheduler._start_request_kv_swap_out(request, "test") is None
    scheduler._drain_managed_context_transfer_metadata()

    scheduler.finish_requests("req", RequestStatus.FINISHED_ABORTED)

    assert "req" not in scheduler.requests
    assert scheduler._request_kv_swaps["req"].status == "expired"
    assert manager.block_pool.freed_block_ids == []

    scheduler._update_from_managed_context_transfer_finished(
        ManagedContextTransferOutput(completed_store_event_ids=[0])
    )

    assert "req" not in scheduler._request_kv_swaps
    assert "req" not in manager.req_to_blocks
    assert manager.block_pool.freed_block_ids == [11, 10]
    assert list(scheduler._managed_context_cpu_free_block_ids) == [2, 3, 0, 1]


def _scheduler_for_cpu_capacity_pressure(
    *,
    evict_on_capacity: bool,
) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    key = ("trace", "T0001")
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_blocks = 1
    scheduler._managed_context_cpu_free_block_ids = deque()
    scheduler._managed_context_archive_order = deque([key])
    scheduler._managed_context_archive = {key: _span("T0001", "cpu_offloaded", 1)}
    scheduler._managed_context_restore_reservations = {}
    scheduler._managed_context_cpu_evict_on_capacity = evict_on_capacity
    return scheduler


def test_managed_context_cpu_alloc_evicts_when_enabled() -> None:
    scheduler = _scheduler_for_cpu_capacity_pressure(evict_on_capacity=True)
    released = []

    def fake_release_managed_context_span(key, reason) -> None:
        released.append((key, reason))
        scheduler._managed_context_archive.pop(key, None)
        scheduler._managed_context_cpu_free_block_ids.append(0)

    scheduler._release_managed_context_span = fake_release_managed_context_span

    assert scheduler._alloc_managed_context_cpu_blocks(1) == [0]
    assert released == [(("trace", "T0001"), "cpu-block-limit")]


def test_managed_context_cpu_alloc_can_preserve_archive_on_capacity() -> None:
    scheduler = _scheduler_for_cpu_capacity_pressure(evict_on_capacity=False)
    released = []

    def fake_release_managed_context_span(key, reason) -> None:
        released.append((key, reason))
        scheduler._managed_context_archive.pop(key, None)
        scheduler._managed_context_cpu_free_block_ids.append(0)

    scheduler._release_managed_context_span = fake_release_managed_context_span

    assert scheduler._alloc_managed_context_cpu_blocks(1) is None
    assert released == []
    assert ("trace", "T0001") in scheduler._managed_context_archive


def _scheduler_for_archive_capacity_skip() -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler._managed_context_enabled = True
    scheduler._managed_context_cpu_offload_immediate = True
    scheduler._managed_context_cpu_evict_on_capacity = False
    scheduler._managed_context_skip_compaction_on_cpu_capacity = True
    scheduler._managed_context_cpu_free_block_ids = deque()
    scheduler._managed_context_cpu_max_blocks = 1
    scheduler._managed_context_archive_max_blocks = None
    scheduler._managed_context_archive = {}
    scheduler._managed_context_archive_order = deque()
    scheduler._managed_context_restore_reservations = {}
    scheduler._managed_context_active_restores = {}
    scheduler._managed_context_deferred_restores = {}
    scheduler._managed_context_pending_loads = {}
    scheduler._managed_context_store_event_to_span = {}
    scheduler._managed_context_load_event_to_request_id = {}
    scheduler._managed_context_cpu_max_pending_store_events = 0
    scheduler._managed_context_cpu_max_pending_load_events = 0
    scheduler._managed_context_hot_gpu_stats = ManagedContextHotGPUStats(
        budget_blocks=0
    )
    scheduler._phase4_pinned_blocks = {}
    scheduler._managed_context_trace_id = lambda request: "trace"
    return scheduler


def test_managed_context_archive_skips_compaction_on_hard_cpu_capacity() -> None:
    scheduler = _scheduler_for_archive_capacity_skip()
    blocks = [
        SimpleNamespace(is_null=False, logical_start=0, block_id=1),
        SimpleNamespace(is_null=False, logical_start=16, block_id=2),
    ]
    compaction_mgr = SimpleNamespace(req_to_blocks={"req": blocks})
    request = SimpleNamespace(request_id="req", _all_token_ids=list(range(64)))

    span_ids = scheduler._archive_managed_context_span(
        request,
        compaction_mgr=compaction_mgr,
        evict_start=0,
        evict_end=32,
        explicit_block_range=(0, 2),
        last_turn_evicted=1,
        stride_used=2,
    )

    assert span_ids is None
    assert scheduler._managed_context_archive == {}


def test_archive_capacity_skip_log_is_rate_limited(monkeypatch) -> None:
    scheduler = _scheduler_for_archive_capacity_skip()
    blocks = [
        SimpleNamespace(is_null=False, logical_start=0, block_id=1),
        SimpleNamespace(is_null=False, logical_start=16, block_id=2),
    ]
    compaction_mgr = SimpleNamespace(req_to_blocks={"req": blocks})
    request = SimpleNamespace(request_id="req", _all_token_ids=list(range(64)))
    warnings = []

    def fake_warning(message, *args, **kwargs) -> None:
        warnings.append(message)

    monkeypatch.setattr(scheduler_mod.logger, "warning", fake_warning)
    for _ in range(2):
        assert scheduler._archive_managed_context_span(
            request,
            compaction_mgr=compaction_mgr,
            evict_start=0,
            evict_end=32,
            explicit_block_range=(0, 2),
            last_turn_evicted=1,
            stride_used=2,
        ) is None
    scheduler._managed_context_cpu_free_block_ids.append(0)
    assert scheduler._archive_managed_context_span(
        request,
        compaction_mgr=compaction_mgr,
        evict_start=0,
        evict_end=32,
        explicit_block_range=(0, 2),
        last_turn_evicted=1,
        stride_used=2,
    ) is None

    assert warnings.count(
        "[MANAGED-CONTEXT-COMPACT-SKIP-CAPACITY] req=%s trace=%s "
        "evict=[%d,%d) blocks=%d cpu_free=%d cpu_max=%d "
        "gpu_pinned_blocks=%d managed=%s"
    ) == 2


def test_retry_managed_context_gpu_pinned_offload_retries_when_capacity_exists() -> None:
    scheduler = object.__new__(Scheduler)
    key = ("trace", "T0001")
    span = _span("T0001", "gpu_pinned", 1)
    scheduler._managed_context_enabled = True
    scheduler._managed_context_cpu_offload_immediate = True
    scheduler._managed_context_cpu_evict_on_capacity = False
    scheduler._managed_context_cpu_free_block_ids = deque([0])
    scheduler._managed_context_cpu_max_pending_store_events = 0
    scheduler._managed_context_store_event_to_span = {}
    scheduler._managed_context_archive_order = deque([key])
    scheduler._managed_context_archive = {key: span}
    started = []

    def fake_start_managed_context_cpu_offload(span, reason):
        started.append((span.span_id, reason))
        span.status = "offload_pending"
        return None

    scheduler._start_managed_context_cpu_offload = (
        fake_start_managed_context_cpu_offload
    )

    assert scheduler._retry_managed_context_gpu_pinned_offloads("test") == 1
    assert started == [("T0001", "test")]


def test_retry_managed_context_gpu_pinned_offload_skips_hard_capacity() -> None:
    scheduler = object.__new__(Scheduler)
    key = ("trace", "T0001")
    span = _span("T0001", "gpu_pinned", 2)
    scheduler._managed_context_enabled = True
    scheduler._managed_context_cpu_offload_immediate = True
    scheduler._managed_context_cpu_evict_on_capacity = False
    scheduler._managed_context_cpu_free_block_ids = deque([0])
    scheduler._managed_context_cpu_max_pending_store_events = 0
    scheduler._managed_context_store_event_to_span = {}
    scheduler._managed_context_archive_order = deque([key])
    scheduler._managed_context_archive = {key: span}
    scheduler._start_managed_context_cpu_offload = lambda span, reason: None

    assert scheduler._retry_managed_context_gpu_pinned_offloads("test") == 0
    assert span.status == "gpu_pinned"


def _request_with_restore_args() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="request",
        managed_context_defer_restore_until_prefill=True,
        sampling_params=SimpleNamespace(
            extra_args={
                "kve_restore_span_ids": ["T0001"],
                "kve_restore_defer_until_prefill": True,
                "kve_restore_after_visible_tokens": 128,
                "other": "kept",
            }
        ),
    )


def test_managed_context_drops_oversize_restore_by_default() -> None:
    scheduler = object.__new__(Scheduler)
    released = []
    scheduler._release_managed_context_restore_reservation = (
        lambda request_id, reason: released.append(("reserve", request_id, reason))
    )
    scheduler._release_managed_context_deferred_restore = (
        lambda request_id, reason: released.append(("deferred", request_id, reason))
    )
    request = _request_with_restore_args()

    dropped = scheduler._drop_unavailable_managed_context_restore(
        request,
        "restore would exceed max_model_len: visible=16000 hidden=512 max=16384",
    )

    assert dropped
    assert request.managed_context_defer_restore_until_prefill is False
    assert request.sampling_params.extra_args == {"other": "kept"}
    assert released == [
        ("reserve", "request", "restore-dropped"),
        ("deferred", "request", "restore-dropped"),
    ]


def test_managed_context_keeps_unavailable_restore_by_default() -> None:
    scheduler = object.__new__(Scheduler)
    scheduler._release_managed_context_restore_reservation = (
        lambda request_id, reason: None
    )
    scheduler._release_managed_context_deferred_restore = (
        lambda request_id, reason: None
    )
    request = _request_with_restore_args()

    dropped = scheduler._drop_unavailable_managed_context_restore(
        request,
        "span 'T0001' is not available for this trace",
    )

    assert not dropped
    assert request.managed_context_defer_restore_until_prefill is True
    assert "kve_restore_span_ids" in request.sampling_params.extra_args


def test_managed_context_cpu_offload_respects_pending_store_limit() -> None:
    scheduler = object.__new__(Scheduler)
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_pending_store_events = 1
    scheduler._managed_context_store_event_to_span = {
        1: _span("T0001", "offload_pending", 1),
    }
    span = _span("T0002", "gpu_pinned", 1)
    span.entries = [(SimpleNamespace(), [SimpleNamespace(block_id=7)])]

    error = scheduler._start_managed_context_cpu_offload(span, "test")

    assert error == (
        "managed-context CPU store transfer limit reached: pending=1 max=1"
    )
    assert span.status == "gpu_pinned"


def test_managed_context_cpu_offload_counts_request_kv_swap_store_limit() -> None:
    scheduler = object.__new__(Scheduler)
    scheduler._managed_context_cpu_archive_enabled = True
    scheduler._managed_context_cpu_max_pending_store_events = 1
    scheduler._managed_context_store_event_to_span = {}
    scheduler._request_kv_swap_store_event_to_request_id = {1: "request"}
    span = _span("T0002", "gpu_pinned", 1)
    span.entries = [(SimpleNamespace(), [SimpleNamespace(block_id=7)])]

    error = scheduler._start_managed_context_cpu_offload(span, "test")

    assert error == (
        "managed-context CPU store transfer limit reached: pending=1 max=1"
    )
    assert span.status == "gpu_pinned"


def test_managed_context_cpu_load_limit_is_retryable() -> None:
    scheduler = object.__new__(Scheduler)
    scheduler._managed_context_pending_loads = {}
    scheduler._managed_context_cpu_max_pending_load_events = 1
    scheduler._managed_context_load_event_to_request_id = {1: "other"}
    request = SimpleNamespace(request_id="reader")
    span = _span("T0001", "cpu_offloaded", 1)

    start = scheduler._start_managed_context_cpu_load(request, [span])

    assert start.error == (
        "managed-context CPU load transfer limit reached: pending=1 max=1"
    )
    assert start.retryable
