import time
from collections import deque
from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Phase4Pin, Scheduler
from vllm.v1.request import RequestStatus


def _scheduler_with_requests(request_ids: set[str]) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request_id: object() for request_id in request_ids}
    scheduler.running = []
    scheduler.waiting = deque()
    scheduler.skipped_waiting = deque()
    scheduler._phase4_pinned_blocks = {}
    scheduler._phase4_pin_order = deque()
    scheduler._compaction_enabled = True
    scheduler._compaction_max_turns = 1
    scheduler.cache_config = SimpleNamespace(enable_prefix_caching=True)
    return scheduler


def _request(
    trace_id: str, expected_cached_tokens: int | None = 64
) -> SimpleNamespace:
    extra_args = {"kve_phase4_trace_id": trace_id}
    if expected_cached_tokens is not None:
        extra_args["kve_phase4_expected_cached_tokens"] = (
            expected_cached_tokens
        )
    return SimpleNamespace(
        num_prompt_tokens=512,
        sampling_params=SimpleNamespace(
            extra_args=extra_args,
        ),
    )


def _pin(
    *,
    token_count: int = 0,
    consumed_by_request_id: str | None = None,
    consumed_at: float | None = None,
) -> Phase4Pin:
    return Phase4Pin(
        entries=[],
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


def test_phase4_pressure_release_evicts_safe_stale_prefix(
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

    assert scheduler._release_phase4_pressure_pin("pressure")
    assert released == [("trace", "pressure", True)]
    assert "trace" not in scheduler._phase4_pinned_blocks


def test_phase4_stall_pressure_release_requires_no_progress() -> None:
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


def test_managed_context_restore_preempt_abort_uses_free_request() -> None:
    scheduler = _scheduler_with_requests({"restore"})
    request = SimpleNamespace(
        request_id="restore",
        status=RequestStatus.RUNNING,
        position_offset=0,
    )
    scheduler.requests = {"restore": request}
    scheduler._managed_context_active_restores = {"restore": object()}
    scheduler._managed_context_deferred_restores = {}
    freed_requests = []

    def fake_free_request(freed_request):
        freed_requests.append(freed_request)
        scheduler.requests.pop(freed_request.request_id, None)

    scheduler._free_request = fake_free_request
    queued_outputs = []
    scheduler._queue_finished_request_output = queued_outputs.append

    scheduler._preempt_request(request, time.monotonic())

    assert request.status == RequestStatus.FINISHED_ERROR
    assert freed_requests == [request]
    assert queued_outputs == [request]
    assert "restore" not in scheduler.requests
