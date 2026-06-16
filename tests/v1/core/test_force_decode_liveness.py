"""Phase-A liveness backstop tests.

A RUNNING request stuck fully-computed (num_computed_tokens == num_tokens, i.e.
C==T) yields num_new_tokens == 0 and is skipped forever at the
num_new_tokens==0 check, idling the GPU -- the fix3 tail stall. The scheduler
must force one decode for it (unless it is legitimately held), and the
stall_like detector must flag running-but-no-progress (not only the
running-empty/waiting case). See plans/request_kv_swap_preemption.md (Phase A).
"""
from collections import deque
from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler


def _scheduler() -> Scheduler:
    s = object.__new__(Scheduler)
    s._managed_context_active_restores = {}
    s._managed_context_deferred_restores = {}
    s._managed_context_pending_loads = {}
    s.prev_step_scheduled_req_ids = set()
    s._kve_diag_enabled = lambda: False
    return s


def _req(*, request_id="r", num_tokens=512, num_computed=512, **overrides):
    req = SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        num_computed_tokens=num_computed,
        num_cached_tokens=0,
        num_output_placeholders=0,
        padding_pending=False,
        needs_rebuild=False,
        skip_reading_prefix_cache=False,
    )
    for key, value in overrides.items():
        setattr(req, key, value)
    return req


def test_force_decode_rescues_fully_computed_running_request():
    s = _scheduler()
    req = _req(num_tokens=512, num_computed=512)
    assert s._kve_force_decode_stuck_running_request(req) is True
    assert req.num_computed_tokens == 511  # one-behind => num_new_tokens becomes 1
    assert req.needs_rebuild is True
    assert req.skip_reading_prefix_cache is True


def test_force_decode_noop_for_healthy_decode():
    s = _scheduler()
    req = _req(num_tokens=512, num_computed=511)  # already one-behind = healthy pump
    assert s._kve_force_decode_stuck_running_request(req) is False
    assert req.num_computed_tokens == 511


def test_force_decode_noop_for_prefilling_request():
    s = _scheduler()
    req = _req(num_tokens=512, num_computed=128)
    assert s._kve_force_decode_stuck_running_request(req) is False
    assert req.num_computed_tokens == 128


def test_force_decode_skips_placeholders_padding_and_pending_deletions():
    s = _scheduler()
    for overrides in (
        {"num_output_placeholders": 1},
        {"padding_pending": True},
        # replay refill that still has pending deletions is genuinely mid-refill
        # -> leave it to the segmented-advance loop, do not force.
        {
            "_kve_compact_replay_segmented_refill_active": True,
            "_kve_compact_replay_segmented_deletions": [[160, 160, 736, 1648]],
        },
    ):
        req = _req(num_tokens=512, num_computed=512, **overrides)
        assert s._kve_force_decode_stuck_running_request(req) is False, overrides
        assert req.num_computed_tokens == 512, overrides


def test_force_decode_completes_then_forces_stuck_replay():
    # replay-flagged, refill DONE (no pending deletions): the wrapper must
    # complete the refill (breaking the completion<->scheduling deadlock) and
    # then force one decode.
    s = _scheduler()
    completed = {"called": False}

    def _fake_complete(req, *, reason):
        completed["called"] = True
        return True  # leaves the request at C==T (num_computed unchanged)

    s._complete_compact_replay_full_refill = _fake_complete
    req = _req(num_tokens=512, num_computed=512,
               _kve_compact_replay_full_refill_active=True)
    assert s._kve_force_decode_stuck_running_request(req) is True
    assert completed["called"] is True
    assert req.num_computed_tokens == 511


def test_force_decode_skips_replay_when_completion_fails():
    # If the refill cannot be completed cleanly, do NOT force-decode.
    s = _scheduler()
    s._complete_compact_replay_full_refill = lambda req, *, reason: False
    req = _req(num_tokens=512, num_computed=512,
               _kve_compact_replay_segmented_refill_active=True)
    assert s._kve_force_decode_stuck_running_request(req) is False
    assert req.num_computed_tokens == 512


def test_force_decode_skips_managed_restore_states():
    for attr in (
        "_managed_context_active_restores",
        "_managed_context_deferred_restores",
        "_managed_context_pending_loads",
    ):
        s = _scheduler()
        req = _req(request_id="restoring", num_tokens=512, num_computed=512)
        getattr(s, attr)["restoring"] = object()
        assert s._kve_force_decode_stuck_running_request(req) is False, attr
        assert req.num_computed_tokens == 512, attr


def test_force_decode_ignores_zero_token_request():
    s = _scheduler()
    req = _req(num_tokens=0, num_computed=0)
    assert s._kve_force_decode_stuck_running_request(req) is False


def _liveness_scheduler() -> Scheduler:
    s = object.__new__(Scheduler)
    s.running = []
    s.waiting = deque()
    s.skipped_waiting = deque()
    s._kve_diag_enabled = lambda: True
    s._kve_liveness_diag_last_ts = 0.0
    s._env_float = lambda name, default: 0.0  # interval 0 => never rate-limited
    s._kve_gpu_block_pool_diag_summary = lambda: {}
    s._kve_managed_context_diag_summary = lambda: {}
    s._kve_request_diag_samples = lambda reqs: []
    return s


def _call_liveness(s: Scheduler) -> None:
    s._kve_maybe_log_scheduler_liveness(
        total_num_scheduled_tokens=0,
        token_budget=8192,
        preempted_reqs=[],
        scheduled_running_reqs=[],
        scheduled_new_reqs=[],
        scheduled_resumed_reqs=[],
    )


def test_stall_like_flags_running_no_progress():
    # scheduled 0 tokens while a request is RUNNING == the C==T tail stall.
    s = _liveness_scheduler()
    s.running = [object()]
    _call_liveness(s)
    assert s._kve_liveness_diag_last_ts > 0.0  # fired => stall_like was True


def test_stall_like_quiet_when_idle_and_empty():
    # scheduled 0 tokens with nothing running or waiting is not a stall.
    s = _liveness_scheduler()
    _call_liveness(s)
    assert s._kve_liveness_diag_last_ts == 0.0  # early-returned => not flagged
