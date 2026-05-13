# SPDX-License-Identifier: Apache-2.0
"""Regression tests for KV cache compaction in the V1 scheduler.

These tests exercise the interaction between CompactingKVCacheManager
(physical block-level eviction) and Scheduler._compact_request
(logical token-list trim), focusing on correctness edge cases.
"""

import time

import msgspec
import pytest
import torch

from vllm.v1.core.compaction.am_manager import AttentionMatchingKVCacheManager
from vllm.v1.core.compaction.am_runtime import OMPCompaction, build_attention_matching_plan
from vllm.v1.core.compaction.manager import CompactingKVCacheManager
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.core.compaction.types import CompactionEvent
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.outputs import AttentionMatchingCompactionResult, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.utils import ConstantList

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _run_decode_until_compacted(scheduler, request, max_steps: int = 2048):
    """Prefill + decode until the request has exactly one compaction event.

    Returns (step_count, total_generated_tokens). total_generated is the
    MONOTONIC count (request.num_total_generated), which is unaffected by
    compaction-time trimming of _output_token_ids.
    """
    next_sampled = 10_000  # sentinel id so gen tokens are distinguishable
    steps = 0
    while len(request.compaction_events) == 0 and steps < max_steps:
        output = scheduler.schedule()
        if request.request_id not in output.num_scheduled_tokens:
            # Something prevented progress (e.g. no blocks left).
            break
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[next_sampled]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        # Advance sentinel whenever a new token was actually appended.
        if request.num_total_generated > (next_sampled - 10_000):
            next_sampled += 1
        steps += 1
    return steps, request.num_total_generated


def test_compaction_strategy_defaults_to_fifo_manager():
    """FIFO remains the default manager when compaction is enabled."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=16,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=80,
        compaction_stride=16,
    )

    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    assert any(type(mgr) is CompactingKVCacheManager for mgr in managers)
    assert all(not isinstance(mgr, AttentionMatchingKVCacheManager) for mgr in managers)


def test_attention_matching_strategy_requires_compaction_to_activate():
    """Selecting AM alone must not change the non-compaction path."""
    scheduler = create_scheduler(compaction_strategy="attention_matching")

    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    assert all(not isinstance(mgr, CompactingKVCacheManager) for mgr in managers)
    assert all(not isinstance(mgr, AttentionMatchingKVCacheManager) for mgr in managers)


def test_attention_matching_strategy_is_opt_in():
    """AM manager is only instantiated when the AM strategy is selected."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=16,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=80,
        compaction_stride=16,
        compaction_strategy="attention_matching",
    )

    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    assert all(type(mgr) is AttentionMatchingKVCacheManager for mgr in managers)


def test_attention_matching_nnls_handles_rank_deficient_matrix():
    """AM NNLS should stay finite on degenerate solves seen under batching."""
    compactor = OMPCompaction()
    M = torch.tensor(
        [
            [1.0, 1.0, 2.0],
            [2.0, 2.0, 4.0],
            [3.0, 3.0, 6.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    B = compactor._nnls_pg(M, y)

    assert torch.isfinite(B).all()
    assert torch.all(B >= 0)


def test_attention_matching_c2_handles_rank_deficient_system():
    """AM C2 reconstruction should not crash on ill-conditioned systems."""
    compactor = OMPCompaction()
    C1 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    beta = torch.zeros(2, dtype=torch.float32)
    K = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    V = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    queries = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

    C2 = compactor._compute_C2(C1, beta, K, V, queries)

    assert torch.isfinite(C2).all()


def test_attention_matching_c2_raises_on_nonfinite_scores():
    """AM should fail loudly when non-finite intermediates appear."""
    compactor = OMPCompaction()
    C1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    beta = torch.tensor([float("nan"), 0.0], dtype=torch.float32)
    K = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    V = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="non-finite"):
        compactor._compute_C2(C1, beta, K, V, queries)


def test_attention_matching_normalizes_large_nnls_log_weights():
    """AM beta computation should stay finite for very large positive weights."""
    compactor = OMPCompaction()
    B = torch.tensor([1e100, 1e80, 1e60], dtype=torch.float64)

    beta = compactor._stable_log_weights(B)

    assert torch.isfinite(beta).all()
    assert beta.max().item() == 0.0


def test_attention_matching_strategy_does_not_use_scheduler_compaction_path():
    """AM compaction must be worker-driven, not the FIFO scheduler path."""
    block_size = 16
    prompt_len = 50
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=block_size,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=80,
        compaction_stride=16,
        compaction_strategy="attention_matching",
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=256,
        ignore_eos=True,
        block_size=block_size,
    )
    scheduler.add_request(request)

    next_sampled = 10_000
    for _ in range(96):
        output = scheduler.schedule()
        assert request.request_id in output.num_scheduled_tokens
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[next_sampled]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        if request.num_total_generated > (next_sampled - 10_000):
            next_sampled += 1

    assert request.num_computed_tokens > scheduler.cache_config.compaction_window_size
    assert request.compaction_events == []
    assert not request.needs_rebuild


def test_attention_matching_compaction_result_rewrites_request_state():
    """Scheduler should consume worker-side AM compaction results."""
    block_size = 16
    prompt_len = 32
    window = 48
    stride = 16
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=block_size,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=window,
        compaction_stride=stride,
        compaction_strategy="attention_matching",
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=256,
        ignore_eos=True,
        block_size=block_size,
    )
    scheduler.add_request(request)

    next_sampled = 10_000
    source_len = 0
    for _ in range(64):
        output = scheduler.schedule()
        assert request.request_id in output.num_scheduled_tokens
        source_len = request.num_computed_tokens
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[next_sampled]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        if request.num_computed_tokens > window:
            break
        if request.num_total_generated > (next_sampled - 10_000):
            next_sampled += 1

    source_len = request.num_computed_tokens
    plan = build_attention_matching_plan(
        num_computed_tokens=source_len,
        window_size=window,
        stride=stride,
        num_prompt_tokens=request.num_prompt_tokens,
    )
    assert plan is not None

    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    sampled = next_sampled + 1
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[sampled]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            attention_matching_compactions={
                request.request_id: AttentionMatchingCompactionResult(
                    request_id=request.request_id,
                    protected_prefix_len=plan.protected_prefix_len,
                    synthetic_prefix_len=plan.synthetic_prefix_len,
                    exact_kept_tokens=plan.exact_kept_tokens,
                    position_offset_delta=plan.offset_delta,
                )
            },
        ),
    )

    assert request.num_prompt_tokens == stride
    assert request.logical_prompt_len == prompt_len
    assert request.num_computed_tokens == plan.target_len
    assert request.position_offset == plan.offset_delta
    assert request.needs_rebuild
    assert len(request.compaction_events) == 1
    assert request.compaction_events[0].tokens_evicted == plan.offset_delta
    assert request.all_token_ids[:stride] == [0] * stride


def test_attention_matching_plan_can_protect_prompt_prefix():
    """AM should compact only after the protected prompt prefix."""
    plan = build_attention_matching_plan(
        num_computed_tokens=100,
        window_size=48,
        stride=16,
        num_prompt_tokens=32,
        protected_prefix_len=32,
    )

    assert plan is not None
    assert plan.protected_prefix_len == 32
    assert plan.synthetic_prefix_len == 16
    assert plan.exact_kept_tokens == 16
    assert plan.compact_region_len == 52
    assert plan.target_len == 64
    assert plan.offset_delta == 36


def test_attention_matching_stop_uses_logical_prompt_len():
    """AM prompt rewrites must not weaken max_model_len stop checks."""
    prompt_len = 32
    synthetic_prompt_len = 16
    max_model_len = 40
    (request,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=64,
        ignore_eos=True,
    )
    request.num_prompt_tokens = synthetic_prompt_len

    for token_id in range(max_model_len - prompt_len - 1):
        request.append_output_token_ids(10_000 + token_id)
        assert not check_stop(request, max_model_len)

    request.append_output_token_ids(20_000)
    assert check_stop(request, max_model_len)


def test_attention_matching_rejects_resumable_requests():
    """Streaming session updates are not supported by the AM baseline."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=16,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=80,
        compaction_stride=16,
        compaction_strategy="attention_matching",
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=32,
        max_tokens=64,
        ignore_eos=True,
    )
    request.resumable = True

    with pytest.raises(
        AssertionError,
        match="attention_matching baseline does not support resumable streaming",
    ):
        scheduler.add_request(request)


def test_attention_matching_preemption_keeps_request_resumable():
    """AM-compacted requests should be preempted, not aborted."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=16,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=80,
        compaction_stride=16,
        compaction_strategy="attention_matching",
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=32,
        max_tokens=128,
        ignore_eos=True,
        block_size=16,
    )
    scheduler.add_request(request)
    scheduler.schedule()
    assert request in scheduler.running

    request.prompt_token_ids = [0] * 16
    request.num_prompt_tokens = 16
    request._output_token_ids = [10_000 + i for i in range(32)]
    request.output_token_ids = ConstantList(request._output_token_ids)
    request._all_token_ids = request.prompt_token_ids + request._output_token_ids
    request.all_token_ids = ConstantList(request._all_token_ids)
    request.num_computed_tokens = 48
    request.position_offset = 16
    request.attention_matching_active = True
    request.attention_matching_snapshot_version = 3
    request.attention_matching_target_len = 48

    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.time())

    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 48
    assert request.attention_matching_restore_pending
    assert request.request_id not in scheduler.finished_req_ids


def test_attention_matching_resumed_request_requests_restore():
    """Resumed AM requests should carry restore metadata to the worker."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=16,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=80,
        compaction_stride=16,
        compaction_strategy="attention_matching",
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=32,
        max_tokens=128,
        ignore_eos=True,
        block_size=16,
    )
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert request.request_id in first.num_scheduled_tokens

    request.prompt_token_ids = [0] * 16
    request.num_prompt_tokens = 16
    request._output_token_ids = [10_000 + i for i in range(32)]
    request.output_token_ids = ConstantList(request._output_token_ids)
    request._all_token_ids = request.prompt_token_ids + request._output_token_ids
    request.all_token_ids = ConstantList(request._all_token_ids)
    request.num_computed_tokens = 48
    request.position_offset = 16
    request.attention_matching_active = True
    request.attention_matching_snapshot_version = 7
    request.attention_matching_target_len = 48

    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.time())

    resumed = scheduler.schedule()
    cached = resumed.scheduled_cached_reqs

    assert request.request_id in cached.resumed_req_ids
    assert request.request_id in cached.attention_matching_restore_req_ids
    assert cached.attention_matching_snapshot_versions[request.request_id] == 7
    assert cached.position_offsets[request.request_id] == 16
    assert cached.prompt_lengths[request.request_id] == 16
    assert not request.attention_matching_restore_pending


def test_fifo_compacted_request_still_aborts_on_preemption():
    """The new AM resume path must not change FIFO compaction behavior."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=16,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=80,
        compaction_stride=16,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=32,
        max_tokens=128,
        ignore_eos=True,
        block_size=16,
    )
    scheduler.add_request(request)
    scheduler.schedule()
    assert request in scheduler.running

    request.position_offset = 16
    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.time())

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request.request_id in scheduler.finished_req_ids


def test_compaction_trim_non_block_aligned_prompt():
    """Regression test for the prompt-block-alignment bug.

    With prompt_len not a multiple of block_size, the first `block_size -
    (prompt_len % block_size)` generated tokens live in the tail of the last
    prompt block and are NEVER physically evicted (block-level eviction only
    frees whole post-prompt blocks). The scheduler's logical trim must start
    at prompt_aligned_len = ceil(prompt_len/block_size) * block_size, not at
    prompt_len. Otherwise the logical token list removes tokens with the wrong
    identities (same count, different content), silently desynchronizing
    all_token_ids / output_token_ids from what the physical KV actually holds.
    """
    block_size = 16
    prompt_len = 50  # 50 % 16 == 2; prompt_aligned_len = 64
    # Window of 80 + stride of 16 means the first compaction fires around
    # num_computed_tokens == 81, i.e. after 31 generated tokens. At that point
    # block 4 (the first pure-generation block, holding gen[14..29]) is fully
    # written, so we never trigger the separate "evicting a partially-filled
    # block" edge case.
    compaction_window_size = 80
    compaction_stride = 16

    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=block_size,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=compaction_window_size,
        compaction_stride=compaction_stride,
    )

    # Sanity: the scheduler actually registered a CompactingKVCacheManager.
    assert any(
        isinstance(m, CompactingKVCacheManager)
        for m in scheduler.kv_cache_manager.coordinator.single_type_managers
    )

    # One request, non-block-aligned prompt length, enough decode budget to
    # push past the compaction window.
    (request,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=256,
        ignore_eos=True,
        block_size=block_size,
    )
    # Mark prompt_token_ids with identifiable values so we can check identities.
    # create_requests already uses [i]*num_tokens with i=0 for the first request;
    # that's fine as long as we distinguish prompt (0) from generated (>=10000).
    for tok in request.prompt_token_ids:
        assert tok == 0
    scheduler.add_request(request)

    steps, total_generated = _run_decode_until_compacted(scheduler, request)
    assert len(request.compaction_events) == 1, (
        f"Expected exactly one compaction event, got "
        f"{len(request.compaction_events)} after {steps} steps"
    )

    event = request.compaction_events[0]
    assert event.tokens_evicted == compaction_stride
    assert event.position_offset_after == compaction_stride
    assert request.position_offset == compaction_stride

    # ------------------------------------------------------------------
    # Identity checks: the key assertion of this regression test.
    # ------------------------------------------------------------------
    # Physical eviction dropped block index prompt_blocks = ceil(50/16) = 4,
    # which holds gen[14..29]. So the retained logical view must be:
    #     prompt[0..49] + gen[0..13] + gen[30..total_generated-1]
    # and NOT prompt[0..49] + gen[16..total_generated-1] (the bug), and NOT
    # prompt[0..49] + gen[0..total_generated-17] (unaligned-from-end variant).
    all_ids = list(request._all_token_ids)
    out_ids = list(request._output_token_ids)

    expected_len = prompt_len + total_generated - compaction_stride
    assert len(all_ids) == expected_len, (
        f"_all_token_ids has {len(all_ids)} entries, expected {expected_len}"
    )
    assert len(out_ids) == total_generated - compaction_stride

    # Prompt preserved unchanged.
    assert all_ids[:prompt_len] == [0] * prompt_len

    # The first 14 gen tokens (gen[0..13]) live in the tail of the last prompt
    # block and must still be present at indices [prompt_len, prompt_len+14).
    prompt_aligned_len = (
        (prompt_len + block_size - 1) // block_size
    ) * block_size
    gen_tail_in_prompt_block = prompt_aligned_len - prompt_len
    assert gen_tail_in_prompt_block == 14
    assert all_ids[prompt_len : prompt_len + gen_tail_in_prompt_block] == [
        10_000 + i for i in range(gen_tail_in_prompt_block)
    ], (
        "First gen_tail_in_prompt_block generated tokens (physically retained "
        "in the partial last prompt block) were incorrectly trimmed."
    )

    # Immediately after the retained tail there must be gen[14+stride] =
    # gen[30], NOT gen[14] (which was evicted) and NOT gen[16] (old-bug value).
    first_post_evict_idx = prompt_len + gen_tail_in_prompt_block
    assert (
        all_ids[first_post_evict_idx]
        == 10_000 + gen_tail_in_prompt_block + compaction_stride
    ), (
        f"Token at index {first_post_evict_idx} is {all_ids[first_post_evict_idx]}, "
        f"expected gen[{gen_tail_in_prompt_block + compaction_stride}] = "
        f"{10_000 + gen_tail_in_prompt_block + compaction_stride}. "
        "The scheduler is trimming from the wrong offset."
    )

    # output_token_ids reflects the same structure without the prompt.
    assert out_ids[:gen_tail_in_prompt_block] == [
        10_000 + i for i in range(gen_tail_in_prompt_block)
    ]
    assert (
        out_ids[gen_tail_in_prompt_block]
        == 10_000 + gen_tail_in_prompt_block + compaction_stride
    )

    # num_computed_tokens lags len(_all_token_ids) by 1: the just-sampled
    # token has been appended to the list but hasn't yet gone through a forward
    # pass (it's pending for the next schedule call).
    assert request.num_computed_tokens == expected_len - 1


def test_compaction_waits_for_full_evict_block():
    """needs_compaction must defer firing until the last block to be evicted
    is fully filled. Otherwise the manager reports tokens_evicted =
    stride_blocks * block_size but fewer real tokens are physically in that
    block, causing the scheduler to over-decrement num_computed_tokens and
    silently delete the pending just-sampled token from _all_token_ids during
    the trim.

    Config picked so the naive guard (num_computed > window AND gen_blocks >=
    stride_blocks) would fire with a PARTIAL last gen block. Correct guard
    must wait until the last evict block is full, i.e. num_computed_tokens
    >= (prompt_blocks + stride_blocks) * block_size.
    """
    block_size = 16
    prompt_len = 50  # prompt_blocks = 4, prompt_aligned_len = 64
    compaction_window_size = 64  # would fire at num_computed > 64
    compaction_stride = 16

    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=block_size,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=compaction_window_size,
        compaction_stride=compaction_stride,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=256,
        ignore_eos=True,
        block_size=block_size,
    )
    scheduler.add_request(request)

    steps, total_generated = _run_decode_until_compacted(scheduler, request)
    assert len(request.compaction_events) == 1

    # With the correct guard, compaction fires when num_computed_tokens first
    # reaches (prompt_blocks + stride_blocks) * block_size = (4 + 1) * 16 = 80.
    # Before eviction: num_computed = 80, len(all_ids) = 81 (one pending sample),
    # total_generated = 31. After eviction: num_computed = 64,
    # len(all_ids) = 65. The pending sample (gen[30]) must still be present at
    # _all_token_ids[-1].
    assert total_generated == 31, f"expected 31 gen tokens, got {total_generated}"
    all_ids = list(request._all_token_ids)
    assert len(all_ids) == 65, (
        f"partial-block guard failed: len(all_ids)={len(all_ids)}, expected 65"
    )
    # Pending sample gen[30] preserved as last entry.
    assert all_ids[-1] == 10_000 + 30, (
        f"pending sample lost: last token is {all_ids[-1]}, expected "
        f"{10_000 + 30}"
    )
    # num_computed should equal len(all_ids) - 1 (pending-sample invariant).
    assert request.num_computed_tokens == len(all_ids) - 1 == 64


def test_compaction_rejects_lmcache_connector():
    """Startup assertion: compaction must refuse to run with an LMCache KV
    connector, because LMCache's V1 adapter uses _output_token_ids[0] as a
    "first_tok" fingerprint which compaction invalidates.
    """
    with pytest.raises(AssertionError, match="compaction is incompatible with LMCache"):
        create_scheduler(
            block_size=16,
            compaction_window_size=80,
            compaction_stride=16,
            use_kv_connector="LMCacheConnectorV1",
        )


def test_compaction_trim_block_aligned_prompt():
    """Sanity: when prompt_len IS a multiple of block_size, trimming still
    behaves correctly (reduces exactly to pre-fix behavior).
    """
    block_size = 16
    prompt_len = 48  # 48 % 16 == 0
    compaction_window_size = 80
    compaction_stride = 16

    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=block_size,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=compaction_window_size,
        compaction_stride=compaction_stride,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=256,
        ignore_eos=True,
        block_size=block_size,
    )
    scheduler.add_request(request)
    _run_decode_until_compacted(scheduler, request)

    assert len(request.compaction_events) == 1
    assert request.position_offset == compaction_stride

    all_ids = list(request._all_token_ids)
    # gen_tail_in_prompt_block == 0, so the first generated token evicted is
    # gen[0]. Expected layout: prompt[0..47] + gen[stride..total-1].
    assert all_ids[:prompt_len] == [0] * prompt_len
    # First post-prompt token in the retained list should be gen[compaction_stride].
    assert all_ids[prompt_len] == 10_000 + compaction_stride


# ---------------------------------------------------------------------------
# Phase 3.1: compaction_events transport tests
# ---------------------------------------------------------------------------


def test_compaction_event_msgspec_roundtrip():
    """CompactionEvent is a msgspec.Struct and roundtrips cleanly."""
    ev = CompactionEvent(
        num_output_tokens_at_compaction=2097,
        tokens_evicted=512,
        position_offset_after=512,
    )
    encoded = msgspec.msgpack.encode(ev)
    decoded = msgspec.msgpack.decode(encoded, type=CompactionEvent)
    assert decoded.num_output_tokens_at_compaction == 2097
    assert decoded.tokens_evicted == 512
    assert decoded.position_offset_after == 512


def test_engine_core_output_carries_compaction_events():
    """EngineCoreOutput round-trips a list of compaction events, and
    omits the field entirely when None (default).
    """
    e1 = CompactionEvent(100, 16, 16)
    e2 = CompactionEvent(200, 16, 32)

    with_events = EngineCoreOutput(
        request_id="req-0",
        new_token_ids=[1, 2, 3],
        compaction_events=[e1, e2],
    )
    enc = msgspec.msgpack.encode(with_events)
    dec = msgspec.msgpack.decode(enc, type=EngineCoreOutput)
    assert dec.compaction_events is not None
    assert len(dec.compaction_events) == 2
    assert dec.compaction_events[0].num_output_tokens_at_compaction == 100
    assert dec.compaction_events[1].position_offset_after == 32

    without = EngineCoreOutput(request_id="req-1", new_token_ids=[4])
    assert without.compaction_events is None
    enc2 = msgspec.msgpack.encode(without)
    dec2 = msgspec.msgpack.decode(enc2, type=EngineCoreOutput)
    assert dec2.compaction_events is None
    # omit_defaults: no-event case should be strictly shorter on the wire.
    assert len(enc2) < len(enc)


def test_scheduler_attaches_compaction_events_to_engine_core_output():
    """End-to-end against a live scheduler: drive compaction, capture the
    EngineCoreOutput stream, assert at least one output carries the cumulative
    compaction_events list and that it matches request.compaction_events
    position by position.
    """
    block_size = 16
    prompt_len = 50
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=block_size,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=80,
        compaction_stride=16,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=256,
        ignore_eos=True,
        block_size=block_size,
    )
    scheduler.add_request(request)

    captured: list[EngineCoreOutput] = []
    next_sampled = 10_000
    for _ in range(64):
        output = scheduler.schedule()
        if request.request_id not in output.num_scheduled_tokens:
            break
        eco_dict = scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[next_sampled]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        if 0 in eco_dict:
            for eco in eco_dict[0].outputs:
                if eco.request_id == request.request_id:
                    captured.append(eco)
        if request.num_total_generated > (next_sampled - 10_000):
            next_sampled += 1
        if len(request.compaction_events) >= 1:
            # One more step so we see the post-compaction output too.
            one_more = scheduler.schedule()
            if request.request_id in one_more.num_scheduled_tokens:
                eco_dict2 = scheduler.update_from_output(
                    one_more,
                    ModelRunnerOutput(
                        req_ids=[request.request_id],
                        req_id_to_index={request.request_id: 0},
                        sampled_token_ids=[[next_sampled]],
                        logprobs=None,
                        prompt_logprobs_dict={},
                        pooler_output=[],
                    ),
                )
                if 0 in eco_dict2:
                    for eco in eco_dict2[0].outputs:
                        if eco.request_id == request.request_id:
                            captured.append(eco)
            break

    assert len(request.compaction_events) == 1
    with_events = [e for e in captured if e.compaction_events]
    assert len(with_events) >= 1, (
        "No EngineCoreOutput carried compaction_events despite the scheduler "
        "having fired a compaction event."
    )

    # The last captured ECO should carry the full cumulative list identical
    # to request.compaction_events.
    eco = with_events[-1]
    assert len(eco.compaction_events) == len(request.compaction_events)
    for got, want in zip(eco.compaction_events, request.compaction_events):
        assert got.num_output_tokens_at_compaction == want.num_output_tokens_at_compaction
        assert got.tokens_evicted == want.tokens_evicted
        assert got.position_offset_after == want.position_offset_after

    # Outputs emitted BEFORE the compaction fire should have no events.
    pre = captured[: len(captured) - len(with_events)]
    for eco in pre:
        assert eco.compaction_events is None
