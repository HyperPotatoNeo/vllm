# SPDX-License-Identifier: Apache-2.0
"""Regression tests for KV cache compaction in the V1 scheduler.

These tests exercise the interaction between CompactingKVCacheManager
(physical block-level eviction) and Scheduler._compact_request
(logical token-list trim), focusing on correctness edge cases.
"""

import pytest

from vllm.v1.core.compaction.manager import CompactingKVCacheManager
from vllm.v1.outputs import ModelRunnerOutput

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
    assert event.blocks_evicted == compaction_stride // block_size == 1
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
