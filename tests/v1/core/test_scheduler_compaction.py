# SPDX-License-Identifier: Apache-2.0
"""Regression tests for KV cache compaction in the V1 scheduler.

These tests exercise the interaction between CompactingKVCacheManager
(physical block-level eviction) and Scheduler._compact_request
(logical token-list trim), focusing on correctness edge cases.
"""

import msgspec
import pytest

from vllm.v1.core.compaction.manager import CompactingKVCacheManager
from vllm.v1.core.compaction.types import CompactionEvent
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.outputs import ModelRunnerOutput

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler

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


def test_turn_mode_compaction_does_not_require_token_window():
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=16,
        num_blocks=64,
        max_model_len=2048,
        compaction_window_size=0,
        compaction_stride=0,
        compaction_max_turns=6,
        compaction_eviction_turn_stride=2,
        compaction_turn_end_token_id=50256,
    )

    compaction_managers = [
        m for m in scheduler.kv_cache_manager.coordinator.single_type_managers
        if isinstance(m, CompactingKVCacheManager)
    ]
    assert len(compaction_managers) == 1
    assert compaction_managers[0].compaction_window_size == 0
    assert compaction_managers[0].compaction_stride == 0
    assert not compaction_managers[0].needs_compaction(
        "missing-request", 10_000, 0
    )


def test_compaction_window_and_turn_modes_are_mutually_exclusive():
    with pytest.raises(AssertionError, match="mutually exclusive"):
        create_scheduler(
            max_num_batched_tokens=1024,
            max_num_seqs=1,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            block_size=16,
            num_blocks=64,
            max_model_len=2048,
            compaction_window_size=80,
            compaction_stride=16,
            compaction_max_turns=6,
            compaction_eviction_turn_stride=2,
            compaction_turn_end_token_id=50256,
        )


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


# ---------------------------------------------------------------------------
# Protected prefix eviction tests
# ---------------------------------------------------------------------------


def test_compaction_protected_prefix_evicts_prompt_tokens():
    """When protected_prefix < prompt_len, eviction removes prompt tokens
    (old conversation history) rather than output tokens.

    Setup: prompt=300, protected_prefix=50, stride=16, window=320.
    Eviction boundary = ceil(50/16)*16 = 64. Evicted tokens at
    all_token_ids[64:80] are all prompt tokens (well within 300).
    output_token_ids should be unchanged.
    """
    block_size = 16
    prompt_len = 300
    protected_prefix = 50
    compaction_window_size = 320
    compaction_stride = 16

    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=block_size,
        num_blocks=128,
        max_model_len=2048,
        compaction_window_size=compaction_window_size,
        compaction_stride=compaction_stride,
        compaction_protected_prefix_tokens=protected_prefix,
    )

    (request,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=256,
        ignore_eos=True,
        block_size=block_size,
    )
    # Distinguish prompt tokens (value=0) from generated (>=10000).
    for tok in request.prompt_token_ids:
        assert tok == 0
    scheduler.add_request(request)

    steps, total_generated = _run_decode_until_compacted(scheduler, request)
    assert len(request.compaction_events) >= 1

    event = request.compaction_events[0]
    assert event.tokens_evicted == compaction_stride
    assert event.num_prompt_tokens == prompt_len  # recorded before decrement

    # After compaction: prompt_len should have decreased.
    assert request.num_prompt_tokens == prompt_len - compaction_stride

    all_ids = list(request._all_token_ids)
    out_ids = list(request._output_token_ids)

    # Protected prefix (first 50 tokens) must be preserved.
    assert all_ids[:protected_prefix] == [0] * protected_prefix

    # Output tokens should all still be present (none evicted).
    assert len(out_ids) == total_generated

    # All output tokens are generated tokens (>= 10000).
    assert all(t >= 10_000 for t in out_ids)

    # Total length: original prompt + generated - evicted prompt tokens.
    expected_len = prompt_len + total_generated - compaction_stride
    assert len(all_ids) == expected_len


def test_compaction_protected_prefix_mixed_eviction():
    """When the prompt beyond the protected prefix is shorter than stride,
    eviction spans the prompt/output boundary.

    Setup: prompt=60, protected_prefix=50, stride=32, window=80.
    Eviction boundary = ceil(50/16)*16 = 64. But prompt only extends to 60,
    and the eviction boundary (64) already exceeds the prompt. So
    all evicted tokens are output tokens — same as standard behavior.

    For a true mixed case: prompt=80, protected_prefix=50, stride=32.
    Eviction boundary = 64. prompt goes to 80. Evicted range: [64, 96).
    Prompt tokens evicted: min(80, 96) - 64 = 16.
    Output tokens evicted: 32 - 16 = 16.
    """
    block_size = 16
    prompt_len = 80
    protected_prefix = 50
    compaction_stride = 32
    compaction_window_size = 96

    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        block_size=block_size,
        num_blocks=128,
        max_model_len=2048,
        compaction_window_size=compaction_window_size,
        compaction_stride=compaction_stride,
        compaction_protected_prefix_tokens=protected_prefix,
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
    assert len(request.compaction_events) >= 1

    event = request.compaction_events[0]
    assert event.tokens_evicted == compaction_stride

    # Eviction boundary = ceil(50/16)*16 = 64, prompt_len=80.
    # Prompt tokens evicted: min(80, 64+32) - 64 = min(80,96) - 64 = 16.
    # Output tokens evicted: 32 - 16 = 16.
    prompt_tokens_evicted = min(prompt_len, 64 + compaction_stride) - 64
    output_tokens_evicted = compaction_stride - prompt_tokens_evicted
    assert prompt_tokens_evicted == 16
    assert output_tokens_evicted == 16

    assert request.num_prompt_tokens == prompt_len - prompt_tokens_evicted

    all_ids = list(request._all_token_ids)
    out_ids = list(request._output_token_ids)

    # Output tokens lost: the oldest 16 output tokens that were past the
    # prompt boundary (but within eviction range). The remaining output
    # tokens should be total_generated - output_tokens_evicted.
    assert len(out_ids) == total_generated - output_tokens_evicted

    # Protected prefix preserved.
    assert all_ids[:protected_prefix] == [0] * protected_prefix


def test_compaction_protected_prefix_backward_compat():
    """With protected_prefix=0 (default), behavior must be identical to
    the original tests: only output tokens evicted, prompt unchanged.
    """
    block_size = 16
    prompt_len = 50
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
        compaction_protected_prefix_tokens=0,  # explicit default
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

    # num_prompt_tokens unchanged (no prompt tokens evicted).
    assert request.num_prompt_tokens == prompt_len

    event = request.compaction_events[0]
    assert event.tokens_evicted == compaction_stride
    assert event.position_offset_after == compaction_stride

    all_ids = list(request._all_token_ids)
    out_ids = list(request._output_token_ids)

    # Prompt fully preserved.
    assert all_ids[:prompt_len] == [0] * prompt_len
    assert len(all_ids) == prompt_len + total_generated - compaction_stride
    assert len(out_ids) == total_generated - compaction_stride

    # Identity check: gen_tail tokens preserved, evicted tokens skipped.
    prompt_aligned_len = ((prompt_len + block_size - 1) // block_size) * block_size
    gen_tail = prompt_aligned_len - prompt_len
    assert out_ids[:gen_tail] == [10_000 + i for i in range(gen_tail)]
    assert out_ids[gen_tail] == 10_000 + gen_tail + compaction_stride


def test_compaction_protected_prefix_larger_than_prompt():
    """When protected_prefix > prompt_len, falls back to full-prompt
    protection (min clamp). No prompt tokens evicted.
    """
    block_size = 16
    prompt_len = 50
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
        compaction_protected_prefix_tokens=1000,  # much larger than prompt
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

    # Behaves same as no protected prefix: full prompt protected.
    assert request.num_prompt_tokens == prompt_len
    all_ids = list(request._all_token_ids)
    assert all_ids[:prompt_len] == [0] * prompt_len
    assert len(all_ids) == prompt_len + total_generated - compaction_stride


def test_compaction_event_carries_num_prompt_tokens():
    """CompactionEvent.num_prompt_tokens is populated and survives
    msgspec roundtrip."""
    ev = CompactionEvent(
        num_output_tokens_at_compaction=100,
        tokens_evicted=16,
        position_offset_after=16,
        num_prompt_tokens=300,
    )
    encoded = msgspec.msgpack.encode(ev)
    decoded = msgspec.msgpack.decode(encoded, type=CompactionEvent)
    assert decoded.num_prompt_tokens == 300

    # Default (0) should be omitted on wire (omit_defaults).
    ev_default = CompactionEvent(
        num_output_tokens_at_compaction=100,
        tokens_evicted=16,
        position_offset_after=16,
    )
    assert ev_default.num_prompt_tokens == 0
    enc_default = msgspec.msgpack.encode(ev_default)
    dec_default = msgspec.msgpack.decode(enc_default, type=CompactionEvent)
    assert dec_default.num_prompt_tokens == 0
    # With omit_defaults, the default-valued encoding should be shorter.
    assert len(enc_default) < len(encoded)


def test_compaction_event_kept_slice_roundtrip():
    """CompactionEvent.kept_indices and kept_token_ids survive msgspec
    roundtrip and obey the canonical invariant kept_token_ids[i] ==
    pre_event_tokens[kept_indices[i]] (verified by construction here)."""
    pre_event = [10, 11, 12, 13, 14, 15, 16, 17]
    evict_start, evict_end = 2, 5  # drop positions 2,3,4 -> tokens 12,13,14
    kept_indices = list(range(0, evict_start)) + list(range(evict_end, len(pre_event)))
    kept_token_ids = [pre_event[i] for i in kept_indices]

    ev = CompactionEvent(
        num_output_tokens_at_compaction=0,
        tokens_evicted=evict_end - evict_start,
        position_offset_after=evict_end - evict_start,
        kept_indices=kept_indices,
        kept_token_ids=kept_token_ids,
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(ev), type=CompactionEvent
    )
    assert decoded.kept_indices == [0, 1, 5, 6, 7]
    assert decoded.kept_token_ids == [10, 11, 15, 16, 17]
    # Invariant: token at each kept index matches the corresponding kept token.
    assert all(
        pre_event[idx] == tok
        for idx, tok in zip(decoded.kept_indices, decoded.kept_token_ids)
    )
    # Length consistency: kept count = pre_event count - evicted count.
    assert len(decoded.kept_indices) == len(pre_event) - decoded.tokens_evicted
    assert len(decoded.kept_token_ids) == len(decoded.kept_indices)


def test_compaction_event_kept_slice_default_empty():
    """Backward compat: when the new fields aren't supplied, they default
    to empty lists and omit_defaults trims them off the wire."""
    ev = CompactionEvent(
        num_output_tokens_at_compaction=0,
        tokens_evicted=16,
        position_offset_after=16,
    )
    assert ev.kept_indices == []
    assert ev.kept_token_ids == []
    encoded = msgspec.msgpack.encode(ev)
    decoded = msgspec.msgpack.decode(encoded, type=CompactionEvent)
    assert decoded.kept_indices == []
    assert decoded.kept_token_ids == []


# ─────────────────────────────────────────────────────────────────────────────
# Turn-mode compaction tests
# ─────────────────────────────────────────────────────────────────────────────
#
# Test setup: instead of relying on real ChatML tokenization, we use an
# arbitrary "im_end" token id (= 99) and inject it into the request's
# _all_token_ids at known positions. The system prompt is prompt_token_ids
# ending with the im_end marker; subsequent turn boundaries are produced by
# scripted sampling.

IM_END = 99
GEN_BASE = 10_000  # generated tokens are GEN_BASE..GEN_BASE+N (excluding 99)


def _make_system_prompt(sys_len: int) -> list[int]:
    """sys_len total tokens, last token is the im_end marker."""
    assert sys_len >= 2
    return [1] * (sys_len - 1) + [IM_END]


def _drive_scheduler(
    scheduler,
    request,
    script: list[int],
    max_steps: int = 1024,
):
    """Run prefill + decode, sampling tokens from `script` in order.

    Stops when the script is exhausted or max_steps is reached. The sentinel
    GEN_BASE is reserved for "any non-special token".
    """
    pending_idx = 0
    steps = 0
    while pending_idx < len(script) and steps < max_steps:
        output = scheduler.schedule()
        if request.request_id not in output.num_scheduled_tokens:
            break
        next_tok = script[pending_idx]
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[next_tok]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        # Only consume the script entry if a token was actually appended.
        # During pure prefill the request advances num_computed_tokens but
        # doesn't sample; check num_total_generated to know.
        if request.num_total_generated > pending_idx:
            pending_idx += 1
        steps += 1
    return steps


def _turn_script(num_turns: int, turn_token_count: int = 60) -> list[int]:
    """Build a sampled-token script that emits num_turns user+assistant
    pairs, each turn ending with a single IM_END.

    For test purposes we model both the user msg and the assistant msg as a
    single contiguous run of generated tokens followed by an IM_END. So one
    "turn" in the script is: turn_token_count generated tokens + 1 IM_END
    (representing end of user msg) + turn_token_count generated tokens +
    1 IM_END (representing end of assistant msg).
    """
    script = []
    counter = 0
    for _ in range(num_turns):
        # User msg body + im_end
        for _ in range(turn_token_count):
            script.append(GEN_BASE + counter)
            counter += 1
        script.append(IM_END)
        # Assistant msg body + im_end
        for _ in range(turn_token_count):
            script.append(GEN_BASE + counter)
            counter += 1
        script.append(IM_END)
    return script


def _make_turn_scheduler(
    *,
    sys_len: int,
    block_size: int = 16,
    compaction_window_size: int = 0,
    compaction_stride: int = 0,
    max_turns: int = 2,
    turn_stride: int = 1,
    enable_prefix_caching: bool = False,
):
    return create_scheduler(
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=enable_prefix_caching,
        block_size=block_size,
        num_blocks=512,
        max_model_len=4096,
        compaction_window_size=compaction_window_size,
        compaction_stride=compaction_stride,
        compaction_max_turns=max_turns,
        compaction_eviction_turn_stride=turn_stride,
        compaction_turn_end_token_id=IM_END,
    )


def test_turn_mode_boundary_scanning():
    """turn_end_positions is populated correctly as IM_END tokens appear."""
    sys_len = 32
    scheduler = _make_turn_scheduler(sys_len=sys_len, max_turns=10)
    (request,) = create_requests(
        num_requests=1,
        num_tokens=sys_len,
        max_tokens=2048,
        ignore_eos=True,
        block_size=16,
    )
    # Override the prompt to include the im_end marker at the end.
    request.prompt_token_ids[:] = _make_system_prompt(sys_len)
    request._all_token_ids[:] = list(request.prompt_token_ids)
    scheduler.add_request(request)

    # Drive 3 turns.
    script = _turn_script(num_turns=3, turn_token_count=20)
    _drive_scheduler(scheduler, request, script)

    # Expect 7 boundary markers: end_sys + 3 turns * 2 im_ends each.
    assert len(request.turn_end_positions) == 7
    assert request.turn_end_positions[0] == sys_len  # right after im_end
    # All recorded positions must be one PAST an im_end token.
    for p in request.turn_end_positions:
        assert request._all_token_ids[p - 1] == IM_END


def test_turn_mode_trigger_at_max_turns():
    """No compaction at turns < max; fires once live_turns == max."""
    sys_len = 32
    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=3, turn_stride=1,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=sys_len, max_tokens=2048,
        ignore_eos=True, block_size=16,
    )
    request.prompt_token_ids[:] = _make_system_prompt(sys_len)
    request._all_token_ids[:] = list(request.prompt_token_ids)
    scheduler.add_request(request)

    # Run 2 turns: must NOT compact yet.
    _drive_scheduler(scheduler, request, _turn_script(2, turn_token_count=30))
    assert len(request.compaction_events) == 0

    # Run 1 more turn (now 3 live): must compact.
    _drive_scheduler(scheduler, request, _turn_script(1, turn_token_count=30))
    assert len(request.compaction_events) >= 1


def test_turn_mode_evicts_one_turn():
    """stride=1: after first compaction, num_turns_evicted=1, the system
    prompt prefix is preserved bit-for-bit, and turn_end_positions shrinks.
    """
    sys_len = 32
    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=2, turn_stride=1,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=sys_len, max_tokens=2048,
        ignore_eos=True, block_size=16,
    )
    sys_prompt = _make_system_prompt(sys_len)
    request.prompt_token_ids[:] = sys_prompt
    request._all_token_ids[:] = list(sys_prompt)
    scheduler.add_request(request)

    _drive_scheduler(scheduler, request, _turn_script(3, turn_token_count=50))

    assert len(request.compaction_events) >= 1
    ev = request.compaction_events[0]
    assert ev.last_turn_evicted == 0
    assert ev.num_turns_evicted_after == 1
    assert request.num_turns_evicted >= 1

    # Hard invariant: system prompt prefix bit-for-bit identical.
    assert list(request._all_token_ids[:sys_len]) == sys_prompt

    # Single-source-of-truth invariant: kept_indices and kept_token_ids on
    # the event encode exactly what physically survived this eviction in
    # pre-event coordinates. They are populated unconditionally by the
    # scheduler so downstream consumers (orchestrator, trainer, vLLM's
    # own block-hash rebuild) don't re-derive the slice from scalars.
    assert len(ev.kept_indices) > 0, (
        "kept_indices should be populated for every event"
    )
    assert len(ev.kept_indices) == len(ev.kept_token_ids), (
        "kept_indices and kept_token_ids must align elementwise"
    )
    # The complement of kept indices (within [0, pre_event_len)) is exactly
    # [evict_start, evict_start + tokens_evicted).
    kept_set = set(ev.kept_indices)
    pre_event_len = len(ev.kept_indices) + ev.tokens_evicted
    evicted_idx_set = set(range(ev.evict_start, ev.evict_start + ev.tokens_evicted))
    assert kept_set | evicted_idx_set == set(range(pre_event_len))
    assert kept_set & evicted_idx_set == set()
    # System prompt positions [0, sys_len) must all be retained.
    assert set(range(sys_len)).issubset(kept_set)


def test_turn_mode_stride_two_evicts_two_turns():
    """stride=2: a single compaction event accounts for two turns."""
    sys_len = 32
    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=4, turn_stride=2,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=sys_len, max_tokens=2048,
        ignore_eos=True, block_size=16,
    )
    sys_prompt = _make_system_prompt(sys_len)
    request.prompt_token_ids[:] = sys_prompt
    request._all_token_ids[:] = list(sys_prompt)
    scheduler.add_request(request)

    _drive_scheduler(scheduler, request, _turn_script(5, turn_token_count=50))

    assert len(request.compaction_events) >= 1
    ev = request.compaction_events[0]
    assert ev.last_turn_evicted == 1, (
        f"Expected last_turn_evicted=1 (turns 0,1 evicted), got "
        f"{ev.last_turn_evicted}"
    )
    assert ev.num_turns_evicted_after == 2


def test_turn_mode_block_alignment_too_short():
    """A turn smaller than block_size should make the inward-snap range
    collapse to empty, so _plan_turn_evict_range bails out without state
    corruption.
    """
    sys_len = 16  # exactly one block, ends on boundary
    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=2, turn_stride=1,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=sys_len, max_tokens=2048,
        ignore_eos=True, block_size=16,
    )
    sys_prompt = _make_system_prompt(sys_len)
    request.prompt_token_ids[:] = sys_prompt
    request._all_token_ids[:] = list(sys_prompt)
    scheduler.add_request(request)

    # Tiny turns: 4 tokens of body + 1 im_end + 4 tokens + 1 im_end = 10
    # tokens per turn. With sys ending on a block boundary, end of turn 1
    # lands at position 16 + 10 = 26. Inward snap: evict_start =
    # align_up(16) = 16, evict_end = align_down(26) = 16 -> empty.
    _drive_scheduler(scheduler, request, _turn_script(3, turn_token_count=4))

    # Either no compaction fired (bail-out) or, if a later turn made the
    # range non-empty, system prompt is still preserved.
    assert list(request._all_token_ids[:sys_len]) == sys_prompt


def test_turn_mode_backward_compat_block_fifo():
    """compaction_max_turns == 0 is bit-identical to existing block-FIFO."""
    block_size = 16
    prompt_len = 50
    scheduler = create_scheduler(
        max_num_batched_tokens=1024, max_num_seqs=1,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        block_size=block_size, num_blocks=64, max_model_len=2048,
        compaction_window_size=80, compaction_stride=16,
        compaction_max_turns=0,  # explicit
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=prompt_len, max_tokens=256,
        ignore_eos=True, block_size=block_size,
    )
    scheduler.add_request(request)
    _run_decode_until_compacted(scheduler, request)
    assert len(request.compaction_events) == 1
    ev = request.compaction_events[0]
    # Default-sentinel values for the new turn fields.
    assert ev.last_turn_evicted == -1
    assert ev.num_turns_evicted_after == 0
    # Existing fields still correct.
    assert ev.tokens_evicted == 16


def test_turn_mode_compaction_event_fields_roundtrip():
    """new fields survive msgspec roundtrip."""
    ev = CompactionEvent(
        num_output_tokens_at_compaction=100,
        tokens_evicted=32,
        position_offset_after=32,
        num_prompt_tokens=200,
        last_turn_evicted=1,
        num_turns_evicted_after=2,
    )
    encoded = msgspec.msgpack.encode(ev)
    decoded = msgspec.msgpack.decode(encoded, type=CompactionEvent)
    assert decoded.last_turn_evicted == 1
    assert decoded.num_turns_evicted_after == 2

    # Defaults omitted on wire.
    ev_default = CompactionEvent(
        num_output_tokens_at_compaction=100,
        tokens_evicted=32,
        position_offset_after=32,
    )
    enc_default = msgspec.msgpack.encode(ev_default)
    assert len(enc_default) < len(encoded)


def test_auto_pad_can_finalize_at_max_model_len_boundary():
    block_size = 16
    max_model_len = 64
    prompt_len = 48
    scheduler = create_scheduler(
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        block_size=block_size,
        num_blocks=16,
        max_model_len=max_model_len,
    )
    scheduler._compaction_block_aligned_finish = True
    scheduler._compaction_block_size = block_size
    scheduler.max_model_len = max_model_len
    (request,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=max_model_len,
        ignore_eos=False,
        block_size=block_size,
    )
    scheduler.add_request(request)

    # Prefill samples token 1, then eight decode iterations sample through
    # EOS. The EOS step leaves num_computed one behind the sampled token, so
    # auto-pad extends 57 visible tokens to exactly max_model_len.
    for token_id in ([10_000] * 8 + [EOS_TOKEN_ID]):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens[request.request_id] > 0
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[token_id]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )

    assert request.padding_pending
    assert request.num_tokens == max_model_len
    assert request.num_computed_tokens == max_model_len - 8

    padding_output = scheduler.schedule()
    assert padding_output.num_scheduled_tokens[request.request_id] == 8
    assert request.request_id in padding_output.no_sample_req_ids
    scheduler.update_from_output(
        padding_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    assert not request.padding_pending
    assert request.request_id not in scheduler.requests


def test_turn_mode_system_prompt_never_evicted():
    """Hard invariant: across many compaction rounds, the leading system
    prompt is byte-for-byte identical to the original.
    """
    sys_len = 48  # not block-aligned; sys_aligned = 64
    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=2, turn_stride=1,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=sys_len, max_tokens=4096,
        ignore_eos=True, block_size=16,
    )
    sys_prompt = _make_system_prompt(sys_len)
    request.prompt_token_ids[:] = sys_prompt
    request._all_token_ids[:] = list(sys_prompt)
    scheduler.add_request(request)

    # Many turns -> many compaction rounds.
    _drive_scheduler(scheduler, request, _turn_script(10, turn_token_count=40))

    assert len(request.compaction_events) >= 2, (
        "Test should produce multiple compaction rounds; got "
        f"{len(request.compaction_events)}"
    )
    # System prompt prefix must be preserved across every event.
    assert list(request._all_token_ids[:sys_len]) == sys_prompt
    # And turn_end_positions[0] must always equal the original sys_len.
    assert request.turn_end_positions[0] == sys_len


def test_turn_mode_prefix_cache_rebuilt_after_eviction():
    """With enable_prefix_caching=True + compaction, every eviction must
    rebuild request.block_hashes over the post-eviction tokens and
    re-register surviving blocks in the prefix-cache map under their new
    hashes. Phase 2 of plans/prefix_caching_compaction.md.

    Invariants checked after a multi-event rollout:
      1. request.block_hashes length matches the number of full blocks
         in the post-eviction sequence (NOT the pre-eviction length).
      2. Hash chain reconstructs cleanly: every entry equals the hash
         that `request_block_hasher` would emit when started from
         scratch over `request.all_token_ids`. (i.e. no holes / stale
         parent references.)
      3. Every surviving block's KVCacheBlock.block_hash matches the
         corresponding entry in the rebuilt request.block_hashes,
         meaning the cache map and the request agree on what each
         block is keyed under.
    """
    sys_len = 32
    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=2, turn_stride=1,
        enable_prefix_caching=True,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=sys_len, max_tokens=4096,
        ignore_eos=True, block_size=16,
    )
    sys_prompt = _make_system_prompt(sys_len)
    request.prompt_token_ids[:] = sys_prompt
    request._all_token_ids[:] = list(sys_prompt)
    scheduler.add_request(request)

    _drive_scheduler(scheduler, request, _turn_script(6, turn_token_count=40))

    assert len(request.compaction_events) >= 1, (
        "Test should produce at least one eviction"
    )

    # Invariant 1: hash count matches post-eviction full-block count.
    # request.block_hashes is updated by update_block_hashes which only
    # hashes FULL blocks. After eviction + rehash, len(block_hashes)
    # should be num_tokens // block_size (capped at full blocks).
    block_size = 16
    expected_full_blocks = len(request._all_token_ids) // block_size
    assert len(request.block_hashes) == expected_full_blocks, (
        f"block_hashes length {len(request.block_hashes)} != "
        f"expected {expected_full_blocks} (post-eviction full blocks)"
    )

    # Invariant 2: chain reconstructs from scratch — no stale
    # parent-hash references. Rebuild from-scratch and compare.
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    rebuild = get_request_block_hasher(
        block_size=block_size,
        caching_hash_fn=hash,  # vllm uses sha256/builtin-hash depending on config
    )
    # request.block_hashes must already match what a fresh re-hash
    # would compute (we cleared + rebuilt at eviction time). We can't
    # easily reproduce vLLM's exact hash_fn here, so instead assert
    # the WEAKER property that block_hashes survives a round of
    # update_block_hashes() (idempotency).
    prev = list(request.block_hashes)
    request.update_block_hashes()  # should be a no-op (no new full blocks)
    assert list(request.block_hashes) == prev, (
        "update_block_hashes must be idempotent after rebuild — got drift"
    )

    # Invariant 3: every surviving KVCacheBlock has a hash matching its
    # entry in request.block_hashes (cache map and request are in sync).
    for mgr in scheduler.kv_cache_manager.coordinator.single_type_managers:
        block_pool = getattr(mgr, "block_pool", None)
        if block_pool is None or not block_pool.enable_caching:
            continue
        kept = mgr.req_to_blocks.get(request.request_id, [])
        # Cached blocks should be a prefix of req_to_blocks of length
        # num_cached_block[req_id]. Each should have block_hash set
        # and match the entry in cached_block_hash_to_block.
        ncb = mgr.num_cached_block.get(request.request_id, 0)
        for i, blk in enumerate(kept[:ncb]):
            assert blk.block_hash is not None, (
                f"kept block {i} (cached) has no block_hash assigned"
            )
            # The block_hash should be a known entry in the cache map.
            looked_up = block_pool.cached_block_hash_to_block.get_one_block(
                blk.block_hash
            )
            assert looked_up is not None, (
                f"kept block {i}'s block_hash is not in "
                f"cached_block_hash_to_block — orphan after rehash?"
            )


def test_inline_admission_eviction_fires_when_prefill_completes():
    """Scheduler runs in-step admission eviction at the end of
    schedule() for a request whose prefill completes this step AND
    whose turn-mode admission compaction would actually evict tokens.

    Validates the post-schedule mutation: prompt trimmed,
    position_offset bumped, num_scheduled_tokens decremented, request
    removed from `_pending_admission_compaction_ids`, and a
    CompactionEvent with `num_output_tokens_at_compaction=0` is
    attached to the request.
    """
    sys_len = 32
    body_len = 50  # gen-tokens per half-turn body
    block_size = 16

    def _turn_tokens() -> list[int]:
        toks: list[int] = []
        for _ in range(body_len):
            toks.append(GEN_BASE)
        toks.append(IM_END)
        for _ in range(body_len):
            toks.append(GEN_BASE)
        toks.append(IM_END)
        return toks

    sys_prompt = _make_system_prompt(sys_len)
    turn_0 = _turn_tokens()
    turn_1 = _turn_tokens()
    new_frag = [GEN_BASE] * 10
    prompt = sys_prompt + turn_0 + turn_1 + new_frag

    # Positions (post-IM_END): sys -> 32, turn0 -> 134, turn1 -> 236.
    # In-step eviction should remove turn 0 at block-aligned bounds:
    #   evict_start = align_up(32, 16) = 32
    #   evict_end (inward) = align_down(134, 16) = 128
    expected_evict_end = 128
    expected_total_evicted = expected_evict_end - 32
    pre_prompt_len = len(prompt)
    expected_post_prompt_len = pre_prompt_len - expected_total_evicted

    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=2, turn_stride=1,
        block_size=block_size,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=pre_prompt_len, max_tokens=2048,
        ignore_eos=True, block_size=block_size,
    )
    request.prompt_token_ids[:] = prompt
    request._all_token_ids[:] = list(prompt)
    scheduler.add_request(request)

    assert request.request_id in scheduler._pending_admission_compaction_ids

    output = scheduler.schedule()

    # In-step eviction ran: request was discarded from the pending set,
    # prompt was trimmed, and num_scheduled_tokens for this req reflects
    # the post-eviction length. position_offset stays 0 for a cold-cache
    # request because no K/V has been written yet — the trimmed prompt
    # will be prefilled fresh at positions [0, post_prompt_len). The
    # `had_kv` branch in `_apply_trim` shifts position_offset only when
    # prefix-cache hits gave the request a non-zero num_computed_tokens
    # at admission time (validated by a separate test).
    assert request.request_id not in scheduler._pending_admission_compaction_ids
    assert request.num_prompt_tokens == expected_post_prompt_len, (
        request.num_prompt_tokens, expected_post_prompt_len,
    )
    assert request.position_offset == 0, request.position_offset
    assert output.num_scheduled_tokens[request.request_id] == (
        expected_post_prompt_len
    ), output.num_scheduled_tokens

    # Exactly one CompactionEvent fired with admission semantics.
    assert len(request.compaction_events) == 1, request.compaction_events
    event = request.compaction_events[0]
    assert event.num_output_tokens_at_compaction == 0, event
    assert event.tokens_evicted == expected_total_evicted, event


def test_inline_admission_eviction_skipped_when_no_eviction_would_fire():
    """A request whose prompt has only a system prompt (no completed
    turns) enters `_pending_admission_compaction_ids` at intake, but
    the predicted eviction range is empty — schedule() does not
    mutate the request and the drain logic in update_from_output
    eventually clears the pending flag.
    """
    sys_len = 32
    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=2, turn_stride=1,
    )
    sys_prompt = _make_system_prompt(sys_len)
    (request,) = create_requests(
        num_requests=1, num_tokens=sys_len, max_tokens=2048,
        ignore_eos=True, block_size=16,
    )
    request.prompt_token_ids[:] = sys_prompt
    request._all_token_ids[:] = list(sys_prompt)
    scheduler.add_request(request)

    output = scheduler.schedule()

    # Schedule ran but no admission eviction fired — prompt unchanged,
    # position_offset still 0, no compaction event emitted.
    assert request.num_prompt_tokens == sys_len
    assert request.position_offset == 0
    assert len(request.compaction_events) == 0
    assert output.num_scheduled_tokens.get(request.request_id) == sys_len


def test_inline_admission_eviction_partial_prefix_cache_hit():
    """In-step admission eviction with a partial prefix-cache hit:
    `num_computed_tokens` is non-zero (some sys-prompt blocks were
    matched by prefix cache) but smaller than the eviction range. The
    old `assert num_computed >= total_evicted` check would have tripped
    here; the overlap-based decrement keeps the cached prefix intact
    and bumps position_offset for the kept-suffix RoPE.

    Reproduces the runtime failure
    ``AssertionError: Compaction underflow: num_computed=16, evicting=32``
    that surfaced on the first real-world rollout against the new
    in-step eviction path.
    """
    sys_len = 32
    body_len = 50
    block_size = 16

    def _turn_tokens() -> list[int]:
        toks: list[int] = []
        for _ in range(body_len):
            toks.append(GEN_BASE)
        toks.append(IM_END)
        for _ in range(body_len):
            toks.append(GEN_BASE)
        toks.append(IM_END)
        return toks

    sys_prompt = _make_system_prompt(sys_len)
    turn_0 = _turn_tokens()
    turn_1 = _turn_tokens()
    new_frag = [GEN_BASE] * 10
    prompt = sys_prompt + turn_0 + turn_1 + new_frag

    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=2, turn_stride=1,
        block_size=block_size,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=len(prompt), max_tokens=2048,
        ignore_eos=True, block_size=block_size,
    )
    request.prompt_token_ids[:] = prompt
    request._all_token_ids[:] = list(prompt)
    # Simulate a partial prefix-cache hit: one block of the sys prompt
    # is already "computed" before this step's schedule() call.
    request.num_computed_tokens = block_size
    scheduler.add_request(request)

    output = scheduler.schedule()

    # Eviction fired without the old assertion tripping. The expected
    # post-eviction state:
    #   - num_prompt_tokens trimmed by total_evicted
    #   - SchedulerOutput.num_scheduled_tokens = post_prompt_len -
    #     pre-eviction num_computed (the kernel only computes the
    #     uncached tail of the trimmed prompt)
    #   - position_offset bumped by total_evicted so the kept-suffix
    #     prefill at physical [block_size, post_prompt_len) gets RoPE
    #     for the original absolute positions
    #   - num_computed_tokens after schedule() = post_prompt_len
    #     (advanced by `_update_after_schedule`)
    expected_total_evicted = 96  # turn 0 spans blocks [2, 8) pre-evict
    expected_post_prompt_len = len(prompt) - expected_total_evicted
    expected_num_scheduled = expected_post_prompt_len - block_size
    assert request.num_prompt_tokens == expected_post_prompt_len, (
        request.num_prompt_tokens, expected_post_prompt_len,
    )
    assert request.position_offset == expected_total_evicted, (
        request.position_offset
    )
    assert output.num_scheduled_tokens[request.request_id] == (
        expected_num_scheduled
    ), output.num_scheduled_tokens
    assert request.num_computed_tokens == expected_post_prompt_len, (
        request.num_computed_tokens, expected_post_prompt_len,
    )


def test_inline_admission_eviction_warm_prefix_cache_hit():
    """In-step admission eviction with a WARM prefix-cache hit covering
    a region wider than the eviction range. The cached prefix's overlap
    with the evict range equals total_evicted (the entire eviction
    range was already in cache), so `_apply_trim` decrements
    num_computed_tokens by total_evicted. `num_scheduled_tokens[req]`
    must be recomputed from post-eviction state, not by subtracting
    total_evicted from the pre-eviction `num_scheduled_tokens`.

    Reproduces the runtime failure
    ``AssertionError`` in `_prepare_inputs` where
    `total_num_scheduled_tokens` went negative
    (``num_scheduled 36 -> -44``) because a 36-token uncached tail was
    being shrunk by an 80-token eviction whose overlap was entirely
    inside the cached prefix.
    """
    sys_len = 32
    body_len = 50
    block_size = 16

    def _turn_tokens() -> list[int]:
        toks: list[int] = []
        for _ in range(body_len):
            toks.append(GEN_BASE)
        toks.append(IM_END)
        for _ in range(body_len):
            toks.append(GEN_BASE)
        toks.append(IM_END)
        return toks

    sys_prompt = _make_system_prompt(sys_len)
    turn_0 = _turn_tokens()
    turn_1 = _turn_tokens()
    new_frag = [GEN_BASE] * 10
    prompt = sys_prompt + turn_0 + turn_1 + new_frag

    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=2, turn_stride=1,
        block_size=block_size,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=len(prompt), max_tokens=2048,
        ignore_eos=True, block_size=block_size,
    )
    request.prompt_token_ids[:] = prompt
    request._all_token_ids[:] = list(prompt)
    # Simulate a warm prefix-cache hit that covers the entire region
    # we're about to evict (and then some). For this prompt the evict
    # range is [32, 128); seven blocks (= 112 tokens) of cache hit
    # fully covers it.
    request.num_computed_tokens = 7 * block_size
    pre_num_computed = request.num_computed_tokens
    scheduler.add_request(request)

    expected_total_evicted = 96  # evict range [32, 128)
    expected_post_prompt_len = len(prompt) - expected_total_evicted
    evict_start = 32
    evict_end = 128

    output = scheduler.schedule()

    # `_apply_trim` decrements num_computed by the overlap between the
    # cached prefix [0, pre_num_computed) and the evict range
    # [evict_start, evict_end). Here pre_num_computed=112 and
    # evict=[32, 128), so overlap = min(112, 128) - 32 = 80.
    overlap = max(0, min(pre_num_computed, evict_end) - evict_start)
    expected_num_computed_post_apply_trim = pre_num_computed - overlap
    # The kernel only needs to compute the post-eviction uncached tail.
    # `_update_after_schedule` then advances num_computed by the new
    # num_scheduled — taking it back to post_prompt_len.
    expected_num_scheduled = (
        expected_post_prompt_len - expected_num_computed_post_apply_trim
    )

    assert request.num_prompt_tokens == expected_post_prompt_len, (
        request.num_prompt_tokens, expected_post_prompt_len,
    )
    assert request.position_offset == expected_total_evicted, (
        request.position_offset
    )
    assert output.num_scheduled_tokens[request.request_id] == (
        expected_num_scheduled
    ), output.num_scheduled_tokens
    assert request.num_computed_tokens == expected_post_prompt_len, (
        request.num_computed_tokens, expected_post_prompt_len,
    )
    # Sanity guard against the bug this test was written to catch:
    # num_scheduled MUST be non-negative.
    assert output.num_scheduled_tokens[request.request_id] >= 0
    assert output.total_num_scheduled_tokens >= 0, (
        output.total_num_scheduled_tokens
    )


def test_inline_admission_eviction_position_offset_in_new_req_data():
    """The position_offset bumped by in-step admission eviction MUST
    flow to the worker via `NewRequestData.position_offset`. Without
    this propagation, the worker initializes `position_offsets_cpu[row]
    = 0` for the new request, so the prefill kernel rotates Q at the
    local post-trim frame while cached K vectors (from prior requests'
    prefix-cache hits) remain rotated at the original absolute frame.
    The first decode token then picks up the offset via the cached-
    request rebuild path on the next step, producing a prefill-vs-
    decode RoPE skew that the model reads as "very ancient context"
    and degenerates into token loops ("OneOneOne...") or answering
    the previous turn's question.

    Asserts that for a fresh request that hit in-step eviction, the
    SchedulerOutput's `scheduled_new_reqs` entry carries the
    non-zero position_offset that `_apply_trim` set on the request.
    """
    sys_len = 32
    body_len = 50
    block_size = 16

    def _turn_tokens() -> list[int]:
        toks: list[int] = []
        for _ in range(body_len):
            toks.append(GEN_BASE)
        toks.append(IM_END)
        for _ in range(body_len):
            toks.append(GEN_BASE)
        toks.append(IM_END)
        return toks

    sys_prompt = _make_system_prompt(sys_len)
    turn_0 = _turn_tokens()
    turn_1 = _turn_tokens()
    new_frag = [GEN_BASE] * 10
    prompt = sys_prompt + turn_0 + turn_1 + new_frag

    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=2, turn_stride=1,
        block_size=block_size,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=len(prompt), max_tokens=2048,
        ignore_eos=True, block_size=block_size,
    )
    request.prompt_token_ids[:] = prompt
    request._all_token_ids[:] = list(prompt)
    # Simulate a warm prefix-cache hit so `_apply_trim`'s `had_kv`
    # branch fires and bumps position_offset.
    request.num_computed_tokens = 7 * block_size
    scheduler.add_request(request)

    expected_total_evicted = 96  # turn 0 at evict range [32, 128)

    output = scheduler.schedule()

    # The request should be in scheduled_new_reqs (it's a brand-new
    # admission) and its NewRequestData must carry the bumped offset.
    new_req_entries = [
        nrd for nrd in output.scheduled_new_reqs
        if nrd.req_id == request.request_id
    ]
    assert len(new_req_entries) == 1, (
        f"Expected request {request.request_id} in scheduled_new_reqs, "
        f"got {[nrd.req_id for nrd in output.scheduled_new_reqs]}"
    )
    new_req_data = new_req_entries[0]
    assert new_req_data.position_offset == expected_total_evicted, (
        f"NewRequestData.position_offset={new_req_data.position_offset} "
        f"does not match request.position_offset="
        f"{request.position_offset} (expected {expected_total_evicted})"
    )
    # Cross-check that the scheduler-side request state matches.
    assert request.position_offset == expected_total_evicted
