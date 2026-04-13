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
    compaction_window_size: int = 256,
    compaction_stride: int = 16,
    max_turns: int = 2,
    turn_stride: int = 1,
):
    return create_scheduler(
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
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
        compaction_stride=16, compaction_window_size=64,
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
        compaction_window_size=128, compaction_stride=16,
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


def test_turn_mode_stride_two_evicts_two_turns():
    """stride=2: a single compaction event accounts for two turns."""
    sys_len = 32
    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=4, turn_stride=2,
        compaction_window_size=128, compaction_stride=16,
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
        compaction_window_size=64, compaction_stride=16,
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


def test_turn_mode_system_prompt_never_evicted():
    """Hard invariant: across many compaction rounds, the leading system
    prompt is byte-for-byte identical to the original.
    """
    sys_len = 48  # not block-aligned; sys_aligned = 64
    scheduler = _make_turn_scheduler(
        sys_len=sys_len, max_turns=2, turn_stride=1,
        compaction_window_size=128, compaction_stride=16,
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
