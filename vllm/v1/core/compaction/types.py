# SPDX-License-Identifier: Apache-2.0
"""Wire types for KV cache compaction.

This module is intentionally dependency-free (only msgspec) so that both
vllm.v1.engine and vllm.v1.core can import from it without creating a
circular import. The heavier CompactingKVCacheManager lives in manager.py
and imports from here.
"""

import msgspec


class CompactionEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Record of a single compaction event. Attached to a Request and
    forwarded to the inference client through EngineCoreOutput /
    ChatCompletionResponse so training can replay the KV drop schedule.

    Wire format: msgspec.Struct(array_like=True) — serialized as a compact
    positional array. omit_defaults lets us add optional fields later (e.g.
    evicted_block_indices for non-FIFO eviction strategies) without breaking
    wire compatibility.
    """

    # Monotonic count of tokens generated at the moment this event fired.
    # This is the cumulative boundary in completion-token space where the
    # trainer's segmented_forward should drop KV between segments.
    num_output_tokens_at_compaction: int

    # How many tokens were physically evicted from the KV cache. For FIFO
    # sliding-window eviction this equals stride_blocks * block_size.
    tokens_evicted: int

    # Cumulative position_offset after this event. RoPE of a new query at
    # physical position P = P + position_offset, recovering the absolute
    # logical position from before eviction.
    position_offset_after: int

    # Prompt length of the request when this event fired. For standard
    # (full-prompt-protected) compaction this equals the original prompt.
    # For protected-prefix eviction this shrinks as prompt tokens get
    # evicted. Needed by the trainer to compute per-event eviction
    # boundaries during replay. Default 0 for wire compat (omit_defaults).
    num_prompt_tokens: int = 0

    # Start position of the eviction range in the CURRENT (possibly
    # already-partially-trimmed) token sequence. For admission-time events
    # (num_output_tokens_at_compaction == 0) this is the position in the
    # prompt; for mid-gen events it's in _all_token_ids. The orchestrator
    # uses this to replay the exact same token deletion before training.
    evict_start: int = 0

    # Debug-only: the actual token IDs that were evicted, in order. Populated
    # only when VLLM_COMPACTION_DEBUG_TOKENS=1 is set in the scheduler's env.
    # Empty by default so omit_defaults drops it on the wire for production
    # runs. Lets consumers detokenize and inspect exactly what was removed.
    evicted_token_ids: list[int] = msgspec.field(default_factory=list)

    # Turn mode: index of the last turn that was evicted by this event
    # (0-indexed, inclusive — turn 0 is the first user+assistant pair after
    # the system prompt). Defaults to -1 in block-FIFO mode (no turn
    # tracking).
    last_turn_evicted: int = -1

    # Turn mode: cumulative number of turns physically evicted on the
    # request after this event. Defaults to 0 in block-FIFO mode.
    num_turns_evicted_after: int = 0

    # Indices (in pre-event/CURRENT coordinates at the moment this event
    # fires) of tokens that physically survive this eviction. Sorted
    # ascending; length = (pre-event token count) - tokens_evicted.
    # This is vLLM's canonical "what's left in KV after this event" view,
    # in the same coordinate frame the consumer is in at event time
    # (post-prior-events, pre-this-event). Empty by default for wire
    # compat with downstream code that hasn't been updated yet, but
    # populated unconditionally by the scheduler.
    kept_indices: list[int] = msgspec.field(default_factory=list)

    # Token IDs at the kept_indices positions, in order. Same length as
    # kept_indices. Lets the orchestrator assemble the next turn's
    # prompt as [sys, kept_token_ids, u_new] to hit vLLM's rebuilt
    # prefix cache, and lets the trainer splice its KV by token rather
    # than re-deriving the eviction range from scalar fields.
    kept_token_ids: list[int] = msgspec.field(default_factory=list)

    # Length of the new_user_fragment in this admission event — the
    # tail of the prompt that lies AFTER the last completed turn
    # (typically the in-progress turn's user message + assistant
    # template). vLLM's single-forward pre-eviction path no longer
    # uses this value internally (kept turns and new_user_fragment
    # both attend over post-eviction K/V), but it is still emitted so
    # the trainer can split each admission boundary into pre- and
    # post-fragment segments when its segmented_forward mirror needs
    # the boundary. Default 0 for wire compat (omit_defaults) and for
    # mid-gen events that don't expose a fragment boundary.
    new_user_fragment_len: int = 0

    # Managed-context extension: scheduler-local IDs for archived KV spans
    # captured during this eviction. Empty unless KVE_MANAGED_CONTEXT=1 and
    # the request is Phase4/turn-compaction eligible.
    archived_span_ids: list[str] = msgspec.field(default_factory=list)

    # Full replay fallback: length of the untrimmed writer timeline when
    # this eviction fired. This is the exact death index for the evicted
    # KV rows under the compact replay mask. Appended for array_like wire
    # compatibility with older CompactionEvent readers.
    writer_len_at_compaction: int = 0

    # Managed-context extension: per-span [start, end) bounds for each
    # entry in archived_span_ids, flattened pairs in the SAME pre-event
    # coordinate frame as evict_start/kept_indices. Length is always
    # 2 * len(archived_span_ids). Lets a consumer map every archived span
    # to the exact rows that died at this event (per-turn span splitting
    # archives several sub-ranges of [evict_start, evict_end) under one
    # event), which is what later restore events reference by span_id.
    archived_span_bounds: list[int] = msgspec.field(default_factory=list)

    # Managed-context restore lifecycle. 0 = eviction (default).
    # 1 = hidden-restore ATTACH: restored_span_ids became visible to
    # subsequent queries. 2 = restore RELEASE: they left visibility.
    # For kind 1/2 events the eviction fields are zeros/defaults;
    # num_output_tokens_at_compaction, num_prompt_tokens and
    # writer_len_at_compaction carry the timing frame, and
    # visibility_boundary_computed pins the exact query boundary.
    event_kind: int = 0

    # kind 1/2 only: span ids attached/released by this event, in span
    # order. Resolve each id to its token rows via the archiving eviction
    # event's archived_span_ids + archived_span_bounds.
    restored_span_ids: list[str] = msgspec.field(default_factory=list)

    # kind 1/2 only: the request's num_computed_tokens (current-frame) at
    # the moment of the visibility change. Queries at current positions
    # >= this value see (kind 1) / no longer see (kind 2) the restored
    # spans. Distinct from writer_len_at_compaction when the change lands
    # mid-prefill (deferred restore): prompt tokens past this boundary
    # attend to the spans even though they were already in the token list
    # when the restore attached. -1 on eviction events.
    visibility_boundary_computed: int = -1

    # kind 1 only, recall token-surfacing (KVE_RECALL_SURFACE_TOKENS): the
    # token ids of the SINGLE restored span, so a trainer whose sample never
    # saw the span's birth (the span was evicted on a side-channel handshake
    # call before this sample's first captured turn) can reconstruct its KV
    # by forwarding these tokens at restored_span_pos_start.. positions.
    # Empty when surfacing is off or the span's rows are already in-sample.
    restored_span_token_ids: list[int] = msgspec.field(default_factory=list)

    # kind 1 only: absolute RoPE position of restored_span_token_ids[0]
    # (the rest follow contiguously). This is the position the engine used
    # when it first computed the span's KV; the trainer must reuse it so the
    # recalled keys carry the same rotation the sampler attended to. -1 when
    # no tokens are surfaced.
    restored_span_pos_start: int = -1

    # KV-selection extension: explicit pre-event token ranges evicted by this
    # event, flattened as [start0, end0, start1, end1, ...]. Empty for legacy
    # contiguous events, where evict_start/tokens_evicted remains sufficient.
    evicted_ranges: list[int] = msgspec.field(default_factory=list)

    # KV-selection extension: absolute turn ids in the candidate band, the
    # subset kept by the selector, and the subset physically evicted.
    selection_candidate_turn_indices: list[int] = msgspec.field(default_factory=list)
    selection_kept_turn_indices: list[int] = msgspec.field(default_factory=list)
    selection_evicted_turn_indices: list[int] = msgspec.field(default_factory=list)
