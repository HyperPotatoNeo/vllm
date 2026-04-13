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
