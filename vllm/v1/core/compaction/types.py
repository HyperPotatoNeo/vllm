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
