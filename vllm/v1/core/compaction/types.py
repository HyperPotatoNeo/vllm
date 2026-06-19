# SPDX-License-Identifier: Apache-2.0
"""Wire types for KV cache compaction.

This module is intentionally dependency-free (only msgspec) so that both
vllm.v1.engine and vllm.v1.core can import from it without creating a
circular import. The heavier CompactingKVCacheManager lives in manager.py
and imports from here.
"""

from typing import Any

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

    # Optional trailing metadata. Defaults preserve compatibility with older
    # three-field FIFO events while carrying enough state for trainer-side AM
    # replay when compaction_strategy="attention_matching".
    num_prompt_tokens: int = 0
    evict_start: int = 0
    compaction_strategy: str = "fifo"
    source_len: int = 0
    target_len: int = 0
    protected_prefix_len: int = 0
    synthetic_prefix_len: int = 0
    exact_kept_tokens: int = 0
    attention_matching_query_source: str = ""
    attention_matching_max_queries_per_kv_head: int = 0
    attention_matching_query_seed: int = 0
    attention_matching_zerobeta: bool = False
    attention_matching_pre_sample: bool = False
    attention_matching_replay_steps: list[dict[str, Any]] | None = None
    attention_matching_cache_hit_tokens: int = 0
    attention_matching_selected_indices: list[list[list[int]]] | None = None
    attention_matching_forget_gate_enabled: bool = False
    attention_matching_forget_gate_alpha: float = 0.5
    attention_matching_forget_gate_applied: bool = False
    attention_matching_hidden_tail_token_ids: list[int] | None = None
