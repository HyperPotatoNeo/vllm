# SPDX-License-Identifier: Apache-2.0
"""AM-specific prefix-cache helpers.

These helpers intentionally avoid importing scheduler or worker classes so the
same key material is used on both sides of the vLLM V1 boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from collections.abc import Sequence

from vllm.v1.core.compaction.am_runtime import (
    AttentionMatchingCompactionPlan,
    advance_attention_matching_turn_boundary,
    build_attention_matching_turn_plan,
)


@dataclass(frozen=True)
class AttentionMatchingPrefixCacheStep:
    """One deterministic AM turn-window cache identity update."""

    plan: AttentionMatchingCompactionPlan
    compacted_tokens_hash: str
    query_seed: int
    prefix_cache_key: str
    parent_key: str | None
    parent_key_start: int
    position_offset_before: int
    position_offset_after: int
    # Full physical prompt after applying this step and appending any original
    # prompt tokens not yet processed by the replay.  This is the request shape
    # used for compressed-prefix admission of a shallow replay step.
    physical_token_ids_after: tuple[int, ...]


@dataclass(frozen=True)
class AttentionMatchingPrefixCacheReplay:
    """Result of replaying turn-window AM cache identities over a prompt."""

    physical_token_ids: tuple[int, ...]
    position_offset: int
    steps: tuple[AttentionMatchingPrefixCacheStep, ...]

    @property
    def final_step(self) -> AttentionMatchingPrefixCacheStep | None:
        if not self.steps:
            return None
        return self.steps[-1]


def hash_attention_matching_tokens(token_ids: Sequence[int]) -> str:
    """Stable compact hash for token regions summarized by AM."""
    digest = hashlib.blake2b(digest_size=16)
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def build_cross_turn_query_seed(
    *,
    base_seed: int,
    cache_salt: str | None,
    compacted_tokens_hash: str,
    protected_prefix_len: int,
    synthetic_prefix_len: int,
    query_source: str,
    max_queries_per_kv_head: int,
    zerobeta: bool,
    parent_key: str | None,
    position_offset_before: int = 0,
    forget_gate_enabled: bool = False,
    forget_gate_alpha: float = 0.5,
) -> int:
    """Build a request-independent AM query seed for cacheable memory.

    Random-query AM remains stochastic in distribution, but a given compacted
    source state must deterministically produce the same synthetic KV if it is
    going to be reused across turns and requests.
    """
    if forget_gate_enabled:
        key_material = (
            "attention_matching_cross_turn_query_seed_v2",
            int(base_seed),
            cache_salt,
            compacted_tokens_hash,
            protected_prefix_len,
            synthetic_prefix_len,
            query_source,
            max_queries_per_kv_head,
            zerobeta,
            parent_key,
            int(position_offset_before),
            round(float(forget_gate_alpha), 10),
        )
    else:
        key_material = (
            "attention_matching_cross_turn_query_seed_v1",
            int(base_seed),
            cache_salt,
            compacted_tokens_hash,
            protected_prefix_len,
            synthetic_prefix_len,
            query_source,
            max_queries_per_kv_head,
            zerobeta,
            parent_key,
            int(position_offset_before),
        )
    digest = hashlib.blake2b(repr(key_material).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


def build_attention_matching_prefix_cache_key(
    *,
    cache_salt: str | None,
    protected_prefix_len: int,
    synthetic_prefix_len: int,
    compacted_tokens_hash: str,
    query_source: str,
    max_queries_per_kv_head: int,
    query_seed: int,
    zerobeta: bool,
    parent_key: str | None,
    parent_key_start: int,
    position_offset_before: int = 0,
    position_offset_after: int = 0,
    tail_signature: tuple[int, int, int] | None = None,
    forget_gate_enabled: bool = False,
    forget_gate_alpha: float = 0.5,
) -> str:
    """Build the namespace for AM synthetic memory.

    For the cross-turn path, ``tail_signature`` is intentionally ``None``. The
    synthetic KV depends on the compacted source region and AM parameters, while
    exact tail tokens are still hashed normally after the synthetic block chain.

    For conservative modes or query sources that depend on the exact tail,
    callers may pass a tail signature to prevent over-reuse.
    """
    if forget_gate_enabled:
        key_material = (
            "attention_matching_prefix_cache_v3",
            cache_salt,
            protected_prefix_len,
            synthetic_prefix_len,
            compacted_tokens_hash,
            query_source,
            max_queries_per_kv_head,
            query_seed,
            zerobeta,
            parent_key,
            parent_key_start,
            int(position_offset_before),
            int(position_offset_after),
            tail_signature,
            round(float(forget_gate_alpha), 10),
        )
    else:
        key_material = (
            "attention_matching_prefix_cache_v2",
            cache_salt,
            protected_prefix_len,
            synthetic_prefix_len,
            compacted_tokens_hash,
            query_source,
            max_queries_per_kv_head,
            query_seed,
            zerobeta,
            parent_key,
            parent_key_start,
            int(position_offset_before),
            int(position_offset_after),
            tail_signature,
        )
    return hashlib.blake2b(repr(key_material).encode(), digest_size=32).hexdigest()


def _iter_turn_chunks(
    token_ids: Sequence[int],
    *,
    turn_end_token_id: int,
    turn_padding_token_id: int | None,
) -> list[list[int]]:
    """Split rendered prompt tokens at message boundaries."""
    chunks: list[list[int]] = []
    start = 0
    pos = 0
    while pos < len(token_ids):
        if token_ids[pos] != turn_end_token_id:
            pos += 1
            continue

        boundary = advance_attention_matching_turn_boundary(
            token_ids,
            pos + 1,
            len(token_ids),
            turn_padding_token_id,
        )
        chunks.append([int(t) for t in token_ids[start:boundary]])
        start = boundary
        pos = boundary

    if start < len(token_ids):
        chunks.append([int(t) for t in token_ids[start:]])
    return chunks


def build_attention_matching_turn_prefix_cache_replay(
    *,
    token_ids: Sequence[int],
    base_seed: int,
    cache_salt: str | None,
    synthetic_prefix_len: int,
    max_turns: int,
    keep_recent_turns: int,
    turn_end_token_id: int | None,
    turn_padding_token_id: int | None,
    protect_first_user: bool,
    query_source: str,
    max_queries_per_kv_head: int,
    zerobeta: bool,
    forget_gate_enabled: bool = False,
    forget_gate_alpha: float = 0.5,
    min_protected_prefix_len: int = 0,
    initial_parent_key: str | None = None,
    initial_parent_key_start: int = 0,
    initial_position_offset: int = 0,
) -> AttentionMatchingPrefixCacheReplay | None:
    """Replay chained turn-window AM cache identities for a full prompt.

    Cross-turn compressed admission must ask the prefix cache for the same
    AM namespace that a previous request wrote. After the first AM rewrite,
    later rewrites summarize the already-compressed physical prefix, not the
    original raw text. Replaying message-boundary compactions locally recovers
    that parent-key chain while staying independent of scheduler/worker state.
    """
    if turn_end_token_id is None:
        return None

    physical_token_ids: list[int] = []
    parent_key: str | None = initial_parent_key
    parent_key_start = int(initial_parent_key_start)
    position_offset = int(initial_position_offset)
    steps: list[AttentionMatchingPrefixCacheStep] = []

    original_pos = 0
    for chunk in _iter_turn_chunks(
        token_ids,
        turn_end_token_id=turn_end_token_id,
        turn_padding_token_id=turn_padding_token_id,
    ):
        original_pos += len(chunk)
        physical_token_ids.extend(chunk)
        while True:
            plan = build_attention_matching_turn_plan(
                num_computed_tokens=len(physical_token_ids),
                synthetic_prefix_len=synthetic_prefix_len,
                token_ids=physical_token_ids,
                max_turns=max_turns,
                keep_recent_turns=keep_recent_turns,
                turn_end_token_id=turn_end_token_id,
                turn_padding_token_id=turn_padding_token_id,
                protect_first_user=protect_first_user,
                min_protected_prefix_len=min_protected_prefix_len,
            )
            if plan is None:
                break

            compacted_tokens_hash = hash_attention_matching_tokens(
                physical_token_ids[
                    plan.protected_prefix_len : plan.source_len
                    - plan.exact_kept_tokens
                ]
            )
            query_seed = build_cross_turn_query_seed(
                base_seed=base_seed,
                cache_salt=cache_salt,
                compacted_tokens_hash=compacted_tokens_hash,
                protected_prefix_len=plan.protected_prefix_len,
                synthetic_prefix_len=plan.synthetic_prefix_len,
                query_source=query_source,
                max_queries_per_kv_head=max_queries_per_kv_head,
                zerobeta=zerobeta,
                parent_key=parent_key,
                position_offset_before=position_offset,
                forget_gate_enabled=forget_gate_enabled,
                forget_gate_alpha=forget_gate_alpha,
            )
            position_offset_before = position_offset
            position_offset_after = position_offset + plan.offset_delta
            prefix_cache_key = build_attention_matching_prefix_cache_key(
                cache_salt=cache_salt,
                protected_prefix_len=plan.protected_prefix_len,
                synthetic_prefix_len=plan.synthetic_prefix_len,
                compacted_tokens_hash=compacted_tokens_hash,
                query_source=query_source,
                max_queries_per_kv_head=max_queries_per_kv_head,
                query_seed=query_seed,
                zerobeta=zerobeta,
                parent_key=parent_key,
                parent_key_start=parent_key_start,
                position_offset_before=position_offset_before,
                position_offset_after=position_offset_after,
                tail_signature=None,
                forget_gate_enabled=forget_gate_enabled,
                forget_gate_alpha=forget_gate_alpha,
            )

            position_offset = position_offset_after
            next_physical_token_ids = (
                physical_token_ids[: plan.protected_prefix_len]
                + [0] * plan.synthetic_prefix_len
                + physical_token_ids[plan.exact_region_start : plan.source_len]
            )
            full_physical_token_ids_after = (
                next_physical_token_ids
                + [int(t) for t in token_ids[original_pos:]]
            )
            steps.append(
                AttentionMatchingPrefixCacheStep(
                    plan=plan,
                    compacted_tokens_hash=compacted_tokens_hash,
                    query_seed=query_seed,
                    prefix_cache_key=prefix_cache_key,
                    parent_key=parent_key,
                    parent_key_start=parent_key_start,
                    position_offset_before=position_offset_before,
                    position_offset_after=position_offset,
                    physical_token_ids_after=tuple(full_physical_token_ids_after),
                )
            )

            physical_token_ids = next_physical_token_ids
            parent_key = prefix_cache_key
            parent_key_start = plan.protected_prefix_len

    return AttentionMatchingPrefixCacheReplay(
        physical_token_ids=tuple(physical_token_ids),
        position_offset=position_offset,
        steps=tuple(steps),
    )
