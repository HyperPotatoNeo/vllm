# SPDX-License-Identifier: Apache-2.0
"""
CompactingKVCacheManager: block-level KV cache compaction for vLLM V1.

Extends FullAttentionManager to support evicting the oldest post-prompt blocks
when a request's KV length exceeds a configurable window. Eviction is FIFO
by default (oldest generation blocks first), with an optional callable for
custom strategies.

After compaction, the scheduler trims the request's token IDs and reduces
num_computed_tokens so every vLLM consumer sees a consistent shorter sequence.
Only position_offset (for RoPE correction) needs special handling.
"""

from collections.abc import Callable
from dataclasses import dataclass

from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager


@dataclass(frozen=True)
class CompactionEvent:
    """Record of a single compaction event. Stored on Request."""

    num_output_tokens_at_compaction: int  # total generated when this fired
    tokens_evicted: int
    blocks_evicted: int
    position_offset_after: int  # cumulative offset after this event


class CompactingKVCacheManager(FullAttentionManager):
    """FullAttentionManager with block-level KV cache compaction.

    When compaction_window_size > 0, this manager supports evicting
    stride_blocks oldest post-prompt blocks via compact_request().
    """

    def __init__(
        self,
        *args,
        compaction_window_size: int = 0,
        compaction_stride: int = 0,
        eviction_fn: Callable[[int, int, int], list[int]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compaction_window_size = compaction_window_size
        self.compaction_stride = compaction_stride
        self.eviction_fn = eviction_fn  # None = default FIFO
        if compaction_stride > 0:
            assert compaction_stride % self.block_size == 0, (
                f"compaction_stride ({compaction_stride}) must be a multiple "
                f"of block_size ({self.block_size})"
            )

    @property
    def stride_blocks(self) -> int:
        return self.compaction_stride // self.block_size

    def needs_compaction(
        self,
        request_id: str,
        num_computed_tokens: int,
        prompt_tokens: int,
    ) -> bool:
        """Check if compaction should fire for this request."""
        if self.compaction_window_size <= 0:
            return False
        if num_computed_tokens <= self.compaction_window_size:
            return False
        # Only compact if enough generation blocks exist to evict
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None:
            return False
        prompt_blocks = (prompt_tokens + self.block_size - 1) // self.block_size
        gen_blocks = len(blocks) - prompt_blocks
        return gen_blocks >= self.stride_blocks

    def compact_request(
        self, request_id: str, prompt_tokens: int
    ) -> int:
        """Evict oldest post-prompt blocks. Returns tokens evicted.

        Steps:
        1. Determine which block indices to evict (FIFO or custom)
        2. Free those blocks back to pool
        3. Splice req_to_blocks (delete entries)
        4. Return tokens evicted
        """
        blocks = self.req_to_blocks[request_id]
        prompt_blocks = (prompt_tokens + self.block_size - 1) // self.block_size

        if self.eviction_fn is not None:
            evict_indices = self.eviction_fn(
                len(blocks), prompt_blocks, self.stride_blocks
            )
        else:
            # Inline FIFO: evict oldest post-prompt blocks
            gen_blocks = len(blocks) - prompt_blocks
            actual = min(self.stride_blocks, gen_blocks)
            evict_indices = list(range(prompt_blocks, prompt_blocks + actual))

        if not evict_indices:
            return 0

        # Free evicted blocks to pool
        evicted_blocks = [blocks[i] for i in evict_indices]
        self.block_pool.free_blocks(evicted_blocks)

        # Splice: contiguous range (FIFO) uses fast slice deletion
        start, end = evict_indices[0], evict_indices[-1] + 1
        if end - start == len(evict_indices):
            del blocks[start:end]
        else:
            # Non-contiguous (custom eviction_fn)
            for i in sorted(evict_indices, reverse=True):
                del blocks[i]

        return len(evict_indices) * self.block_size
