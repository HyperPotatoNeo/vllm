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

KNOWN INCOMPATIBILITIES:
- LMCache KV transfer: refused at Scheduler startup. LMCache's V1 adapter
  (`lmcache_integration/vllm_v1_adapter.py`) reads
  `request._output_token_ids[0]` as a "first_tok" fingerprint for the KV
  transfer protocol, and compaction invalidates that index. Combining them
  raises AssertionError in Scheduler.__init__.
- Async scheduling: refused at Scheduler startup. num_output_placeholders is
  nonzero whenever update_from_output runs, so the compaction trigger would
  never satisfy, silently degrading to full-context.
- num_cached_tokens (stats-only metric; metrics/stats.py, logging.py) is NOT
  decremented when compaction removes tokens from the logical view. The
  reported cached-token counts drift relative to num_computed_tokens after
  compaction. No correctness impact on the scheduler/worker state.
"""

from collections.abc import Callable

# Re-export CompactionEvent from the dependency-free types module so
# existing callers (scheduler.py, package __init__, tests) keep working.
from vllm.v1.core.compaction.types import CompactionEvent  # noqa: F401
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager


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
        """Check if compaction should fire for this request.

        Two guards:
        1. Window: num_computed_tokens must exceed the user-configured window.
        2. Full-block safety: the LAST block that would be evicted must be
           fully filled. We report `tokens_evicted = stride_blocks * block_size`
           to the scheduler, which decrements num_computed_tokens by that
           amount. If the last evicted block holds fewer than block_size real
           tokens (e.g. compaction fires right when the first post-prompt gen
           block has just 1 token), the scheduler would over-decrement
           num_computed_tokens, desynchronizing it from the physical KV and
           causing the next forward pass to re-compute prompt tokens (and
           silently deleting any just-sampled-but-not-yet-forwarded token from
           _all_token_ids during the trim).

        The full-block guard is: num_computed_tokens >= (prompt_blocks +
        stride_blocks) * block_size. This is because block index
        (prompt_blocks + stride_blocks - 1) — the last block we'd evict —
        is fully filled iff num_computed_tokens has reached slot
        (prompt_blocks + stride_blocks) * block_size.

        In realistic configs (window_size >> stride_blocks * block_size) the
        window guard dominates and the full-block guard is a no-op. It
        matters when window_size is configured close to prompt_len.
        """
        if self.compaction_window_size <= 0:
            return False
        if num_computed_tokens <= self.compaction_window_size:
            return False
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None:
            return False
        prompt_blocks = (prompt_tokens + self.block_size - 1) // self.block_size
        required_full_length = (prompt_blocks + self.stride_blocks) * self.block_size
        return num_computed_tokens >= required_full_length

    def compact_request(
        self,
        request_id: str,
        prompt_tokens: int,
        explicit_block_range: tuple[int, int] | None = None,
    ) -> int:
        """Evict oldest post-prompt blocks. Returns tokens evicted.

        Steps:
        1. Determine which block indices to evict (FIFO, custom, or explicit
           block range for turn-mode eviction)
        2. Free those blocks back to pool
        3. Splice req_to_blocks (delete entries)
        4. Return tokens evicted

        explicit_block_range overrides both stride_blocks and prompt_tokens
        when set: evicts blocks[start:end] verbatim. Used by turn-mode
        compaction in the scheduler, which computes its own block-aligned
        range from completed-turn boundaries. prompt_tokens is still passed
        for symmetry but ignored in this path.
        """
        blocks = self.req_to_blocks[request_id]
        prompt_blocks = (prompt_tokens + self.block_size - 1) // self.block_size

        if explicit_block_range is not None:
            start, end = explicit_block_range
            assert 0 <= start <= end <= len(blocks), (
                f"explicit_block_range=({start},{end}) out of bounds for "
                f"req={request_id[:8]} (len={len(blocks)})"
            )
            evict_indices = list(range(start, end))
        elif self.eviction_fn is not None:
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
