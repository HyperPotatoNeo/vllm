# SPDX-License-Identifier: Apache-2.0
"""
Attention-matching compaction manager scaffold.

This is intentionally a separate opt-in codepath. The current FIFO
CompactingKVCacheManager remains the default and is unchanged.

The AM path is wired at the config/coordinator boundary so future work can
attach attention-matching-specific state and worker interactions without
mutating the FIFO path. Actual AM compaction happens on the worker, which
rewrites the live KV cache and reports the new logical length back to the
scheduler. The scheduler then calls `finalize_compaction` here to free the
tail blocks and splice the request's block table.
"""

from vllm.v1.core.compaction.manager import CompactingKVCacheManager


class AttentionMatchingKVCacheManager(CompactingKVCacheManager):
    """Reserved manager for the attention-matching baseline."""

    strategy_name = "attention_matching"

    def needs_compaction(
        self,
        request_id: str,
        num_computed_tokens: int,
        prompt_tokens: int,
    ) -> bool:
        # The worker owns the AM trigger and live-KV rewrite. Returning False
        # here prevents the FIFO scheduler path from touching AM requests.
        return False

    def compact_request(self, request_id: str, prompt_tokens: int) -> int:
        raise RuntimeError(
            "attention_matching compaction is worker-driven; "
            "Scheduler._compact_request should never call compact_request() "
            "for AttentionMatchingKVCacheManager."
        )

    def finalize_compaction(self, request_id: str, new_num_computed_tokens: int) -> int:
        """Free the physical tail blocks after a worker-side AM rewrite."""
        blocks = self.req_to_blocks[request_id]
        new_num_blocks = (new_num_computed_tokens + self.block_size - 1) // self.block_size
        if new_num_blocks >= len(blocks):
            return 0
        evicted_blocks = blocks[new_num_blocks:]
        self.block_pool.free_blocks(evicted_blocks)
        del blocks[new_num_blocks:]
        return len(evicted_blocks) * self.block_size

    def set_cached_prefix_blocks(self, request_id: str, num_cached_blocks: int) -> None:
        """Reset prefix-cache bookkeeping after an in-place AM KV rewrite.

        AM can overwrite middle/tail blocks with synthetic KV. Protected prefix
        blocks remain valid, but all mutable blocks must be treated as uncached
        so they can be reinserted under AM-specific block hashes.
        """
        blocks = self.req_to_blocks[request_id]
        self.num_cached_block[request_id] = min(
            max(num_cached_blocks, 0),
            len(blocks),
        )

    def privatize_shared_blocks(
        self, request_id: str, num_tokens: int
    ) -> tuple[list[int], list[int]]:
        """Copy-on-write shared prefix-cache blocks before AM may mutate them.

        Prefix-cache hits intentionally share physical blocks across requests.
        AM later rewrites KV in place, so any shared blocks in the live request
        must be replaced by fresh private blocks before the worker can mutate
        them. The worker receives the returned src/dst block IDs and copies KV.
        """
        blocks = self.req_to_blocks[request_id]
        num_blocks = min(
            (num_tokens + self.block_size - 1) // self.block_size,
            len(blocks),
        )
        if num_blocks <= 0:
            return [], []

        src_block_ids: list[int] = []
        dst_block_ids: list[int] = []
        first_replaced_idx: int | None = None
        for idx in range(num_blocks):
            block = blocks[idx]
            if block.is_null or block.ref_cnt <= 1:
                continue
            new_block = self.block_pool.get_new_blocks(1)[0]
            blocks[idx] = new_block
            self.block_pool.free_blocks([block])
            src_block_ids.append(block.block_id)
            dst_block_ids.append(new_block.block_id)
            if first_replaced_idx is None:
                first_replaced_idx = idx

        if first_replaced_idx is not None:
            self.num_cached_block[request_id] = min(
                self.num_cached_block.get(request_id, 0),
                first_replaced_idx,
            )
        return src_block_ids, dst_block_ids

    def clear_request_block_hashes(self, request_id: str) -> int:
        """Remove stale prefix-cache hashes from all blocks owned by a request.

        Cross-turn AM can reuse cached blocks and then rewrite the request's KV
        layout in place.  Those physical blocks must be treated as fresh blocks
        before the AM-specific prefix-cache key is written; otherwise vLLM may
        try to cache a block that still carries an old token-hash namespace.
        """
        cleared = 0
        for block in self.req_to_blocks[request_id]:
            if block.is_null or block.block_hash is None:
                continue
            self.block_pool._maybe_evict_cached_block(block)
            if block.block_hash is not None:
                block.reset_hash()
            cleared += 1
        return cleared
