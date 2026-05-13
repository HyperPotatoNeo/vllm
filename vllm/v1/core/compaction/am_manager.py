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
