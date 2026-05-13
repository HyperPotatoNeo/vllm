# SPDX-License-Identifier: Apache-2.0
# This package is intentionally minimal at the top level to avoid eagerly
# importing CompactingKVCacheManager (which pulls in FullAttentionManager
# and, transitively, vllm.v1.engine). That chain creates a circular import
# when vllm.v1.engine tries to load CompactionEvent for its own msgspec
# struct definitions. Import from submodules directly:
#     from vllm.v1.core.compaction.types import CompactionEvent
#     from vllm.v1.core.compaction.manager import CompactingKVCacheManager
#     from vllm.v1.core.compaction.am_manager import AttentionMatchingKVCacheManager
