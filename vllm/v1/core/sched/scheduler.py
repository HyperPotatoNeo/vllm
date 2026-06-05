# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsReader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    ManagedContextCopyEvent,
    ManagedContextTransferMetadata,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import (
    DraftTokenIds,
    KVConnectorOutput,
    ManagedContextTransferOutput,
    ModelRunnerOutput,
)
from vllm.v1.core.compaction.manager import CompactingKVCacheManager
from vllm.v1.core.compaction.types import CompactionEvent
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.utils import ConstantList
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


def _pop_contiguous_managed_context_cpu_blocks(
    free_block_ids: deque[int],
    num_blocks: int,
) -> list[int] | None:
    """Pop a contiguous increasing run of CPU archive block IDs if available."""
    if num_blocks <= 0 or len(free_block_ids) < num_blocks:
        return None

    free_set = {int(block_id) for block_id in free_block_ids}
    for start in sorted(free_set):
        run = list(range(start, start + num_blocks))
        if all(block_id in free_set for block_id in run):
            chosen = set(run)
            remaining = deque(
                int(block_id)
                for block_id in free_block_ids
                if int(block_id) not in chosen
            )
            free_block_ids.clear()
            free_block_ids.extend(remaining)
            return run
    return None


def _managed_context_cpu_reload_block_demand(
    spans: Iterable["ManagedContextSpan"],
) -> int:
    """Return the GPU blocks needed to reload cold CPU-offloaded spans."""
    return sum(
        span.kv_block_count for span in spans if span.status == "cpu_offloaded"
    )


def _managed_context_gpu_reload_wait_reason(
    *,
    reload_blocks: int,
    extra_blocks: int,
    free_blocks: int,
) -> str | None:
    """Return a retryable wait reason when reload admission would overfill GPU."""
    required_blocks = max(0, reload_blocks) + max(0, extra_blocks)
    if required_blocks <= max(0, free_blocks):
        return None
    return (
        "managed-context CPU reload is waiting for GPU blocks: "
        f"reload_blocks={reload_blocks} extra_blocks={extra_blocks} "
        f"required={required_blocks} free={free_blocks}"
    )


@dataclass
class Phase4Pin:
    entries: list[tuple[Any, list[Any]]]
    token_count: int
    block_count: int
    request_id: str
    call_idx: int | None
    created_at: float
    consumed_by_request_id: str | None = None
    consumed_at: float | None = None
    status: str = "gpu_pinned"
    cpu_block_ids_by_group: tuple[list[int], ...] = ()
    logical_start_by_group: tuple[list[int], ...] = ()
    store_event_id: int | None = None
    load_event_id: int | None = None
    last_error: str | None = None


@dataclass
class ManagedContextSpan:
    span_id: str
    trace_id: str
    request_id: str
    absolute_turn_start: int
    absolute_turn_end: int
    token_ids: list[int]
    entries: list[tuple[Any, list[Any]]]
    kv_block_count: int
    logical_start_by_group: list[list[int]]
    position_offset_frame: int
    evict_start: int
    evict_end: int
    created_at: float
    status: str = "gpu_pinned"
    cpu_block_ids_by_group: tuple[list[int], ...] = ()
    offload_event_id: int | None = None
    last_error: str | None = None
    pending_load_count: int = 0


@dataclass
class ManagedContextHotGPUStats:
    budget_blocks: int
    resident_blocks: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    promotions: int = 0


@dataclass
class ManagedContextActiveRestore:
    span_ids: list[str]
    entries: list[tuple[Any, list[Any]]]
    block_ids: tuple[list[int], ...]
    num_tokens: int
    created_at: float


@dataclass
class ManagedContextDeferredRestore:
    span_ids: list[str]
    spans: list[ManagedContextSpan]
    restored_entries_by_span: dict[str, list[tuple[Any, list[Any]]]]
    skip_hot_hit_span_ids: set[str]
    created_at: float


@dataclass
class ManagedContextPendingLoad:
    request_id: str
    span_ids: list[str]
    spans: list[ManagedContextSpan]
    restored_entries_by_span: dict[str, list[tuple[Any, list[Any]]]]
    block_ids: tuple[list[int], ...]
    num_tokens: int
    event_id: int
    created_at: float


@dataclass(frozen=True)
class ManagedContextCPULoadStart:
    error: str | None = None
    retryable: bool = False


@dataclass
class RequestKVSwap:
    request_id: str
    cpu_block_ids_by_group: tuple[list[int], ...]
    logical_start_by_group: tuple[list[int], ...]
    kv_block_count: int
    num_computed_tokens: int
    position_offset: int
    status: str
    created_at: float
    entries: list[tuple[Any, list[Any]]]
    store_event_id: int | None = None
    load_event_id: int | None = None
    last_error: str | None = None


_PREEMPTION_FREED = "freed"
_PREEMPTION_ASYNC_PENDING = "async_pending"
_PREEMPTION_DEFERRED = "deferred"
_PREEMPTION_FINISHED = "finished"
_PREEMPTION_FAILED = "failed"


@dataclass(frozen=True)
class PreemptionResult:
    kind: str
    error: str | None = None


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self._validate_compaction_mode_config()
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self._pending_engine_core_outputs: dict[
            int, list[EngineCoreOutput]
        ] = defaultdict(list)
        self.prev_step_scheduled_req_ids: set[str] = set()
        self._kve_sched_sig_step = 0

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        if self.vllm_config.kv_transfer_config is not None:
            # Refuse to even construct an LMCache connector when compaction
            # is enabled: the LMCache V1 adapter uses
            # request._output_token_ids[0] as a "first_tok" fingerprint, which
            # compaction invalidates. Checking here (before connector
            # construction) avoids loading the lmcache package at all and
            # gives a clear error before any side effects.
            if (
                self.cache_config.compaction_window_size > 0
                or self.cache_config.compaction_max_turns > 0
            ):
                kv_connector_name = (
                    self.vllm_config.kv_transfer_config.kv_connector or ""
                ).lower()
                assert "lmcache" not in kv_connector_name, (
                    "KV cache compaction is incompatible with LMCache KV "
                    "transfer: the LMCache V1 adapter uses "
                    "request._output_token_ids[0] as a fingerprint, which "
                    "compaction invalidates. Disable compaction "
                    "(--compaction-window-size 0) or use a different "
                    f"KV connector. Got kv_connector="
                    f"{self.vllm_config.kv_transfer_config.kv_connector!r}."
                )
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
            compaction_window_size=self.cache_config.compaction_window_size,
            compaction_stride=self.cache_config.compaction_stride,
            compaction_max_turns=self.cache_config.compaction_max_turns,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None and hasattr(
            self.connector, "bind_gpu_block_pool"
        ):
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        # Compaction: check if any KV cache manager supports compaction.
        self._compaction_enabled = any(
            isinstance(mgr, CompactingKVCacheManager)
            for mgr in self.kv_cache_manager.coordinator.single_type_managers
        )
        self._compaction_protected_prefix = (
            self.cache_config.compaction_protected_prefix_tokens
        )
        # Block size of the compaction manager (cached once at init). Used
        # to block-align protected_prefix_len in `_ppl()` so the piecewise
        # position rule transitions at a block boundary — RoPE rotations
        # are baked into K at write time at block granularity, so the Q
        # rotation gate must also be block-aligned to avoid partial-block
        # frame mismatches at the sys/post-sys boundary.
        self._compaction_block_size = 0
        if self._compaction_enabled:
            for mgr in self.kv_cache_manager.coordinator.single_type_managers:
                if isinstance(mgr, CompactingKVCacheManager):
                    self._compaction_block_size = mgr.block_size
                    break
        # Auto-pad the trailing partial block at request finish (see
        # `compaction_block_aligned_finish` in CacheConfig). Cached at
        # init for the hot-path check in `update_from_output`.
        self._compaction_block_aligned_finish = (
            self.cache_config.compaction_block_aligned_finish
        )
        self._compaction_filler_token_id = (
            self.cache_config.compaction_filler_token_id
        )
        # Turn-mode compaction state.
        self._compaction_max_turns = self.cache_config.compaction_max_turns
        self._compaction_eviction_turn_stride = (
            self.cache_config.compaction_eviction_turn_stride
        )
        # Lazily set on first use (needs request.sampling_params for fallback
        # auto-detect). Set up-front when explicitly configured.
        self._compaction_turn_end_token_id: int | None = (
            self.cache_config.compaction_turn_end_token_id
        )
        self._compaction_assume_aligned_turn_boundaries = (
            self.cache_config.compaction_assume_aligned_turn_boundaries
        )
        if self._compaction_enabled:
            assert not self.scheduler_config.async_scheduling, (
                "KV cache compaction is incompatible with async scheduling: "
                "num_output_placeholders is nonzero whenever update_from_output "
                "runs, so the compaction trigger is never satisfied and the "
                "run degenerates to full context. Pass async_scheduling=False "
                "to LLM() (or --async-scheduling false) when enabling compaction."
            )
            # LMCache incompatibility is checked above, before connector
            # construction, to avoid loading the lmcache package at all.
            logger.warning(
                "[COMPACT] enabled window=%d stride=%d protected_prefix=%d "
                "max_turns=%d turn_stride=%d turn_end_id=%s aligned_turns=%s",
                self.cache_config.compaction_window_size,
                self.cache_config.compaction_stride,
                self._compaction_protected_prefix,
                self._compaction_max_turns,
                self._compaction_eviction_turn_stride,
                self._compaction_turn_end_token_id,
                self._compaction_assume_aligned_turn_boundaries,
            )

        # Requests with turn-mode admission compaction pending until
        # their prefill-completing step. On that step,
        # `_apply_inline_admission_eviction` (called from `schedule()`
        # BEFORE the SchedulerOutput is built) runs the eviction loop —
        # blocks freed, block_table spliced, tokens trimmed,
        # position_offset bumped — so the worker's prefill kernel sees
        # the post-eviction state on its first pass. Single forward,
        # single sample. See plans/single_forward_pre_eviction.md.
        self._pending_admission_compaction_ids: set[str] = set()
        # Phase4 relies on the next turn prefix-cache-hitting the exact
        # retained rollout state. The normal prefix cache is LRU/evictable, so
        # keep one pinned finished-state block list per rollout trace until the
        # next request for that trace has attached it.
        self._phase4_pinned_blocks: dict[str, Phase4Pin] = {}
        self._phase4_pin_order: deque[str] = deque()
        self._phase4_pin_hit_repeats: defaultdict[
            tuple[str, str, str, int, int, int, int], int
        ] = defaultdict(int)
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        # Managed context is an opt-in diagnostic path layered on turn-mode
        # compaction. It pins evicted spans under Phase4 trace IDs so a later
        # request can validate explicit restore controls without changing the
        # default eviction/free behavior.
        self._managed_context_enabled = (
            os.environ.get("KVE_MANAGED_CONTEXT", "0") == "1"
        )
        self._managed_context_recall_max_spans = self._env_int(
            "KVE_MANAGED_CONTEXT_RECALL_MAX_SPANS", 0
        )
        self._managed_context_recall_max_kv_blocks = self._env_optional_int(
            "KVE_MANAGED_CONTEXT_RECALL_MAX_KV_BLOCKS"
        )
        self._managed_context_archive_max_blocks = self._env_optional_int(
            "KVE_MANAGED_CONTEXT_ARCHIVE_MAX_BLOCKS"
        )
        self._managed_context_archive_ttl_seconds = self._env_float(
            "KVE_MANAGED_CONTEXT_ARCHIVE_TTL_SECONDS", 1800.0
        )
        self._managed_context_align_positions = (
            os.environ.get("KVE_MANAGED_CONTEXT_ALIGN_POSITIONS", "1") != "0"
        )
        self._managed_context_scheduler_accounted_restore = (
            os.environ.get(
                "KVE_MANAGED_CONTEXT_SCHEDULER_ACCOUNTED_RESTORE",
                "0",
            )
            == "1"
        )
        self._managed_context_archive: dict[
            tuple[str, str], ManagedContextSpan
        ] = {}
        self._managed_context_archive_order: deque[tuple[str, str]] = deque()
        self._managed_context_next_span_by_trace: defaultdict[str, int] = (
            defaultdict(int)
        )
        self._managed_context_active_restores: dict[
            str, ManagedContextActiveRestore
        ] = {}
        self._managed_context_deferred_restores: dict[
            str, ManagedContextDeferredRestore
        ] = {}
        self._managed_context_archive_device = os.environ.get(
            "KVE_MANAGED_CONTEXT_ARCHIVE_DEVICE", "gpu"
        ).strip().lower()
        raw_cpu_offload_policy = os.environ.get(
            "KVE_MANAGED_CONTEXT_CPU_OFFLOAD_POLICY", ""
        ).strip().lower()
        explicit_cpu_archive_devices = {
            "cpu_explicit",
            "cpu-explicit",
            "cpu_deferred",
            "cpu-deferred",
        }
        if not raw_cpu_offload_policy:
            raw_cpu_offload_policy = (
                "explicit"
                if self._managed_context_archive_device
                in explicit_cpu_archive_devices
                else "immediate"
            )
        if raw_cpu_offload_policy not in ("immediate", "explicit"):
            logger.warning(
                "[MANAGED-CONTEXT] unknown CPU offload policy %r; using "
                "immediate",
                raw_cpu_offload_policy,
            )
            raw_cpu_offload_policy = "immediate"
        self._managed_context_cpu_offload_policy = raw_cpu_offload_policy
        self._managed_context_cpu_archive_enabled = (
            self._managed_context_archive_device
            in ("cpu", *explicit_cpu_archive_devices)
            or os.environ.get("KVE_MANAGED_CONTEXT_CPU_OFFLOAD", "0") == "1"
        )
        self._managed_context_cpu_offload_immediate = (
            self._managed_context_cpu_archive_enabled
            and self._managed_context_cpu_offload_policy == "immediate"
        )
        self._managed_context_cpu_max_blocks = self._env_int(
            "KVE_MANAGED_CONTEXT_CPU_OFFLOAD_MAX_BLOCKS", 0
        )
        self._managed_context_cpu_evict_on_capacity = (
            os.environ.get(
                "KVE_MANAGED_CONTEXT_CPU_EVICT_ON_CAPACITY",
                "0",
            )
            .strip()
            .lower()
            not in ("0", "false", "no", "off")
        )
        self._managed_context_skip_compaction_on_cpu_capacity = (
            os.environ.get(
                "KVE_MANAGED_CONTEXT_SKIP_COMPACTION_ON_CPU_CAPACITY",
                "1",
            )
            .strip()
            .lower()
            not in ("0", "false", "no", "off")
        )
        self._managed_context_cpu_free_block_ids: deque[int] = deque(
            range(max(0, self._managed_context_cpu_max_blocks))
        )
        self._managed_context_next_transfer_event_id = 0
        self._managed_context_store_events_to_submit: dict[
            int, ManagedContextCopyEvent
        ] = {}
        self._managed_context_load_events_to_submit: dict[
            int, ManagedContextCopyEvent
        ] = {}
        self._managed_context_cpu_max_pending_store_events = self._env_int(
            "KVE_MANAGED_CONTEXT_CPU_MAX_PENDING_STORE_EVENTS", 0
        )
        self._managed_context_cpu_max_pending_load_events = self._env_int(
            "KVE_MANAGED_CONTEXT_CPU_MAX_PENDING_LOAD_EVENTS", 0
        )
        self._managed_context_store_event_to_span: dict[
            int, ManagedContextSpan
        ] = {}
        self._managed_context_load_event_to_request_id: dict[int, str] = {}
        self._managed_context_pending_loads: dict[
            str, ManagedContextPendingLoad
        ] = {}
        self._managed_context_finished_load_req_ids: set[str] = set()
        self._request_kv_swaps: dict[str, RequestKVSwap] = {}
        self._request_kv_swap_ready_queue: deque[str] = deque()
        self._request_kv_swap_store_event_to_request_id: dict[int, str] = {}
        self._request_kv_swap_load_event_to_request_id: dict[int, str] = {}
        self._request_kv_swap_finished_load_req_ids: set[str] = set()
        self._request_kv_swap_suspended = False
        self._phase4_pin_store_event_to_trace_id: dict[int, str] = {}
        self._phase4_pin_load_event_to_trace_id: dict[int, str] = {}
        self._managed_context_restore_reservations: dict[
            str, set[tuple[str, str]]
        ] = {}
        self._managed_context_hot_gpu_order: deque[tuple[str, str]] = deque()
        raw_hot_gpu_budget = self._env_optional_int(
            "KVE_MANAGED_CONTEXT_GPU_HOT_BLOCK_BUDGET"
        )
        if raw_hot_gpu_budget is None:
            # Default-on hot cache for CPU-restored spans. The auto budget holds
            # roughly one full model window worth of restored KV blocks, capped
            # by the CPU archive capacity. Set the env var to 0 to force cold
            # CPU reload behavior.
            auto_hot_gpu_budget = max(1, self.max_model_len // self.block_size)
            raw_hot_gpu_budget = min(
                auto_hot_gpu_budget,
                max(0, self._managed_context_cpu_max_blocks),
            )
        if not self._managed_context_cpu_archive_enabled:
            raw_hot_gpu_budget = 0
        self._managed_context_hot_gpu_stats = ManagedContextHotGPUStats(
            budget_blocks=max(0, raw_hot_gpu_budget)
        )
        if self._managed_context_enabled and (
            not self._compaction_enabled
            or self._compaction_max_turns <= 0
            or not self.cache_config.enable_prefix_caching
            or self.use_v2_model_runner
            or self._managed_context_recall_max_spans <= 0
        ):
            logger.warning(
                "[MANAGED-CONTEXT] disabled: requires turn compaction, "
                "prefix caching, legacy GPU model runner, and "
                "KVE_MANAGED_CONTEXT_RECALL_MAX_SPANS>0"
            )
            self._managed_context_enabled = False
        if self._managed_context_cpu_archive_enabled:
            if not self._managed_context_enabled:
                self._managed_context_cpu_archive_enabled = False
            elif self._managed_context_cpu_max_blocks <= 0:
                logger.warning(
                    "[MANAGED-CONTEXT] CPU archive disabled: set "
                    "KVE_MANAGED_CONTEXT_CPU_OFFLOAD_MAX_BLOCKS>0"
                )
                self._managed_context_cpu_archive_enabled = False
            elif len(self.kv_cache_manager.coordinator.single_type_managers) != 1:
                logger.warning(
                    "[MANAGED-CONTEXT] CPU archive disabled: currently supports "
                    "one KV cache group"
                )
                self._managed_context_cpu_archive_enabled = False
            elif self.connector is not None:
                logger.warning(
                    "[MANAGED-CONTEXT] CPU archive disabled: generic KV "
                    "connector is configured"
                )
                self._managed_context_cpu_archive_enabled = False
            elif (
                self.parallel_config.pipeline_parallel_size != 1
                or self.parallel_config.tensor_parallel_size != 1
            ):
                logger.warning(
                    "[MANAGED-CONTEXT] CPU archive disabled: async managed "
                    "context CPU reload currently supports single-rank "
                    "PP/TP only"
                )
                self._managed_context_cpu_archive_enabled = False
            else:
                logger.warning(
                    "[MANAGED-CONTEXT] CPU archive enabled: async transfers, "
                    "max_cpu_blocks=%d hot_gpu_blocks=%d policy=%s "
                    "evict_on_capacity=%s max_pending_store=%d "
                    "max_pending_load=%d skip_compaction_on_capacity=%s",
                    self._managed_context_cpu_max_blocks,
                    self._managed_context_hot_gpu_stats.budget_blocks,
                    self._managed_context_cpu_offload_policy,
                    self._managed_context_cpu_evict_on_capacity,
                    self._managed_context_cpu_max_pending_store_events,
                    self._managed_context_cpu_max_pending_load_events,
                    self._managed_context_skip_compaction_on_cpu_capacity,
                )
        if not self._managed_context_cpu_archive_enabled:
            self._managed_context_hot_gpu_stats.budget_blocks = 0
            self._managed_context_cpu_offload_immediate = False
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        if self.vllm_config.model_config.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_reader = RoutedExpertsReader.create()

            assert len(kv_cache_config.kv_cache_groups) > 0, (
                "enable_return_routed_experts requires at least one kv cache group"
            )
            # Find the attention group for routed experts indexing.
            self.routed_experts_attn_gid = 0
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                if isinstance(group.kv_cache_spec, AttentionSpec):
                    self.routed_experts_attn_gid = gid
                    break
            min_block_size = min(
                [
                    group.kv_cache_spec.block_size
                    for group in kv_cache_config.kv_cache_groups
                ]
            )
            num_groups = len(kv_cache_config.kv_cache_groups)
            self.max_num_kv_tokens = (
                kv_cache_config.num_blocks // num_groups
            ) * min_block_size
            dcp_size = self.vllm_config.parallel_config.decode_context_parallel_size
            pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
            if pcp_size * dcp_size > 1:
                self.max_num_kv_tokens *= pcp_size * dcp_size

            self.routed_experts_reader.attach_buffer(
                max_num_kv_tokens=self.max_num_kv_tokens,
                vllm_config=self.vllm_config,
            )

        self._pause_state: PauseState = PauseState.UNPAUSED
        self._kve_liveness_diag_last_ts = 0.0

    def _validate_compaction_mode_config(self) -> None:
        window_enabled = self.cache_config.compaction_window_size > 0
        turn_enabled = self.cache_config.compaction_max_turns > 0
        assert not (window_enabled and turn_enabled), (
            "KV cache compaction modes are mutually exclusive: set either "
            "compaction_window_size/compaction_stride for token-window FIFO "
            "eviction, or compaction_max_turns for turn-mode eviction, not "
            "both."
        )
        if not window_enabled:
            assert self.cache_config.compaction_stride == 0, (
                "compaction_stride requires compaction_window_size > 0; "
                "leave compaction_stride at 0 for turn-mode eviction"
            )

    @staticmethod
    def _kve_diag_enabled() -> bool:
        return os.environ.get("KVE_SCHED_LIVENESS_DIAG", "0") == "1"

    @staticmethod
    def _kve_blocks_from_entries(entries: list[tuple[Any, list[Any]]]) -> int:
        return sum(len(blocks) for _, blocks in entries if blocks)

    def _kve_request_visible_block_count(self, request: Request) -> int:
        try:
            block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
        except Exception:  # pragma: no cover - best-effort diagnostics only.
            return -1
        return sum(len(group) for group in block_ids)

    def _kve_managed_context_diag_summary(self) -> dict[str, Any]:
        archive_by_status: defaultdict[str, int] = defaultdict(int)
        archive_blocks_by_status: defaultdict[str, int] = defaultdict(int)
        for span in self._managed_context_archive.values():
            archive_by_status[span.status] += 1
            archive_blocks_by_status[span.status] += span.kv_block_count
        protected_archive_keys = self._managed_context_cpu_archive_protected_keys()
        unprotected_gpu_pinned_spans = 0
        unprotected_gpu_pinned_blocks = 0
        for key, span in self._managed_context_archive.items():
            if span.status == "gpu_pinned" and key not in protected_archive_keys:
                unprotected_gpu_pinned_spans += 1
                unprotected_gpu_pinned_blocks += span.kv_block_count

        active_restore_blocks = sum(
            self._kve_blocks_from_entries(restore.entries)
            for restore in self._managed_context_active_restores.values()
        )
        deferred_restore_blocks = sum(
            sum(
                self._kve_blocks_from_entries(entries)
                for entries in deferred.restored_entries_by_span.values()
            )
            for deferred in self._managed_context_deferred_restores.values()
        )
        pending_load_blocks = sum(
            self._kve_blocks_from_entries(
                [
                    entry
                    for entries in pending.restored_entries_by_span.values()
                    for entry in entries
                ]
            )
            for pending in self._managed_context_pending_loads.values()
        )
        return {
            "archive_spans": dict(sorted(archive_by_status.items())),
            "archive_blocks": dict(sorted(archive_blocks_by_status.items())),
            "active_restores": len(self._managed_context_active_restores),
            "active_restore_blocks": active_restore_blocks,
            "deferred_restores": len(self._managed_context_deferred_restores),
            "deferred_restore_blocks": deferred_restore_blocks,
            "pending_loads": len(self._managed_context_pending_loads),
            "pending_load_blocks": pending_load_blocks,
            "cpu_pending_store_events": len(
                self._managed_context_store_event_to_span
            ),
            "cpu_pending_load_events": len(
                self._managed_context_load_event_to_request_id
            ),
            "cpu_transfer_caps": {
                "store": self._managed_context_cpu_max_pending_store_events,
                "load": self._managed_context_cpu_max_pending_load_events,
            },
            "restore_reservations": len(
                self._managed_context_restore_reservations
            ),
            "phase4_pins": len(self._phase4_pinned_blocks),
            "phase4_pin_blocks": sum(
                pin.block_count for pin in self._phase4_pinned_blocks.values()
            ),
            "cpu_free_blocks": len(self._managed_context_cpu_free_block_ids),
            "cpu_max_blocks": self._managed_context_cpu_max_blocks,
            "unprotected_gpu_pinned_spans": unprotected_gpu_pinned_spans,
            "unprotected_gpu_pinned_blocks": unprotected_gpu_pinned_blocks,
            "hot_gpu": {
                "budget": self._managed_context_hot_gpu_stats.budget_blocks,
                "resident": self._managed_context_hot_gpu_stats.resident_blocks,
                "hits": self._managed_context_hot_gpu_stats.hits,
                "misses": self._managed_context_hot_gpu_stats.misses,
                "evictions": self._managed_context_hot_gpu_stats.evictions,
                "promotions": self._managed_context_hot_gpu_stats.promotions,
            },
        }

    def _kve_gpu_block_pool_diag_summary(self) -> list[dict[str, int]]:
        summaries: list[dict[str, int]] = []
        for group_idx, manager in enumerate(
            self.kv_cache_manager.coordinator.single_type_managers
        ):
            block_pool = manager.block_pool
            summaries.append(
                {
                    "group": group_idx,
                    "total": block_pool.num_gpu_blocks,
                    "free": block_pool.get_num_free_blocks(),
                    "used": block_pool.num_gpu_blocks
                    - block_pool.get_num_free_blocks(),
                    "cached_hashes": len(
                        block_pool.cached_block_hash_to_block._cache
                    ),
                    "cached_this_step": len(block_pool.cached_block_ids_this_step),
                }
            )
        return summaries

    def _kve_request_diag_line(self, request: Request) -> str:
        request_id = request.request_id
        status = getattr(request.status, "name", str(request.status))
        active = self._managed_context_active_restores.get(request_id)
        deferred = self._managed_context_deferred_restores.get(request_id)
        pending = self._managed_context_pending_loads.get(request_id)
        active_blocks = (
            self._kve_blocks_from_entries(active.entries)
            if active is not None
            else 0
        )
        deferred_blocks = (
            sum(
                self._kve_blocks_from_entries(entries)
                for entries in deferred.restored_entries_by_span.values()
            )
            if deferred is not None
            else 0
        )
        pending_blocks = (
            sum(
                self._kve_blocks_from_entries(entries)
                for entries in pending.restored_entries_by_span.values()
            )
            if pending is not None
            else 0
        )
        reserved = self._managed_context_restore_reservations.get(request_id)
        return (
            f"{request_id[:8]} status={status} pos={request.position_offset} "
            f"computed={request.num_computed_tokens}/{request.num_tokens} "
            f"prompt={request.num_prompt_tokens} "
            f"visible_blocks={self._kve_request_visible_block_count(request)} "
            f"pad={int(bool(request.padding_pending))} "
            f"active={active_blocks} deferred={deferred_blocks} "
            f"pending={pending_blocks} reserved={len(reserved or ())}"
        )

    def _kve_request_diag_samples(
        self,
        requests: Iterable[Request],
        *,
        limit: int = 8,
    ) -> list[str]:
        samples: list[str] = []
        for request in requests:
            samples.append(self._kve_request_diag_line(request))
            if len(samples) >= limit:
                break
        return samples

    def _kve_log_compacted_preempt_abort(
        self,
        request: Request,
        *,
        phase: str,
    ) -> None:
        if not self._kve_diag_enabled():
            return
        logger.warning(
            "[SCHED-COMPACT-PREEMPT-DIAG] phase=%s req=%s pools=%s "
            "managed=%s request=%s running=%d waiting=%d skipped=%d "
            "finished_pending=%d",
            phase,
            request.request_id[:8],
            self._kve_gpu_block_pool_diag_summary(),
            self._kve_managed_context_diag_summary(),
            self._kve_request_diag_line(request),
            len(self.running),
            len(self.waiting),
            len(self.skipped_waiting),
            len(self.finished_req_ids),
        )

    def _kve_maybe_log_scheduler_liveness(
        self,
        *,
        total_num_scheduled_tokens: int,
        token_budget: int,
        preempted_reqs: list[Request],
        scheduled_running_reqs: list[Request],
        scheduled_new_reqs: list[Request],
        scheduled_resumed_reqs: list[Request],
    ) -> None:
        if not self._kve_diag_enabled():
            return
        waiting_count = len(self.waiting) + len(self.skipped_waiting)
        stall_like = (
            total_num_scheduled_tokens == 0
            and not self.running
            and waiting_count > 0
        )
        always = os.environ.get("KVE_SCHED_LIVENESS_ALWAYS", "0") == "1"
        if not stall_like and not always:
            return
        interval = self._env_float("KVE_SCHED_LIVENESS_INTERVAL_SECONDS", 5.0)
        now = time.monotonic()
        if now - self._kve_liveness_diag_last_ts < interval:
            return
        self._kve_liveness_diag_last_ts = now
        logger.warning(
            "[SCHED-LIVENESS] stall_like=%s total_sched_tokens=%d "
            "token_budget=%d running=%d waiting=%d skipped=%d "
            "scheduled_running=%d scheduled_new=%d scheduled_resumed=%d "
            "preempted=%d pools=%s managed=%s running_samples=%s "
            "waiting_samples=%s skipped_samples=%s",
            stall_like,
            total_num_scheduled_tokens,
            token_budget,
            len(self.running),
            len(self.waiting),
            len(self.skipped_waiting),
            len(scheduled_running_reqs),
            len(scheduled_new_reqs),
            len(scheduled_resumed_reqs),
            len(preempted_reqs),
            self._kve_gpu_block_pool_diag_summary(),
            self._kve_managed_context_diag_summary(),
            self._kve_request_diag_samples(self.running),
            self._kve_request_diag_samples(self.waiting),
            self._kve_request_diag_samples(self.skipped_waiting),
        )

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet"
        )
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # As an exception, if `num_new_tokens` is less than `block_size`, the
            # state is simply not cached, requiring no special handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass
        return num_new_tokens

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        # KV cache compaction: admission eviction is driven by
        # `_apply_inline_admission_eviction` at the end of this method.
        # When a request's prefill completes this step and it is in
        # `_pending_admission_compaction_ids`, blocks are freed,
        # block_table spliced, prompt tokens trimmed, and
        # position_offset bumped — all before SchedulerOutput is built.
        # The worker then prefills the post-eviction sequence in a
        # single normal forward.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self._prune_phase4_pins()
        self.kv_cache_manager.new_step_starts()
        self._retry_managed_context_gpu_pinned_offloads("schedule-start")
        pressure_preempted_req = self._preempt_request_for_kv_swap_pressure(
            scheduled_timestamp,
            reason="schedule-start-headroom",
            protected_request_ids=set(),
        )
        if pressure_preempted_req is not None:
            preempted_reqs.append(pressure_preempted_req)

        # First, schedule the RUNNING requests.
        if self._compaction_block_aligned_finish:
            _pad_pend = [r.request_id[:8] for r in self.running if r.padding_pending]
            if _pad_pend:
                logger.info(
                    "[COMPACT/auto-pad] schedule(): %d padding-pending in running queue: %s",
                    len(_pad_pend), _pad_pend,
                )
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            self._activate_deferred_managed_context_restore_if_ready(request)
            if request.is_finished():
                req_index += 1
                continue

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            _hidden_kv_tokens = self._managed_context_active_restores.get(
                request.request_id
            )
            hidden_kv_num_tokens = (
                _hidden_kv_tokens.num_tokens
                if _hidden_kv_tokens is not None
                else 0
            )
            # Normal decode keeps one position in reserve for sampling at the
            # model-length boundary. Auto-pad forwards are internal no-sample
            # steps, so they must be allowed to materialize the final valid
            # token at position max_model_len - 1.
            max_sched_len = self.max_model_len
            if not request.padding_pending:
                max_sched_len -= 1
            num_new_tokens = min(
                num_new_tokens,
                max(
                    0,
                    max_sched_len
                    - hidden_kv_num_tokens
                    - request.num_computed_tokens,
                ),
            )
            num_new_tokens = (
                self._cap_managed_context_deferred_prefill_tokens(
                    request,
                    num_computed_tokens=request.num_computed_tokens,
                    num_new_tokens=num_new_tokens,
                )
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        self._restamp_reprefill_logical_starts(
                            request, reason="running-alloc"
                        )
                        break

                    # The request cannot be scheduled.
                    pressure_preempted_req = (
                        self._preempt_request_for_kv_swap_pressure(
                            scheduled_timestamp,
                            reason="running-alloc-pressure",
                            protected_request_ids={
                                request.request_id,
                                *(
                                    scheduled_req.request_id
                                    for scheduled_req in scheduled_running_reqs
                                ),
                            },
                        )
                    )
                    if pressure_preempted_req is not None:
                        preempted_reqs.append(pressure_preempted_req)
                        break

                    if self._release_phase4_pressure_pin(
                        "running-alloc-pressure"
                    ):
                        continue

                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        preempted_req_index = self.running.index(preempted_req)
                        self.running.pop(preempted_req_index)
                    else:
                        preempted_req_index = len(self.running) - 1
                        preempted_req = self.running.pop()

                    was_scheduled_this_step = (
                        preempted_req in scheduled_running_reqs
                    )
                    preemption_result = self._preempt_request(
                        preempted_req,
                        scheduled_timestamp,
                        allow_async_kv_swap=not was_scheduled_this_step,
                    )
                    if preemption_result.kind == _PREEMPTION_DEFERRED:
                        self.running.insert(preempted_req_index, preempted_req)
                        if (
                            os.environ.get("KVE_TRACE_REQUEST_KV_SWAP") == "1"
                            and preemption_result.error is not None
                        ):
                            logger.warning(
                                "[REQUEST-KV-SWAP-PREEMPT-DEFER] req=%s "
                                "error=%s",
                                preempted_req.request_id[:8],
                                preemption_result.error,
                            )
                        break
                    if was_scheduled_this_step:
                        preempted_req_id = preempted_req.request_id
                        scheduled_running_reqs.remove(preempted_req)
                        token_budget += num_scheduled_tokens.pop(preempted_req_id)
                        req_to_new_blocks.pop(preempted_req_id)
                        scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                        preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                            preempted_req_id, None
                        )
                        if preempted_encoder_inputs:
                            # Restore encoder compute budget if the preempted
                            # request had encoder inputs scheduled in this step.
                            num_embeds_to_restore = sum(
                                preempted_req.get_num_encoder_embeds(i)
                                for i in preempted_encoder_inputs
                            )
                            encoder_compute_budget += num_embeds_to_restore
                        req_index -= 1
                    if not preempted_req.is_finished():
                        preempted_reqs.append(preempted_req)
                    if preemption_result.kind == _PREEMPTION_ASYNC_PENDING:
                        # Request KV swap-out frees GPU blocks only after the
                        # D2H store completion arrives. Stop this allocation
                        # retry loop until a later scheduler tick observes that
                        # completion and actual block release.
                        break
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        pad_pending_barrier = (
            os.environ.get("KVE_PAD_PENDING_BARRIER") == "1"
            and any(req.padding_pending for req in scheduled_running_reqs)
        )
        if pad_pending_barrier:
            logger.info(
                "[SCHED-DIAG] padding-pending barrier: deferring waiting "
                "requests until padding forwards complete; scheduled=%s",
                [req.request_id[:8] for req in scheduled_running_reqs],
            )

        # Next, schedule the WAITING requests.
        if (
            not preempted_reqs
            and self._pause_state == PauseState.UNPAUSED
            and not pad_pending_barrier
        ):
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                if not self._request_kv_swap_resident_first_enabled():
                    self._request_kv_swap_prioritize_ready_waiting()
                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                restore_spans, restore_error = (
                    self._validate_managed_context_restore_request(request)
                )
                if restore_error is not None:
                    if self._drop_unavailable_managed_context_restore(
                        request,
                        restore_error,
                    ):
                        restore_spans = []
                    else:
                        request_queue.pop_request()
                        self._abort_waiting_phase4_request(request, restore_error)
                        continue
                if restore_spans:
                    self._reserve_managed_context_restore_spans(
                        request_id,
                        restore_spans,
                    )
                offload_spans, offload_error = (
                    self._validate_managed_context_offload_request(request)
                )
                if offload_error is not None:
                    request_queue.pop_request()
                    self._abort_waiting_phase4_request(request, offload_error)
                    continue
                if offload_spans:
                    offload_error = self._start_managed_context_cpu_offloads(
                        offload_spans,
                        "request",
                    )
                    if offload_error is not None:
                        logger.warning(
                            "[MANAGED-CONTEXT-OFFLOAD-SKIP] req=%s %s",
                            request_id[:8],
                            offload_error,
                        )

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ):
                    if self._request_kv_swap_should_park_waiting_request(
                        request,
                        token_budget=token_budget,
                    ):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue
                    promoted_blocked_request = (
                        self._try_promote_blocked_waiting_request(
                            request,
                            token_budget=token_budget,
                        )
                    )
                else:
                    promoted_blocked_request = True

                if not promoted_blocked_request:
                    block_waiting_admission = (
                        self._request_kv_swap_should_block_waiting_admission(
                            request
                        )
                    )
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    if block_waiting_admission:
                        break
                    continue

                if (
                    restore_spans
                    and self._managed_context_cpu_archive_enabled
                    and any(span.status == "offload_pending" for span in restore_spans)
                ):
                    if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
                        logger.warning(
                            "[MANAGED-CONTEXT-RESTORE-WAIT-STORE] req=%s spans=%s",
                            request_id[:8],
                            [
                                span.span_id
                                for span in restore_spans
                                if span.status == "offload_pending"
                            ],
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
                pending_inherit_event: CompactionEvent | None = None
                phase4_pin_trace_to_mark_consumed = ""
                phase4_pin_call_to_mark_consumed = ""
                position_offset_before_restore_align = request.position_offset

                def clear_pending_phase4_pin_consumed() -> None:
                    if phase4_pin_trace_to_mark_consumed:
                        self._clear_phase4_pin_consumed(
                            phase4_pin_trace_to_mark_consumed,
                            request.request_id,
                        )

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    trace_prefix_dep = os.environ.get("KVE_TRACE_PREFIX_DEP") == "1"
                    drop_unready_prefix_dep = (
                        os.environ.get("KVE_DROP_UNREADY_PREFIX_DEPS") == "1"
                    )
                    drop_live_prefix_dep = (
                        os.environ.get("KVE_DROP_LIVE_PREFIX_DEPS") == "1"
                    )
                    truncate_unready_prefix_dep = (
                        os.environ.get("KVE_TRUNCATE_UNREADY_PREFIX_DEPS") == "1"
                    )
                    truncate_live_prefix_dep = (
                        os.environ.get("KVE_TRUNCATE_LIVE_PREFIX_DEPS") == "1"
                    )
                    live_block_owner = {}
                    if (
                        trace_prefix_dep
                        or drop_unready_prefix_dep
                        or drop_live_prefix_dep
                        or truncate_unready_prefix_dep
                        or truncate_live_prefix_dep
                    ):
                        for live_request in self.running:
                            live_blocks = self.kv_cache_manager.get_blocks(
                                live_request.request_id
                            )
                            live_status = getattr(
                                live_request.status, "name", str(live_request.status)
                            )
                            for live_group_idx, live_group in enumerate(
                                live_blocks.blocks
                            ):
                                for live_block_idx, live_block in enumerate(live_group):
                                    live_block_owner.setdefault(
                                        live_block.block_id,
                                        (
                                            live_request.request_id[:8],
                                            live_group_idx,
                                            live_block_idx,
                                            bool(live_request.padding_pending),
                                            live_request.num_computed_tokens,
                                            live_request.num_tokens,
                                            live_status,
                                        ),
                                    )

                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens, inherited_offset = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )
                    if (
                        (trace_prefix_dep
                         or drop_unready_prefix_dep
                         or drop_live_prefix_dep
                         or truncate_unready_prefix_dep
                         or truncate_live_prefix_dep)
                        and num_new_local_computed_tokens > 0
                    ):
                        dep_count = 0
                        unready_count = 0
                        first_live_dep_block_idx = None
                        first_unready_dep_block_idx = None
                        dep_samples = []
                        unready_samples = []
                        for group_idx, group in enumerate(new_computed_blocks.blocks):
                            for block_idx, block in enumerate(group):
                                owner = live_block_owner.get(block.block_id)
                                if owner is None:
                                    continue
                                (
                                    owner_req_id,
                                    owner_group_idx,
                                    owner_block_idx,
                                    owner_padding_pending,
                                    owner_num_computed,
                                    owner_num_tokens,
                                    owner_status,
                                ) = owner
                                if owner_req_id == request.request_id[:8]:
                                    continue
                                dep_count += 1
                                if first_live_dep_block_idx is None:
                                    first_live_dep_block_idx = block_idx
                                owner_block_end = (
                                    owner_block_idx + 1
                                ) * self._compaction_block_size
                                owner_unready = (
                                    self._compaction_block_size > 0
                                    and owner_num_computed < owner_block_end
                                )
                                if owner_unready:
                                    unready_count += 1
                                    if first_unready_dep_block_idx is None:
                                        first_unready_dep_block_idx = block_idx
                                if len(dep_samples) < 8:
                                    dep_samples.append(
                                        "g%d:b%d:id=%d->req=%s/g%d:b%d/"
                                        "pending=%d/computed=%d/%d/status=%s"
                                        % (
                                            group_idx,
                                            block_idx,
                                            block.block_id,
                                            owner_req_id,
                                            owner_group_idx,
                                            owner_block_idx,
                                            int(owner_padding_pending),
                                            owner_num_computed,
                                            owner_num_tokens,
                                            owner_status,
                                        )
                                    )
                                if owner_unready and len(unready_samples) < 8:
                                    unready_samples.append(
                                        "g%d:b%d:id=%d->req=%s/g%d:b%d/"
                                        "computed=%d/%d"
                                        % (
                                            group_idx,
                                            block_idx,
                                            block.block_id,
                                            owner_req_id,
                                            owner_group_idx,
                                            owner_block_idx,
                                            owner_num_computed,
                                            owner_num_tokens,
                                        )
                                    )
                        logger.warning(
                            "[PREFIX-DEP] req=%s cached=%d prompt=%d "
                            "deps=%d unready=%d samples=%s unready_samples=%s",
                            request.request_id[:8],
                            num_new_local_computed_tokens,
                            request.num_tokens,
                            dep_count,
                            unready_count,
                            dep_samples,
                            unready_samples,
                        )
                        truncate_at = None
                        truncate_reason = ""
                        if (
                            truncate_unready_prefix_dep
                            and first_unready_dep_block_idx is not None
                        ):
                            truncate_at = first_unready_dep_block_idx
                            truncate_reason = "unready"
                        elif (
                            truncate_live_prefix_dep
                            and first_live_dep_block_idx is not None
                        ):
                            truncate_at = first_live_dep_block_idx
                            truncate_reason = "live"
                        if truncate_at is not None:
                            old_cached_tokens = num_new_local_computed_tokens
                            safe_blocks = max(0, truncate_at)
                            truncated = tuple(
                                list(group[:safe_blocks])
                                for group in new_computed_blocks.blocks
                            )
                            new_computed_blocks = (
                                self.kv_cache_manager.create_kv_cache_blocks(
                                    truncated
                                )
                            )
                            num_new_local_computed_tokens = (
                                safe_blocks * self._compaction_block_size
                            )
                            inherited_offset = 0
                            if num_new_local_computed_tokens > 0:
                                for block_idx, block in enumerate(
                                    new_computed_blocks.blocks[0]
                                ):
                                    if block.logical_start < 0:
                                        continue
                                    block_offset = (
                                        block.logical_start
                                        - block_idx
                                        * self._compaction_block_size
                                    )
                                    if block_offset != 0:
                                        inherited_offset = block_offset
                                        break
                            request.position_offset = inherited_offset
                            logger.warning(
                                "[PREFIX-DEP-TRUNC] req=%s reason=%s "
                                "cached=%d->%d blocks=%d->%d deps=%d "
                                "unready=%d inherited_offset=%d",
                                request.request_id[:8],
                                truncate_reason,
                                old_cached_tokens,
                                num_new_local_computed_tokens,
                                old_cached_tokens
                                // max(1, self._compaction_block_size),
                                safe_blocks,
                                dep_count,
                                unready_count,
                                inherited_offset,
                            )
                        should_drop_live_dep = (
                            (drop_live_prefix_dep and dep_count > 0)
                            or (drop_unready_prefix_dep and unready_count > 0)
                        )
                        if should_drop_live_dep:
                            logger.warning(
                                "[PREFIX-DEP-DROP] req=%s dropping local "
                                "prefix hit cached=%d prompt=%d deps=%d "
                                "unready=%d inherited_offset=%d",
                                request.request_id[:8],
                                num_new_local_computed_tokens,
                                request.num_tokens,
                                dep_count,
                                unready_count,
                                inherited_offset,
                            )
                            new_computed_blocks = (
                                self.kv_cache_manager.empty_kv_cache_blocks
                            )
                            num_new_local_computed_tokens = 0
                            inherited_offset = 0
                            request.position_offset = 0
                    # Build the synthetic inherit event now, but only append it
                    # after allocate_slots succeeds. Some prefix-readiness
                    # diagnostics intentionally defer a WAITING request after
                    # get_computed_blocks seeded position_offset; appending here
                    # would leave stale trainer metadata on the retry.
                    expected_cached_tokens = self._phase4_expected_cached_tokens(
                        request
                    )
                    phase4_refill_after_pin_release = (
                        self._phase4_pin_release_reprefill_active(request)
                    )
                    if phase4_refill_after_pin_release:
                        inherited_offset = request.position_offset
                    if (
                        (
                            num_new_local_computed_tokens > 0
                            or expected_cached_tokens is not None
                        )
                        and self._compaction_enabled
                        and self._compaction_max_turns > 0
                        and not phase4_refill_after_pin_release
                    ):
                        sys_boundary_raw = self._effective_prompt_tokens(request)
                        extra_args = (
                            request.sampling_params.extra_args
                            if request.sampling_params is not None
                            else {}
                        ) or {}
                        phase4_trace_id = str(
                            extra_args.get("kve_phase4_trace_id", "")
                        )
                        phase4_call_idx = extra_args.get(
                            "kve_phase4_call_idx", ""
                        )
                        if expected_cached_tokens is None:
                            if (
                                os.environ.get(
                                    "KVE_TRACE_PHASE4_PREFIX_HIT", ""
                                ) == "1"
                            ):
                                logger.warning(
                                    "[PHASE4-PREFIX/SKIP] req=%s trace=%s "
                                    "call=%s prompt=%d cached=%d "
                                    "reason=no-explicit-phase4-boundary",
                                    request.request_id[:8],
                                    phase4_trace_id,
                                    phase4_call_idx,
                                    request.num_prompt_tokens,
                                    num_new_local_computed_tokens,
                                )
                        else:
                            pinned = None
                            if not phase4_refill_after_pin_release:
                                pinned = self._phase4_pinned_cache_blocks(
                                    phase4_trace_id, expected_cached_tokens
                                )
                            phase4_pin_trace_to_mark_consumed = ""
                            if pinned is not None:
                                (
                                    pinned_blocks,
                                    pinned_cached_tokens,
                                    pinned_inherited_offset,
                                ) = pinned
                                old_cached_tokens = num_new_local_computed_tokens
                                new_computed_blocks = pinned_blocks
                                num_new_local_computed_tokens = (
                                    pinned_cached_tokens
                                )
                                inherited_offset = pinned_inherited_offset
                                request.position_offset = inherited_offset
                                phase4_pin_trace_to_mark_consumed = (
                                    phase4_trace_id
                                )
                                phase4_pin_call_to_mark_consumed = str(
                                    phase4_call_idx
                                )
                                # Managed CPU restore can defer before
                                # allocate_slots(). At the prefix hit point the
                                # request has already started depending on this
                                # retained KV, so mark it consumed here.
                                self._mark_phase4_pin_consumed(
                                    phase4_pin_trace_to_mark_consumed,
                                    request.request_id,
                                )
                                if (
                                    os.environ.get(
                                        "KVE_TRACE_PHASE4_PREFIX_HIT", ""
                                    ) == "1"
                                    or old_cached_tokens
                                    < expected_cached_tokens
                                ):
                                    repeat = 1
                                    should_log_pin_hit = True
                                    if (
                                        os.environ.get(
                                            "KVE_PHASE4_PIN_HIT_RATE_LIMIT", ""
                                        )
                                        == "1"
                                    ):
                                        repeat_key = (
                                            request.request_id,
                                            phase4_trace_id,
                                            str(phase4_call_idx),
                                            old_cached_tokens,
                                            num_new_local_computed_tokens,
                                            expected_cached_tokens,
                                            inherited_offset,
                                        )
                                        self._phase4_pin_hit_repeats[
                                            repeat_key
                                        ] += 1
                                        repeat = self._phase4_pin_hit_repeats[
                                            repeat_key
                                        ]
                                        should_log_pin_hit = (
                                            repeat <= 4
                                            or repeat & (repeat - 1) == 0
                                            or repeat % 1000 == 0
                                        )
                                    if should_log_pin_hit:
                                        logger.warning(
                                            "[PHASE4-PIN-HIT] req=%s trace=%s "
                                            "call=%s cached=%d->%d expected=%d "
                                            "position_offset=%d repeat=%d "
                                            "running=%d waiting=%d skipped=%d "
                                            "token_budget=%d pins=%d",
                                            request.request_id[:8],
                                            phase4_trace_id,
                                            phase4_call_idx,
                                            old_cached_tokens,
                                            num_new_local_computed_tokens,
                                            expected_cached_tokens,
                                            inherited_offset,
                                            repeat,
                                            len(self.running),
                                            len(request_queue),
                                            len(self.skipped_waiting),
                                            token_budget,
                                            len(self._phase4_pinned_blocks),
                                        )
                            if (
                                num_new_local_computed_tokens
                                > expected_cached_tokens
                            ):
                                old_cached_tokens = num_new_local_computed_tokens
                                bs = self._compaction_block_size
                                if bs > 0:
                                    safe_blocks = expected_cached_tokens // bs
                                    capped_cached_tokens = safe_blocks * bs
                                else:
                                    safe_blocks = expected_cached_tokens
                                    capped_cached_tokens = expected_cached_tokens
                                truncated = tuple(
                                    list(group[:safe_blocks])
                                    for group in new_computed_blocks.blocks
                                )
                                new_computed_blocks = (
                                    self.kv_cache_manager.create_kv_cache_blocks(
                                        truncated
                                    )
                                )
                                num_new_local_computed_tokens = (
                                    capped_cached_tokens
                                )
                                inherited_offset = 0
                                if num_new_local_computed_tokens > 0:
                                    for block_idx, block in enumerate(
                                        new_computed_blocks.blocks[0]
                                    ):
                                        if block.logical_start < 0:
                                            continue
                                        block_offset = (
                                            block.logical_start
                                            - block_idx * max(1, bs)
                                        )
                                        if block_offset != 0:
                                            inherited_offset = block_offset
                                            break
                                request.position_offset = inherited_offset
                                if (
                                    os.environ.get(
                                        "KVE_TRACE_PHASE4_PREFIX_HIT", ""
                                    ) == "1"
                                ):
                                    logger.warning(
                                        "[PHASE4-PREFIX-CAP] req=%s trace=%s "
                                        "call=%s cached=%d->%d expected=%d "
                                        "position_offset=%d",
                                        request.request_id[:8],
                                        phase4_trace_id,
                                        phase4_call_idx,
                                        old_cached_tokens,
                                        num_new_local_computed_tokens,
                                        expected_cached_tokens,
                                        inherited_offset,
                                    )
                            delta = (
                                expected_cached_tokens
                                - num_new_local_computed_tokens
                            )
                            if (
                                os.environ.get(
                                    "KVE_TRACE_PHASE4_PREFIX_HIT", ""
                                ) == "1"
                                or delta > 0
                            ):
                                logger.warning(
                                    "[PHASE4-PREFIX] req=%s trace=%s "
                                    "call=%s prompt=%d cached=%d "
                                    "expected_cached=%d delta=%d "
                                    "actual_nuf=%d expected_nuf=%d "
                                    "position_offset=%d",
                                    request.request_id[:8],
                                    phase4_trace_id,
                                    phase4_call_idx,
                                    request.num_prompt_tokens,
                                    num_new_local_computed_tokens,
                                    expected_cached_tokens,
                                    delta,
                                    request.num_prompt_tokens
                                    - num_new_local_computed_tokens,
                                    request.num_prompt_tokens
                                    - expected_cached_tokens,
                                    inherited_offset,
                                )
                            if delta > 0:
                                phase4_prefix_miss_msg = (
                                    "vLLM would re-prefill retained Phase4 "
                                    "tokens; aborting request instead: "
                                    f"req={request.request_id[:8]} "
                                    f"trace={phase4_trace_id} "
                                    f"call={phase4_call_idx} "
                                    f"prompt={request.num_prompt_tokens} "
                                    f"cached={num_new_local_computed_tokens} "
                                    f"expected_cached={expected_cached_tokens} "
                                    f"delta={delta} "
                                    f"actual_nuf={request.num_prompt_tokens - num_new_local_computed_tokens} "
                                    f"expected_nuf={request.num_prompt_tokens - expected_cached_tokens} "
                                    f"position_offset={inherited_offset}"
                                )
                                if phase4_refill_after_pin_release:
                                    logger.warning(
                                        "[PHASE4-PREFIX-REFILL] %s",
                                        phase4_prefix_miss_msg,
                                    )
                                elif (
                                    os.environ.get(
                                        "KVE_ALLOW_PHASE4_REFILL", ""
                                    ) == "1"
                                ):
                                    logger.error(
                                        "[PHASE4-PREFIX-REFILL-ALLOWED] %s",
                                        phase4_prefix_miss_msg,
                                    )
                                else:
                                    pin_load_reason = (
                                        self._phase4_try_load_pin_for_prefix_miss(
                                            phase4_trace_id,
                                            expected_cached_tokens,
                                            request,
                                            phase4_prefix_miss_msg,
                                        )
                                    )
                                    if self._phase4_pin_recovery_defers(
                                        pin_load_reason
                                    ):
                                        logger.warning(
                                            "[PHASE4-PREFIX-DEFER] req=%s "
                                            "trace=%s call=%s reason=%s "
                                            "miss=%s",
                                            request.request_id[:8],
                                            phase4_trace_id,
                                            phase4_call_idx,
                                            pin_load_reason,
                                            phase4_prefix_miss_msg,
                                        )
                                        request_queue.pop_request()
                                        clear_pending_phase4_pin_consumed()
                                        step_skipped_waiting.prepend_request(
                                            request
                                        )
                                        continue
                                    if self._phase4_prefix_miss_reprefill_enabled():
                                        if self._phase4_requeue_prefix_miss_for_reprefill(
                                            request,
                                            phase4_prefix_miss_msg,
                                        ):
                                            request_queue.pop_request()
                                            clear_pending_phase4_pin_consumed()
                                            step_skipped_waiting.prepend_request(
                                                request
                                            )
                                            continue
                                    logger.error(
                                        "[PHASE4-PREFIX-ABORT] %s "
                                        "pin_recovery=%s",
                                        phase4_prefix_miss_msg,
                                        pin_load_reason,
                                    )
                                    request_queue.pop_request()
                                    clear_pending_phase4_pin_consumed()
                                    self._abort_waiting_phase4_request(
                                        request, phase4_prefix_miss_msg
                                    )
                                    continue
                            # Block-align to match inference's piecewise rule
                            # (which now uses block-aligned protected_prefix_len
                            # via _ppl). The trainer's per_call_segmented_forward
                            # uses evict_start as the piecewise split, so both
                            # sides must use the same block-aligned boundary
                            # or trainer-K and inference-K diverge on the
                            # boundary block's post-sys slots.
                            bs = self._compaction_block_size
                            if bs > 0:
                                sys_boundary = (
                                    (sys_boundary_raw + bs - 1) // bs
                                ) * bs
                                sys_boundary = min(
                                    sys_boundary, request.num_prompt_tokens
                                )
                            else:
                                sys_boundary = sys_boundary_raw
                            # Use the explicit Phase4 boundary, not the
                            # opportunistic global prefix-cache hit. Extra
                            # cached tokens can be valid for inference, but
                            # the trainer mirror only owns the rollout-local
                            # carried state at expected_cached_tokens.
                            nuf_len = max(
                                1,
                                request.num_prompt_tokens
                                - expected_cached_tokens,
                            )
                            pending_inherit_event = CompactionEvent(
                                num_output_tokens_at_compaction=0,
                                tokens_evicted=0,
                                position_offset_after=inherited_offset,
                                num_prompt_tokens=request.num_prompt_tokens,
                                evict_start=sys_boundary,
                                evicted_token_ids=[],
                                last_turn_evicted=-1,
                                num_turns_evicted_after=request.num_turns_evicted,
                                kept_indices=[],
                                kept_token_ids=[],
                                new_user_fragment_len=nuf_len,
                            )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            clear_pending_phase4_pin_consumed()
                            if pending_inherit_event is not None:
                                request.position_offset = 0
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                restore_defer_until_prefill = (
                    bool(restore_spans)
                    and self._managed_context_defer_restore_until_prefill(request)
                    and num_computed_tokens < request.num_prompt_tokens
                )
                if restore_spans and not restore_defer_until_prefill:
                    min_visible_prefix = self._worker_protected_prefix_len(
                        request
                    )
                    if num_computed_tokens < min_visible_prefix:
                        reason = (
                            "managed-context restore requires the visible "
                            "system prefix to be cached before hidden KV is "
                            "attached: "
                            f"cached={num_computed_tokens} "
                            f"required={min_visible_prefix}"
                        )
                        logger.error(
                            "[MANAGED-CONTEXT-PREFIX-ABORT] req=%s %s",
                            request.request_id[:8],
                            reason,
                        )
                        request_queue.pop_request()
                        clear_pending_phase4_pin_consumed()
                        self._abort_waiting_phase4_request(request, reason)
                        continue

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        clear_pending_phase4_pin_consumed()
                        if pending_inherit_event is not None:
                            request.position_offset = 0
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    num_new_tokens = (
                        self._cap_managed_context_deferred_prefill_tokens(
                            request,
                            num_computed_tokens=num_computed_tokens,
                            num_new_tokens=num_new_tokens,
                        )
                    )
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            clear_pending_phase4_pin_consumed()
                            if pending_inherit_event is not None:
                                request.position_offset = 0
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        clear_pending_phase4_pin_consumed()
                        if pending_inherit_event is not None:
                            request.position_offset = 0
                        break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                if (
                    self.scheduler_reserve_full_isl
                    and not self.kv_cache_manager.can_fit_full_sequence(
                        request,
                        num_new_computed_tokens=num_new_local_computed_tokens,
                        new_computed_blocks=new_computed_blocks,
                        num_external_computed_tokens=num_external_computed_tokens,
                        num_encoder_tokens=num_encoder_tokens,
                    )
                ):
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    clear_pending_phase4_pin_consumed()
                    if pending_inherit_event is not None:
                        request.position_offset = 0
                    break

                if (
                    restore_spans
                    and request.request_id
                    not in self._managed_context_active_restores
                    and request.request_id
                    not in self._managed_context_deferred_restores
                    and self._managed_context_restore_needs_cpu_load(restore_spans)
                ):
                    extra_required_gpu_blocks = 0
                    if self._managed_context_scheduler_accounted_restore:
                        extra_required_gpu_blocks = (
                            self._managed_context_visible_allocation_demand(
                                request,
                                num_new_tokens=num_new_tokens,
                                num_new_computed_tokens=num_new_local_computed_tokens,
                                new_computed_blocks=new_computed_blocks,
                                num_lookahead_tokens=effective_lookahead_tokens,
                                num_external_computed_tokens=(
                                    num_external_computed_tokens
                                ),
                                num_encoder_tokens=num_encoder_tokens,
                            )
                        )
                    load_start = self._start_managed_context_cpu_load(
                        request,
                        restore_spans,
                        extra_required_gpu_blocks=extra_required_gpu_blocks,
                    )
                    if load_start.error is not None:
                        if load_start.retryable:
                            if (
                                os.environ.get("KVE_TRACE_MANAGED_CONTEXT")
                                == "1"
                            ):
                                logger.warning(
                                    "[MANAGED-CONTEXT-LOAD-DEFER] req=%s %s",
                                    request.request_id[:8],
                                    load_start.error,
                                )
                            self._reserve_managed_context_restore_spans(
                                request.request_id,
                                restore_spans,
                            )
                            reprefill_after_flush = bool(
                                getattr(
                                    request,
                                    "_kve_reprefill_after_flush",
                                    False,
                                )
                            )
                            if (
                                request.num_computed_tokens == 0
                                and not reprefill_after_flush
                            ):
                                if (
                                    request.position_offset != 0
                                    and os.environ.get(
                                        "KVE_TRACE_MANAGED_CONTEXT"
                                    )
                                    == "1"
                                ):
                                    logger.warning(
                                        "[MANAGED-CONTEXT-LOAD-RESET-POS] "
                                        "req=%s reason=load-defer "
                                        "position_offset=%d->0 "
                                        "cached=%d inherited=%d",
                                        request.request_id[:8],
                                        request.position_offset,
                                        num_new_local_computed_tokens,
                                        inherited_offset,
                                    )
                                request.position_offset = 0
                            else:
                                request.position_offset = (
                                    position_offset_before_restore_align
                                )
                            clear_pending_phase4_pin_consumed()
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        logger.error(
                            "[MANAGED-CONTEXT-LOAD-ABORT] req=%s %s",
                            request.request_id[:8],
                            load_start.error,
                        )
                        request_queue.pop_request()
                        clear_pending_phase4_pin_consumed()
                        self._abort_waiting_phase4_request(
                            request, load_start.error
                        )
                        continue
                    self._release_managed_context_restore_reservation(
                        request.request_id,
                        "load-started",
                    )
                    request = request_queue.pop_request()
                    clear_pending_phase4_pin_consumed()
                    reprefill_after_flush = bool(
                        getattr(
                            request,
                            "_kve_reprefill_after_flush",
                            False,
                        )
                    )
                    if (
                        request.num_computed_tokens == 0
                        and not reprefill_after_flush
                    ):
                        if (
                            request.position_offset != 0
                            and os.environ.get("KVE_TRACE_MANAGED_CONTEXT")
                            == "1"
                        ):
                            logger.warning(
                                "[MANAGED-CONTEXT-LOAD-RESET-POS] req=%s "
                                "reason=load-started position_offset=%d->0 "
                                "cached=%d inherited=%d",
                                request.request_id[:8],
                                request.position_offset,
                                num_new_local_computed_tokens,
                                inherited_offset,
                            )
                        request.position_offset = 0
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    continue

                visible_allocation_demand = (
                    self._managed_context_visible_allocation_demand(
                        request,
                        num_new_tokens=num_new_tokens,
                        num_new_computed_tokens=num_new_local_computed_tokens,
                        new_computed_blocks=new_computed_blocks,
                        num_lookahead_tokens=effective_lookahead_tokens,
                        num_external_computed_tokens=num_external_computed_tokens,
                        num_encoder_tokens=num_encoder_tokens,
                    )
                    if self._managed_context_enabled
                    else -1
                )
                restore_position_aligned = False
                if restore_spans:
                    if not restore_defer_until_prefill:
                        restore_position_aligned = (
                            self._align_managed_context_restore_position(
                                request, restore_spans
                            )
                        )
                        if (
                            restore_position_aligned
                            and pending_inherit_event is not None
                        ):
                            pending_inherit_event.position_offset_after = (
                                request.position_offset
                            )
                    elif os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
                        logger.warning(
                            "[MANAGED-CONTEXT-ALIGN-SKIP] req=%s "
                            "reason=defer-until-prefill "
                            "position_offset=%d spans=%s",
                            request.request_id[:8],
                            request.position_offset,
                            [span.span_id for span in restore_spans],
                        )
                    if (
                        restore_defer_until_prefill
                        and request.request_id
                        not in self._managed_context_deferred_restores
                    ):
                        self._set_managed_context_deferred_restore(
                            request,
                            restore_spans,
                        )
                    if restore_defer_until_prefill:
                        self._activate_deferred_managed_context_restore_if_ready(
                            request,
                            computed_tokens=num_computed_tokens,
                        )

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    logger.warning(
                        "[SCHED-WAITING-ALLOC-NONE] req=%s prompt=%d "
                        "tokens=%d computed=%d local=%d external=%d "
                        "new_tokens=%d token_budget=%d running=%d "
                        "waiting=%d skipped=%d free_gpu_blocks=%d "
                        "estimated_visible_blocks=%d visible_blocks=%d "
                        "position_offset=%d managed=%s",
                        request.request_id[:8],
                        request.num_prompt_tokens,
                        request.num_tokens,
                        num_computed_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                        num_new_tokens,
                        token_budget,
                        len(self.running),
                        len(request_queue),
                        len(self.skipped_waiting),
                        self._managed_context_min_free_gpu_blocks(),
                        visible_allocation_demand,
                        self._kve_request_visible_block_count(request),
                        request.position_offset,
                        self._kve_managed_context_diag_summary(),
                    )
                    if (
                        phase4_pin_trace_to_mark_consumed
                        and os.environ.get("KVE_TRACE_PHASE4_SCHED", "") == "1"
                    ):
                        logger.warning(
                            "[PHASE4-PIN-ALLOC-NONE] req=%s trace=%s call=%s "
                            "prompt=%d num_tokens=%d computed=%d local=%d "
                            "external=%d new_tokens=%d token_budget=%d "
                            "running=%d waiting=%d skipped=%d pins=%d "
                            "position_offset=%d",
                            request.request_id[:8],
                            phase4_pin_trace_to_mark_consumed,
                            phase4_pin_call_to_mark_consumed,
                            request.num_prompt_tokens,
                            request.num_tokens,
                            num_computed_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                            num_new_tokens,
                            token_budget,
                            len(self.running),
                            len(request_queue),
                            len(self.skipped_waiting),
                            len(self._phase4_pinned_blocks),
                            request.position_offset,
                        )
                    if phase4_pin_trace_to_mark_consumed:
                        self._clear_phase4_pin_consumed(
                            phase4_pin_trace_to_mark_consumed,
                            request.request_id,
                        )

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    if pending_inherit_event is not None:
                        request.position_offset = 0
                    elif restore_position_aligned:
                        request.position_offset = (
                            position_offset_before_restore_align
                        )
                    if (
                        not self.running
                        and visible_allocation_demand > 0
                    ):
                        before_pools = self._kve_gpu_block_pool_diag_summary()
                        before_managed = self._kve_managed_context_diag_summary()
                        released_hot_gpu_blocks = (
                            self._managed_context_free_hot_gpu_for_blocks(
                                visible_allocation_demand,
                                protected=(
                                    self._managed_context_cpu_archive_protected_keys()
                                ),
                            )
                        )
                        if released_hot_gpu_blocks:
                            logger.warning(
                                "[MANAGED-CONTEXT-HOT-GPU-ALLOC-PRESSURE] "
                                "req=%s released=%d needed=%d "
                                "pools_before=%s managed_before=%s "
                                "pools_after=%s managed_after=%s",
                                request.request_id[:8],
                                released_hot_gpu_blocks,
                                visible_allocation_demand,
                                before_pools,
                                before_managed,
                                self._kve_gpu_block_pool_diag_summary(),
                                self._kve_managed_context_diag_summary(),
                            )
                            continue
                    pressure_preempted_req = (
                        self._preempt_request_for_kv_swap_pressure(
                            scheduled_timestamp,
                            reason="waiting-alloc-pressure",
                            protected_request_ids={
                                request.request_id,
                                *(
                                    scheduled_req.request_id
                                    for scheduled_req in scheduled_running_reqs
                                ),
                            },
                        )
                    )
                    if pressure_preempted_req is not None:
                        preempted_reqs.append(pressure_preempted_req)
                        break
                    if (
                        not self.running
                        and self._release_phase4_pressure_pin(
                            "waiting-alloc-pressure"
                        )
                    ):
                        continue
                    break

                self._restamp_reprefill_logical_starts(
                    request, reason="waiting-alloc"
                )
                if pending_inherit_event is not None:
                    request.compaction_events.append(pending_inherit_event)
                    logger.info(
                        "[COMPACT/inherit] req=%s seeded "
                        "position_offset=%d (sys=%d, nuf_len=%d, "
                        "cached_tokens=%d); synthetic event emitted "
                        "for trainer mirror",
                        request.request_id[:8],
                        pending_inherit_event.position_offset_after,
                        pending_inherit_event.evict_start,
                        pending_inherit_event.new_user_fragment_len,
                        request.num_prompt_tokens
                        - pending_inherit_event.new_user_fragment_len,
                    )

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    continue

                if (
                    request.request_id
                    not in self._managed_context_active_restores
                    and not restore_defer_until_prefill
                ):
                    self._activate_managed_context_restore(request, restore_spans)
                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

                if os.environ.get("KVE_SERIALIZE_WAITING_REQUESTS") == "1":
                    logger.info(
                        "[SCHED-DIAG] scheduled one waiting request; "
                        "deferring remaining waiting requests to the next "
                        "engine step"
                    )
                    break

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # KV cache compaction: in-step admission eviction.
        # For any request in `_pending_admission_compaction_ids` whose
        # prefill completes this step, run the eviction loop now —
        # before block_ids/all_token_ids are snapshotted into the
        # SchedulerOutput. The worker then prefills the post-eviction
        # sequence in a single normal forward; `new_user_fragment` K/V
        # is computed under attention over the kept context only because
        # `turn_to_evict`'s blocks are no longer in the block_table.
        (
            total_num_scheduled_tokens,
            any_inline_evicted,
        ) = self._apply_inline_admission_eviction(
            num_scheduled_tokens, total_num_scheduled_tokens
        )

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention. Skip cascade
        # attention when in-step eviction fired this step — the common
        # prefix computed against post-eviction block_tables may not
        # match what the kernel reads given the just-spliced layout.
        # Also skip it for active managed-context restores: the scheduler-side
        # KV manager only knows about the visible request blocks, while the
        # worker prepends hidden restored blocks to the block-table row. A
        # visible-only common-prefix length would index the wrong physical
        # prefix in that spliced worker row.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            has_active_managed_context_restore = any(
                req.request_id in self._managed_context_active_restores
                for req in self.running
            )
            if (
                self.running
                and not any_inline_evicted
                and not has_active_managed_context_restore
            ):
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        self._maybe_release_phase4_stall_pressure_pin(
            total_num_scheduled_tokens=total_num_scheduled_tokens
        )

        self._kve_maybe_log_scheduler_liveness(
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            token_budget=token_budget,
            preempted_reqs=preempted_reqs,
            scheduled_running_reqs=scheduled_running_reqs,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_resumed_reqs=scheduled_resumed_reqs,
        )

        # Construct the scheduler output.
        # Compute protected_prefix_len per request for the 2-piece
        # position fix (plans/piecewise_position_offset.md). When
        # compaction is disabled this is 0 (no offset is ever applied
        # anyway). For compacted requests this is the sys boundary that
        # the worker uses to gate position_offset application.
        def _ppl(req: Request) -> int:
            return self._worker_protected_prefix_len(req)

        def _hidden(req: Request) -> tuple[tuple[list[int], ...], int, list[str]]:
            return self._managed_context_active_hidden_kv(req.request_id)

        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                    protected_prefix_len=_ppl(req),
                    hidden_kv_block_ids=_hidden(req)[0],
                    hidden_kv_num_tokens=_hidden(req)[1],
                    hidden_kv_span_ids=_hidden(req)[2],
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    protected_prefix_len=_ppl(req),
                    hidden_kv_block_ids=_hidden(req)[0],
                    hidden_kv_num_tokens=_hidden(req)[1],
                    hidden_kv_span_ids=_hidden(req)[2],
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        # KV cache compaction auto-pad: collect request IDs whose
        # scheduled tokens are filler padding (no sampling for these).
        no_sample_req_ids: set[str] = set()
        if self._compaction_block_aligned_finish:
            for req_id in num_scheduled_tokens:
                req = self.requests.get(req_id)
                if req is not None and req.padding_pending:
                    no_sample_req_ids.add(req_id)

        if os.environ.get("KVE_TRACE_SCHED_SIG", "0") == "1":
            self._kve_sched_sig_step += 1
            req_range_env = os.environ.get("KVE_TRACE_SCHED_REQ_NUM_RANGE", "")
            req_nums: set[int] | None = None
            if req_range_env:
                req_nums = set()
                for part in req_range_env.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if ":" in part:
                        lo_s, hi_s = part.split(":", 1)
                        lo, hi = int(lo_s), int(hi_s)
                        req_nums.update(range(lo, hi + 1))
                    else:
                        req_nums.add(int(part))

            def _req_num(req_id: str) -> int | None:
                try:
                    return int(req_id.split("-", 1)[0])
                except (TypeError, ValueError):
                    return None

            for req_id, num_sched in num_scheduled_tokens.items():
                req_num = _req_num(req_id)
                if req_nums is not None and req_num not in req_nums:
                    continue
                req = self.requests.get(req_id)
                if req is None:
                    continue
                pre = req.num_computed_tokens
                post = pre + num_sched
                if req.padding_pending:
                    phase = "pad"
                elif pre < req.num_prompt_tokens:
                    phase = "prefill" if post <= req.num_prompt_tokens else "mixed"
                else:
                    phase = "decode"
                token_ids = req.all_token_ids[pre:post]
                logger.warning(
                    "[SCHED-SIG] step=%d req=%s req_num=%s phase=%s "
                    "sched=%d pre=%d post=%d prompt=%d tokens=%d "
                    "cached=%d pos_off=%d pad_pending=%s no_sample=%s "
                    "events=%d tok_head=%s tok_tail=%s",
                    self._kve_sched_sig_step,
                    req_id[:12],
                    str(req_num),
                    phase,
                    num_sched,
                    pre,
                    post,
                    req.num_prompt_tokens,
                    req.num_tokens,
                    req.num_cached_tokens,
                    req.position_offset,
                    req.padding_pending,
                    req_id in no_sample_req_ids,
                    len(req.compaction_events or []),
                    token_ids[:8],
                    token_ids[-8:],
                )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
            no_sample_req_ids=no_sample_req_ids,
        )
        scheduler_output.managed_context_transfer_metadata = (
            self._drain_managed_context_transfer_metadata()
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _preempt_request(
        self,
        request: Request,
        timestamp: float,
        *,
        allow_async_kv_swap: bool = True,
    ) -> PreemptionResult:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        # Compacted requests carry a trimmed token view plus a non-zero RoPE
        # frame. Flush and re-prefill that compacted state instead of treating
        # this as ordinary preemption.
        if request.position_offset > 0:
            if not allow_async_kv_swap:
                if self._request_kv_swap_strict_preempt_enabled():
                    return PreemptionResult(
                        _PREEMPTION_DEFERRED,
                        "request was scheduled in the current scheduler step",
                    )
            else:
                swap_result = self._preempt_request_for_kv_swap(
                    request,
                    timestamp,
                    reason="compacted",
                )
                if swap_result.kind in (
                    _PREEMPTION_ASYNC_PENDING,
                    _PREEMPTION_DEFERRED,
                ):
                    return swap_result
            if self._preempt_request_for_reprefill(
                request,
                timestamp,
                reason="compacted",
            ):
                return PreemptionResult(_PREEMPTION_FREED)
            logger.warning(
                "Attempted to preempt compacted request %s "
                "(position_offset=%d) — aborting request instead.",
                request.request_id, request.position_offset,
            )
            request.status = RequestStatus.FINISHED_ERROR
            self._kve_log_compacted_preempt_abort(request, phase="before-free")
            self._free_request(request)
            self._kve_log_compacted_preempt_abort(request, phase="after-free")
            self._queue_finished_request_output(request)
            return PreemptionResult(_PREEMPTION_FINISHED)
        if (
            request.request_id in self._managed_context_active_restores
            or request.request_id in self._managed_context_deferred_restores
        ):
            if not allow_async_kv_swap:
                if self._request_kv_swap_strict_preempt_enabled():
                    return PreemptionResult(
                        _PREEMPTION_DEFERRED,
                        "request was scheduled in the current scheduler step",
                    )
            else:
                swap_result = self._preempt_request_for_kv_swap(
                    request,
                    timestamp,
                    reason="managed-context-restore",
                )
                if swap_result.kind in (
                    _PREEMPTION_ASYNC_PENDING,
                    _PREEMPTION_DEFERRED,
                ):
                    return swap_result
            if self._preempt_request_for_reprefill(
                request,
                timestamp,
                reason="managed-context-restore",
            ):
                return PreemptionResult(_PREEMPTION_FREED)
            logger.warning(
                "Attempted to preempt managed-context restore request %s; "
                "aborting request instead.",
                request.request_id,
            )
            request.status = RequestStatus.FINISHED_ERROR
            self._free_request(request)
            self._queue_finished_request_output(request)
            return PreemptionResult(_PREEMPTION_FINISHED)
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        return PreemptionResult(_PREEMPTION_FREED)

    def _preempt_request_for_kv_swap(
        self,
        request: Request,
        timestamp: float,
        *,
        reason: str,
    ) -> PreemptionResult:
        error = self._start_request_kv_swap_out(request, reason)
        if error is not None:
            if os.environ.get("KVE_TRACE_REQUEST_KV_SWAP") == "1":
                logger.warning(
                    "[REQUEST-KV-SWAP-SKIP] req=%s reason=%s error=%s "
                    "action=%s",
                    request.request_id[:8],
                    reason,
                    error,
                    "defer"
                    if self._request_kv_swap_preempt_error_defers(error)
                    else "fallback",
                )
            if self._request_kv_swap_preempt_error_defers(error):
                return PreemptionResult(_PREEMPTION_DEFERRED, error)
            return PreemptionResult(_PREEMPTION_FAILED, error)
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        request.num_preemptions += 1
        if request.spec_token_ids:
            request.spec_token_ids = []
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
        self.waiting.prepend_request(request)
        logger.warning(
            "[REQUEST-KV-SWAP-OUT] req=%s reason=%s blocks=%d "
            "computed=%d position_offset=%d preemptions=%d",
            request.request_id[:8],
            reason,
            self._request_kv_swaps[request.request_id].kv_block_count,
            request.num_computed_tokens,
            request.position_offset,
            request.num_preemptions,
        )
        return PreemptionResult(_PREEMPTION_ASYNC_PENDING)

    def _preempt_request_for_reprefill(
        self,
        request: Request,
        timestamp: float,
        *,
        reason: str,
    ) -> bool:
        """Flush a replayable request and requeue it for full re-prefill."""
        raw_enabled = os.environ.get("KVE_COMPACTED_REPREFILL_ON_PREEMPT", "1")
        if raw_enabled.lower() in ("0", "false", "no", "off"):
            return False
        if request.padding_pending or request.num_output_placeholders:
            logger.warning(
                "[COMPACT-REPREFILL-SKIP] req=%s reason=%s "
                "padding_pending=%s output_placeholders=%d",
                request.request_id[:8],
                reason,
                request.padding_pending,
                request.num_output_placeholders,
            )
            return False
        if request.prompt_token_ids is None:
            logger.warning(
                "[COMPACT-REPREFILL-SKIP] req=%s reason=%s "
                "prompt_token_ids unavailable",
                request.request_id[:8],
                reason,
            )
            return False

        request_id = request.request_id
        old_computed = request.num_computed_tokens
        old_position_offset = request.position_offset
        old_prompt_tokens = request.num_prompt_tokens
        old_num_tokens = request.num_tokens
        old_compaction_events = len(request.compaction_events or [])

        if not self._mark_request_for_full_reprefill(
            request,
            "reprefill-preempt",
            status=RequestStatus.PREEMPTED,
            free_request_kv=True,
            count_preemption=True,
            skip_log_reason=reason,
        ):
            return False
        self.waiting.prepend_request(request)
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
        logger.warning(
            "[COMPACT-REPREFILL] req=%s reason=%s prompt=%d tokens=%d "
            "computed=%d position_offset=%d events=%d preemptions=%d "
            "timestamp=%.6f",
            request_id[:8],
            reason,
            old_prompt_tokens,
            old_num_tokens,
            old_computed,
            old_position_offset,
            old_compaction_events,
            request.num_preemptions,
            timestamp,
        )
        return True

    def _mark_request_for_full_reprefill(
        self,
        request: Request,
        reason: str,
        *,
        status: RequestStatus | None = None,
        position_offset: int | None = None,
        phase4_pin_release: bool = False,
        free_request_kv: bool = False,
        count_preemption: bool = False,
        skip_log_reason: str | None = None,
    ) -> bool:
        """Reset scheduler-visible state so a request can replay its prompt."""
        request_id = request.request_id
        if request.prompt_token_ids is None:
            logger.warning(
                "[COMPACT-REPREFILL-SKIP] req=%s reason=%s "
                "prompt_token_ids unavailable",
                request_id[:8],
                skip_log_reason or reason,
            )
            return False

        if request_id in self._managed_context_pending_loads:
            pending_released = self._release_managed_context_pending_load(
                request_id,
                reason,
                only_if_safe=True,
            )
            if not pending_released:
                logger.warning(
                    "[COMPACT-REPREFILL-SKIP] req=%s reason=%s "
                    "managed-context load in flight",
                    request_id[:8],
                    skip_log_reason or reason,
                )
                return False

        preserve_phase4_reprefill = (
            phase4_pin_release
            or self._phase4_pin_release_reprefill_active(request)
        )
        self._release_managed_context_active_restore(request_id, reason)
        self._release_managed_context_deferred_restore(request_id, reason)
        self._release_managed_context_restore_reservation(request_id, reason)
        self._reserve_managed_context_restore_request(request)
        self._pending_admission_compaction_ids.discard(request_id)
        self.prev_step_scheduled_req_ids.discard(request_id)
        if free_request_kv:
            self.kv_cache_manager.free(request)
            self.encoder_cache_manager.free(request)

        if status is not None:
            request.status = status
        if position_offset is not None:
            request.position_offset = position_offset
        request.num_computed_tokens = 0
        request.num_external_computed_tokens = 0
        request.num_cached_tokens = -1
        if count_preemption:
            request.num_preemptions += 1
        request.spec_token_ids = []
        request.is_prefill_chunk = False
        request.needs_rebuild = True
        request.skip_reading_prefix_cache = True
        request._kve_reprefill_after_flush = True  # type: ignore[attr-defined]
        request._kve_phase4_reprefill_after_pin_release = (  # type: ignore[attr-defined]
            preserve_phase4_reprefill
        )
        return True

    def _restamp_reprefill_logical_starts(
        self,
        request: Request,
        *,
        reason: str,
    ) -> None:
        if (
            not getattr(request, "_kve_reprefill_after_flush", False)
            or request.position_offset <= 0
        ):
            return
        protected_prefix_len = self._worker_protected_prefix_len(request)
        changed = 0
        for manager in self.kv_cache_manager.coordinator.single_type_managers:
            blocks = manager.req_to_blocks.get(request.request_id, [])
            block_size = manager.block_size
            for block_idx, block in enumerate(blocks):
                if block.is_null or block.logical_start < 0:
                    continue
                physical_start = block_idx * block_size
                target_logical_start = physical_start
                if physical_start >= protected_prefix_len:
                    target_logical_start += request.position_offset
                if block.logical_start != target_logical_start:
                    block.logical_start = target_logical_start
                    changed += 1
        if changed and os.environ.get("KVE_TRACE_COMPACT_REPREFILL") == "1":
            logger.warning(
                "[COMPACT-REPREFILL-RESTAMP] req=%s reason=%s "
                "changed=%d position_offset=%d protected_prefix_len=%d",
                request.request_id[:8],
                reason,
                changed,
                request.position_offset,
                protected_prefix_len,
            )

    def _clear_reprefill_after_flush_if_prompt_ready(
        self,
        request: Request,
        *,
        reason: str,
    ) -> None:
        if not getattr(request, "_kve_reprefill_after_flush", False):
            return
        if request.num_computed_tokens < request.num_prompt_tokens:
            return
        request._kve_reprefill_after_flush = False  # type: ignore[attr-defined]
        if getattr(
            request,
            "_kve_phase4_reprefill_after_pin_release",
            False,
        ):
            request._kve_phase4_reprefill_after_pin_release = False  # type: ignore[attr-defined]
        if os.environ.get("KVE_TRACE_COMPACT_REPREFILL") == "1":
            logger.warning(
                "[COMPACT-REPREFILL-DONE] req=%s reason=%s "
                "computed=%d prompt=%d position_offset=%d",
                request.request_id[:8],
                reason,
                request.num_computed_tokens,
                request.num_prompt_tokens,
                request.position_offset,
            )

    def _phase4_requeue_prefix_miss_for_reprefill(
        self,
        request: Request,
        reason: str,
    ) -> bool:
        if not self._phase4_prefix_miss_reprefill_enabled():
            return False
        if request.prompt_token_ids is None:
            return False

        request_id = request.request_id
        if not self._mark_request_for_full_reprefill(
            request,
            "phase4-prefix-miss-reprefill",
            phase4_pin_release=True,
            skip_log_reason=reason,
        ):
            return False

        logger.warning(
            "[PHASE4-PREFIX-REPREFILL] req=%s prompt=%d "
            "position_offset=%d reason=%s",
            request_id[:8],
            request.num_prompt_tokens,
            request.position_offset,
            reason,
        )
        return True

    @staticmethod
    def _phase4_pin_release_reprefill_active(request: Request) -> bool:
        return bool(
            getattr(
                request,
                "_kve_phase4_reprefill_after_pin_release",
                False,
            )
            and getattr(request, "_kve_reprefill_after_flush", False)
        )

    # --- Compaction helpers ---

    def _resolve_turn_end_token_id(self, request: Request) -> int | None:
        """Get the message-end token id for turn-mode compaction.

        Falls back to request.sampling_params.eos_token_id if not configured
        explicitly. For Qwen3 / ChatML these match (<|im_end|> = 151645).
        Cached on the scheduler after first resolution.
        """
        if self._compaction_turn_end_token_id is not None:
            return self._compaction_turn_end_token_id
        eos_id = request.sampling_params.eos_token_id
        if eos_id is not None:
            self._compaction_turn_end_token_id = eos_id
            logger.warning(
                "[COMPACT] turn mode: using eos_token_id=%d as message-end "
                "marker (set --compaction-turn-end-token-id explicitly to "
                "override)",
                eos_id,
            )
            return eos_id
        return None

    def _scan_new_turn_boundaries(self, request: Request) -> None:
        """Extend request.turn_end_positions for any <|im_end|> tokens
        appended since the last scan. O(num_new_tokens). No-op if the
        message-end token id is not resolvable yet.
        """
        end_id = self._resolve_turn_end_token_id(request)
        if end_id is None:
            return
        toks = request._all_token_ids
        start = request.last_turn_scan_pos
        end = len(toks)
        if start >= end:
            return
        positions = request.turn_end_positions
        for i in range(start, end):
            if toks[i] == end_id:
                positions.append(i + 1)  # position AFTER the im_end token
        request.last_turn_scan_pos = end

    def _turn_mode_effective_prompt(self, request: Request) -> int:
        """Eviction boundary in turn mode = end of the system prompt
        = turn_end_positions[0]. Falls back to full prompt while the
        first <|im_end|> has not yet been seen (conservative — no
        compaction can fire in that window).
        """
        self._scan_new_turn_boundaries(request)
        positions = request.turn_end_positions
        if not positions:
            return request.num_prompt_tokens
        return positions[0]

    def _num_live_completed_turns(self, request: Request) -> int:
        """Number of completed (user+assistant) turns currently in the
        request's KV (i.e. not yet evicted).

        turn_end_positions only contains markers for non-evicted content:
          [end_sys, end_U_{e+1}, end_A_{e+1}, end_U_{e+2}, end_A_{e+2}, ...]
        where e = num_turns_evicted. Each completed live turn contributes
        2 positions (user + assistant). The leading end_sys contributes 1.
        Hence: live_completed_turns = (len(positions) - 1) // 2.

        num_turns_evicted is a separate counter used for wire metadata
        (CompactionEvent.last_turn_evicted) and is NOT subtracted here.
        """
        n = len(request.turn_end_positions)
        if n < 1:
            return 0
        return (n - 1) // 2

    def _effective_prompt_tokens(self, request: Request) -> int:
        """Eviction boundary: protected prefix if set, else full prompt.

        When compaction_max_turns > 0, route through turn mode (system
        prompt is the only protected region; live turns are evictable).
        When compaction_protected_prefix_tokens is -1 (auto), detect the
        system prompt boundary by finding the first eos_token in
        prompt_token_ids (marks the end of the system message in chat
        templates like ChatML). The result is cached on the request.
        """
        if self._compaction_max_turns > 0:
            return self._turn_mode_effective_prompt(request)
        if self._compaction_protected_prefix > 0:
            return min(self._compaction_protected_prefix,
                       request.num_prompt_tokens)
        if self._compaction_protected_prefix == -1:
            cached = getattr(request, "_auto_protected_prefix", None)
            if cached is not None:
                return min(cached, request.num_prompt_tokens)
            # Scan for the first eos token in the prompt.
            eos_id = request.sampling_params.eos_token_id
            boundary = request.num_prompt_tokens  # fallback: full prompt
            if eos_id is not None and request.prompt_token_ids is not None:
                for i, tok_id in enumerate(request.prompt_token_ids):
                    if tok_id == eos_id:
                        boundary = i + 1  # protect up to and including eos
                        break
            request._auto_protected_prefix = boundary  # type: ignore[attr-defined]
            logger.warning(
                "[COMPACT] auto-detected system prompt boundary: "
                "%d tokens (prompt_len=%d)",
                boundary, request.num_prompt_tokens,
            )
            return min(boundary, request.num_prompt_tokens)
        return request.num_prompt_tokens

    def _worker_protected_prefix_len(self, request: Request) -> int:
        """Protected-prefix boundary shipped to the worker.

        The raw protected prefix can end in the middle of a KV block, but
        prefix-cache reuse is block-granular. The worker's piecewise RoPE
        gate must therefore transition on a block boundary so a cached block
        is not interpreted partly in the no-offset frame and partly in the
        post-admission offset frame.
        """
        if not self._compaction_enabled:
            return 0
        sys_end = self._effective_prompt_tokens(request)
        bs = self._compaction_block_size
        if bs > 0 and sys_end > 0:
            sys_end = ((sys_end + bs - 1) // bs) * bs
            sys_end = min(sys_end, request.num_prompt_tokens)
        return sys_end

    def _should_compact(self, request: Request) -> bool:
        """Check if compaction should fire for this request.

        In block-FIFO mode, defers to mgr.needs_compaction (window guard
        + full-block safety guard).

        In turn mode, the primary trigger is `live_turns >= max_turns`
        — the live turn count exceeds the configured ceiling. The
        block-level guard from mgr.needs_compaction still acts as a
        secondary safety net (won't fire if there's not enough computed
        tokens past the protected prefix to evict a stride's worth of
        blocks).
        """
        effective_prompt = self._effective_prompt_tokens(request)
        if self._compaction_max_turns > 0:
            # Turn-mode trigger. positions only tracks non-evicted content,
            # so live_completed_turns comes directly from its length.
            # No mgr.needs_compaction fallback: the over-decrement guard
            # there is for block-FIFO; turn mode evicts an exact whole-block
            # range bounded by num_computed_tokens itself, so the same
            # invariant is enforced inside _plan_turn_evict_range.
            self._scan_new_turn_boundaries(request)
            return (
                self._num_live_completed_turns(request)
                >= self._compaction_max_turns
            )
        for mgr in self.kv_cache_manager.coordinator.single_type_managers:
            if isinstance(mgr, CompactingKVCacheManager) and mgr.needs_compaction(
                request.request_id,
                request.num_computed_tokens,
                effective_prompt,
            ):
                return True
        return False

    def _run_admission_eviction_loop(
        self,
        request: Request,
        effective_num_computed: int | None = None,
    ) -> int:
        """Run the turn-mode admission eviction loop on a single request.

        Shared between two callers:
          - `update_from_output` admission branch (post-Phase-D): fires
            in the same step as the worker's two-phase forward, so the
            request-side state is mutated in lockstep with input_batch.
          - `_update_request_as_session` (per-call mid-session admission,
            fired after streaming-input session extension and BEFORE the
            new content is prefilled — see the "Operative intent" section
            of plans/connect_admission_events_to_trainer.md).

        Each iteration calls _compact_request with post_prefill_admission=True,
        which:
          1. Emits a CompactionEvent with num_output_tokens_at_compaction=0
             (admission semantics).
          2. Trims prompt_token_ids in addition to _all_token_ids so
             len(prompt_token_ids) stays consistent with num_prompt_tokens.
          3. Generalized decrement+shift in _apply_trim handles
             num_computed_tokens >= total_evicted correctly.

        Returns the number of eviction iterations that fired.
        """
        iterations = 0
        while True:
            self._scan_new_turn_boundaries(request)
            if (
                self._num_live_completed_turns(request)
                < self._compaction_max_turns
            ):
                break
            evicted = self._compact_request(
                request,
                post_prefill_admission=True,
                effective_num_computed=effective_num_computed,
            )
            if evicted == 0:
                logger.warning(
                    "[COMPACT] admission eviction bail: req=%s "
                    "after %d iters (no further eviction possible)",
                    request.request_id[:8],
                    iterations,
                )
                break
            # Propagate position_offset / block_table changes to the worker
            # via the rebuild path. CachedRequestData's position_offsets dict
            # is populated only for requests in rebuild_req_ids
            # (scheduler.py:1893-1898). Without these two lines,
            # position_offset stays at 0 on the worker side and decode-step
            # Q rotates at `physical + 0` instead of
            # `physical + total_evicted`.
            request.needs_rebuild = True
            self.prev_step_scheduled_req_ids.discard(request.request_id)
            iterations += 1
        return iterations

    def _apply_inline_admission_eviction(
        self,
        num_scheduled_tokens: dict[str, int],
        total_num_scheduled_tokens: int,
    ) -> tuple[int, bool]:
        """Run admission eviction in-step for any pending request whose
        prefill completes this step.

        Mutates request state (block_table, position_offset,
        _all_token_ids, prompt_token_ids) via `_run_admission_eviction_
        loop` and decrements `num_scheduled_tokens[req_id]` by the
        number of tokens trimmed off the prompt so the worker prefills
        only the post-eviction sequence in a single forward.

        Called from `schedule()` BEFORE `num_common_prefix_blocks` is
        computed and BEFORE `_make_cached_request_data` snapshots
        block_ids — the rebuild flag set by the inner loop takes effect
        on this same step's SchedulerOutput.

        Returns:
            (updated_total_num_scheduled_tokens, any_eviction_fired)
        """
        if (
            not self._compaction_enabled
            or not self._pending_admission_compaction_ids
            or self._compaction_max_turns <= 0
        ):
            return total_num_scheduled_tokens, False

        any_evicted = False
        delta_total = 0
        # Snapshot since `_run_admission_eviction_loop` discards from the
        # set via its inner helpers, and we also discard explicitly.
        pending_snapshot = list(self._pending_admission_compaction_ids)
        for req_id in pending_snapshot:
            num_new = num_scheduled_tokens.get(req_id, 0)
            if num_new == 0:
                continue
            request = self.requests.get(req_id)
            if request is None:
                self._pending_admission_compaction_ids.discard(req_id)
                continue
            # Only fire when prefill completes this step. Chunked-prefill
            # mid-chunks defer to the step that actually finishes prefill.
            if (
                request.num_computed_tokens + num_new
                < request.num_prompt_tokens
            ):
                continue

            pre_prompt = request.num_prompt_tokens
            # Snapshot the pre-eviction "new user fragment" length: this
            # is the number of prompt tokens that have NOT been written
            # to KV cache yet (= the tail past the prefix-cache match).
            # Invariant: compaction must never drop these — they're the
            # user's freshly-submitted content. If the post-eviction
            # residual is smaller, V silently truncated the user's
            # prompt (typically because turn-boundary block-align
            # snapped past `cached_tokens` into the new_user region).
            pre_new_user_len = pre_prompt - request.num_computed_tokens
            pre_cached_tokens = request.num_computed_tokens
            # The prefill kernel runs AFTER this method returns. At eviction
            # time `num_computed_tokens` is therefore the live prefix-cache
            # hit length, and the tail past it is the freshly submitted
            # fragment the worker still needs to prefill. Do not use the full
            # prompt length as the eviction ceiling here: turn-boundary
            # align-up can otherwise trim into that fresh tail, producing
            # admission events whose evict range extends past the cache the
            # trainer can replay.
            iterations = self._run_admission_eviction_loop(request)
            self._pending_admission_compaction_ids.discard(req_id)
            if iterations == 0:
                continue
            total_evicted = pre_prompt - request.num_prompt_tokens
            if total_evicted <= 0:
                continue
            # ── DIAGNOSTIC ASSERT: compaction must not drop new-user
            # content. The post-eviction "residual prefill" length
            # (num_prompt_post - num_computed_post) is what the worker
            # will prefill in this step. It MUST equal the pre-eviction
            # new_user_fragment length (= the tail past the original
            # prefix-cache match). If it's smaller, V evicted past the
            # cache boundary and dropped tokens the user submitted but
            # V never wrote to KV — they will not appear in any K/V
            # cache row, silently truncating the user's prompt.
            # Gated by env var so this stays opt-in; flip on during
            # diagnostic runs to surface drops, off in production.
            if os.environ.get("KVE_ASSERT_NO_NEW_USER_DROP", "") == "1":
                post_new_user_len = (
                    request.num_prompt_tokens - request.num_computed_tokens
                )
                assert post_new_user_len == pre_new_user_len, (
                    f"[COMPACT/assert] req={req_id[:8]} compaction "
                    f"dropped new-user content: pre_new_user_len="
                    f"{pre_new_user_len} post_new_user_len="
                    f"{post_new_user_len} (delta="
                    f"{pre_new_user_len - post_new_user_len} tokens "
                    f"truncated from the user's prompt). "
                    f"pre_cached_tokens={pre_cached_tokens} "
                    f"pre_prompt={pre_prompt} "
                    f"post_prompt={request.num_prompt_tokens} "
                    f"post_computed={request.num_computed_tokens} "
                    f"total_evicted={total_evicted}. "
                    f"This means `_plan_turn_evict_range` chose an "
                    f"evict_end that crossed `cached_tokens` (likely "
                    f"because block-align-up of the turn boundary "
                    f"snapped past the prefix-cache match)."
                )
            # Recompute `num_scheduled_tokens` from post-eviction state.
            # Naively subtracting `total_evicted` from `num_new` is wrong
            # under partial / warm prefix-cache hits: when the cache
            # already covered some of the now-evicted positions,
            # `_apply_trim` decrements `num_computed_tokens` by the
            # overlap (not by `total_evicted`), so the residual prefill
            # work is just `num_prompt_post - num_computed_post`. The
            # naive subtraction would drive this NEGATIVE (e.g.
            # `num_scheduled 36 -> -44` for a 80-tok eviction whose
            # 80-tok overlap was entirely inside a 112-tok prefix-cache
            # prefix) and trip
            # `assert total_num_scheduled_tokens > 0` in the worker's
            # `_prepare_inputs`.
            new_num_scheduled = (
                request.num_prompt_tokens - request.num_computed_tokens
            )
            assert new_num_scheduled >= 0, (
                f"compaction: req={req_id[:8]} post-evict scheduled<0: "
                f"num_prompt={request.num_prompt_tokens}, "
                f"num_computed={request.num_computed_tokens}, "
                f"total_evicted={total_evicted}, num_new_pre={num_new}"
            )
            num_scheduled_tokens[req_id] = new_num_scheduled
            delta_total += new_num_scheduled - num_new
            any_evicted = True
            logger.info(
                "[COMPACT/inline] req=%s prefill completes this step; "
                "evicted %d tokens across %d iter(s); "
                "num_scheduled %d -> %d, num_prompt %d -> %d, "
                "num_computed=%d, position_offset=%d",
                req_id[:8], total_evicted, iterations,
                num_new, new_num_scheduled,
                pre_prompt, request.num_prompt_tokens,
                request.num_computed_tokens,
                request.position_offset,
            )

        return total_num_scheduled_tokens + delta_total, any_evicted

    def _compact_request(
        self,
        request: Request,
        *,
        post_prefill_admission: bool = False,
        effective_num_computed: int | None = None,
    ) -> int:
        """Compact a request: splice blocks, trim tokens, update state.

        After this, the request looks like a shorter sequence to all consumers.

        Args:
            request: the Request to compact.
            post_prefill_admission: when True, this call is firing from the
                deferred-admission hook (Path 2: evict after prefill). In
                that case, the evict range overlaps the prompt — so the
                `prompt_token_ids` Python list must also be trimmed (mirror
                of pre-prefill admission's behavior), and the emitted event
                is tagged with num_output_tokens_at_compaction=0 (admission
                semantics) even though num_total_generated may be >0
                because vLLM already sampled the first decoded token from
                the prefill-end logit before this hook ran.
        """
        # Locate the compaction-capable manager up front so we can use its
        # block_size to plan the eviction range (turn mode) before calling
        # the physical compact_request.
        compaction_mgr: "CompactingKVCacheManager | None" = None
        for mgr in self.kv_cache_manager.coordinator.single_type_managers:
            if isinstance(mgr, CompactingKVCacheManager):
                compaction_mgr = mgr
                break
        if compaction_mgr is None:
            return 0
        block_size = compaction_mgr.block_size

        effective_prompt = self._effective_prompt_tokens(request)

        # Plan the eviction range. Turn mode computes a turn-aligned range
        # snapped inward to block boundaries; block-FIFO mode lets the
        # manager pick the default range from (effective_prompt, stride).
        explicit_block_range: tuple[int, int] | None = None
        last_turn_evicted = -1
        stride_used = 0
        archived_span_ids: list[str] | None = []
        if self._compaction_max_turns > 0:
            plan = self._plan_turn_evict_range(
                request, block_size,
                effective_num_computed=effective_num_computed,
            )
            if plan is None:
                return 0  # Nothing safe to evict (e.g. too-short turns)
            evict_start, evict_end, last_turn_evicted, stride_used = plan
            explicit_block_range = (
                evict_start // block_size, evict_end // block_size
            )
            total_evicted = evict_end - evict_start
            archived_span_ids = self._archive_managed_context_span(
                request,
                compaction_mgr=compaction_mgr,
                evict_start=evict_start,
                evict_end=evict_end,
                explicit_block_range=explicit_block_range,
                last_turn_evicted=last_turn_evicted,
                stride_used=stride_used,
            )
            if archived_span_ids is None:
                return 0
            tokens_evicted = compaction_mgr.compact_request(
                request.request_id,
                effective_prompt,
                explicit_block_range=explicit_block_range,
            )
            if tokens_evicted != total_evicted:
                # Manager refused (out-of-bounds guard tripped). Should be
                # impossible given _plan_turn_evict_range's checks.
                trace_id = self._managed_context_trace_id(request)
                for span_id in archived_span_ids:
                    self._release_managed_context_span(
                        (trace_id, span_id), "compact-refused"
                    )
                return 0
        else:
            tokens_evicted = compaction_mgr.compact_request(
                request.request_id, effective_prompt
            )
            if tokens_evicted == 0:
                return 0
            total_evicted = tokens_evicted
            evict_start = (
                (effective_prompt + block_size - 1) // block_size
            ) * block_size
            evict_end = evict_start + total_evicted

        # Debug-only: snapshot the actual evicted token ids for inspection.
        # Gated on an env var so we don't pay the serialization cost in
        # production runs. Consumer can detokenize via `tokenizer.decode`.
        evicted_token_ids: list[int] = []
        if os.environ.get("VLLM_COMPACTION_DEBUG_TOKENS"):
            evicted_token_ids = list(
                request._all_token_ids[evict_start:evict_end]
            )

        # Snapshot the KEPT slice (in pre-event coords) BEFORE mutation.
        # Scheduler is the single source of truth for what physically
        # survives this eviction. Surfaced on the event so consumers
        # don't re-derive the slice from scalar fields (which is how the
        # trainer's evict_start_per_boundary plumbing went dead).
        pre_event_len = len(request._all_token_ids)
        kept_indices = (
            list(range(0, evict_start))
            + list(range(evict_end, pre_event_len))
        )
        kept_token_ids = (
            list(request._all_token_ids[0:evict_start])
            + list(request._all_token_ids[evict_end:pre_event_len])
        )

        # Compute event metadata that we'll use AFTER _apply_trim and
        # smart-bump. The event itself is emitted later so that
        # `position_offset_after` reflects the FINAL post-bump value.
        if self._compaction_max_turns > 0:
            num_turns_evicted_after = (
                request.num_turns_evicted + stride_used
            )
        else:
            num_turns_evicted_after = 0
        # New_user_fragment_len: length of the in-progress turn's tail
        # (the chunk that, under the new single-forward design, vLLM
        # prefills under POST-eviction K/V). Only meaningful for
        # admission events; zero on mid-gen events since they don't
        # expose a fragment boundary. Emitted on the CompactionEvent
        # so the trainer's segmented_forward can mirror the split.
        new_user_fragment_len = self._compute_new_user_fragment_len(
            request, evict_end
        ) if post_prefill_admission else 0

        logger.warning(
            "[COMPACT] req=%s effective_prompt=%d num_prompt=%d "
            "evict=[%d,%d) total=%d generated=%d turn_mode=%s last_turn=%d",
            request.request_id[:8],
            effective_prompt,
            request.num_prompt_tokens,
            evict_start, evict_end,
            total_evicted,
            request.num_total_generated,
            self._compaction_max_turns > 0,
            last_turn_evicted,
        )

        # --- Mutate request to look like a shorter sequence ---
        # Delegates to the shared _apply_trim helper. For mid-gen the
        # evict range is in the generated output, so prompt_token_ids
        # stays untouched. For post-prefill admission, the evict range
        # overlaps the prompt, so we MUST trim prompt_token_ids to keep
        # `len(prompt_token_ids) == num_prompt_tokens` consistent
        # (otherwise the scheduler tries to re-prefill the un-trimmed
        # tail and hangs).
        prompt_tokens_evicted, output_tokens_evicted = self._apply_trim(
            request,
            evict_start=evict_start,
            evict_end=evict_end,
            total_evicted=total_evicted,
            stride_used=stride_used,
            num_turns_evicted_after=num_turns_evicted_after,
            trim_prompt_token_ids=post_prefill_admission,
        )

        if prompt_tokens_evicted > 0:
            logger.warning(
                "[COMPACT] req=%s prompt_evicted=%d output_evicted=%d "
                "new_prompt_len=%d pos_offset=%d",
                request.request_id[:8],
                prompt_tokens_evicted,
                output_tokens_evicted,
                request.num_prompt_tokens,
                request.position_offset,
            )

        # ──── Smart position_offset bump (piecewise position fix) ────
        # Goal: ensure new K writes after this admission land at
        # logical positions ABOVE all surviving K. Achieved by bumping
        # position_offset to max(simple_bump, max_survivor_logical + 1
        # - num_computed_post). Survivors STAY in cache (no re-prefill).
        #
        # This relies on per-block logical_start being populated at
        # allocation time (Phase A of plans/piecewise_position_offset.md).
        # The GPU's position computation must apply the bumped offset
        # ONLY to physical positions >= protected_prefix_len (Phase D),
        # so sys K at logical [0..protected_prefix_len) keeps offset=0
        # and stays correct.
        #
        # Gate on prompt_tokens_evicted > 0 to apply only on admission
        # (mid-gen evicts OUTPUT; its survivor K share the same offset
        # as subsequent decode writes, so the simple bump is sufficient).
        if prompt_tokens_evicted > 0 and request.num_computed_tokens > 0:
            blocks = compaction_mgr.req_to_blocks[request.request_id]
            # Only scan blocks that ACTUALLY have K written. Blocks beyond
            # num_cached_blocks are pre-allocated empty slots for upcoming
            # prefill — they have logical_start stamped (from Phase A) but
            # no K vectors yet, so they shouldn't constrain the offset.
            num_cached_blocks = (
                request.num_computed_tokens + block_size - 1
            ) // block_size
            max_survivor_logical = -1
            for b in blocks[:num_cached_blocks]:
                if b.logical_start >= 0:
                    max_survivor_logical = max(
                        max_survivor_logical,
                        b.logical_start + block_size - 1,
                    )
            if max_survivor_logical >= 0:
                required_offset = (
                    max_survivor_logical + 1
                    - request.num_computed_tokens
                )
                if required_offset > request.position_offset:
                    logger.warning(
                        "[COMPACT-SMART] req=%s position_offset %d -> %d "
                        "(max_survivor_logical=%d, num_computed_post=%d). "
                        "Survivors kept in cache; new prefill writes "
                        "above all survivor positions.",
                        request.request_id[:8],
                        request.position_offset,
                        required_offset,
                        max_survivor_logical,
                        request.num_computed_tokens,
                    )
                    request.position_offset = required_offset

            # Re-stamp logical_start on the "future-K" blocks (those past
            # num_cached_blocks — pre-allocated for upcoming prefill but
            # not yet written). They were stamped at allocate_new_blocks
            # time with the PRE-admission position_offset. After admission
            # bumps position_offset, the worker's forward will rotate K at
            # the NEW offset, so logical_start must match — otherwise a
            # FUTURE inheritor that hits these blocks via prefix-cache
            # will compute the wrong seed offset, causing accumulated
            # Q-K rotation drift across inheritance generations.
            new_offset = request.position_offset
            for new_idx, b in enumerate(blocks[num_cached_blocks:],
                                        start=num_cached_blocks):
                if b.logical_start >= 0:
                    correct_logical_start = new_idx * block_size + new_offset
                    if b.logical_start != correct_logical_start:
                        if os.environ.get("KV_EVICTION_BUG_TRACE") == "1":
                            logger.warning(
                                "[COMPACT-RESTAMP] req=%s block_id=%d "
                                "idx=%d logical_start %d -> %d "
                                "(post-admission new-K block)",
                                request.request_id[:8], b.block_id,
                                new_idx, b.logical_start,
                                correct_logical_start,
                            )
                        b.logical_start = correct_logical_start

        # Emit the CompactionEvent NOW — after _apply_trim and the smart
        # bump have both run. `request.position_offset` is the final
        # post-bump value; trainer's `position_offset_after` mirror must
        # match this exactly (mismatch produces Q-K rotation drift across
        # the trainer/inference boundary). Prior code emitted the event
        # before mutation, relying on smart-bump never raising above the
        # simple bump — which empirically held but isn't a load-bearing
        # invariant. With cross-request offset inheritance (seeded
        # position_offset on prefix-cache hit) smart-bump may fire more
        # often, so the post-mutation emission becomes load-bearing.
        event = CompactionEvent(
            num_output_tokens_at_compaction=(
                0 if post_prefill_admission else request.num_total_generated
            ),
            tokens_evicted=total_evicted,
            position_offset_after=request.position_offset,
            num_prompt_tokens=request.num_prompt_tokens,
            evict_start=evict_start,
            evicted_token_ids=evicted_token_ids,
            last_turn_evicted=last_turn_evicted,
            num_turns_evicted_after=num_turns_evicted_after,
            kept_indices=kept_indices,
            kept_token_ids=kept_token_ids,
            new_user_fragment_len=new_user_fragment_len,
            archived_span_ids=archived_span_ids,
        )
        request.compaction_events.append(event)
        if os.environ.get("KV_EVICTION_BUG_TRACE") == "1":
            logger.warning(
                "[COMPACT-EVENT] req=%s evict_start=%d total=%d "
                "position_offset_after=%d events=%d",
                request.request_id[:8],
                evict_start, total_evicted,
                request.position_offset,
                len(request.compaction_events),
            )

        return total_evicted

    def _plan_turn_evict_range(
        self,
        request: Request,
        block_size: int,
        effective_num_computed: int | None = None,
    ) -> tuple[int, int, int, int] | None:
        """Compute the (evict_start, evict_end, last_turn_evicted, stride)
        plan for a turn-mode eviction.

        - evict_start is align_up(start_of_oldest_live_turn, block_size)
        - evict_end is align_down(end_of_last_evicted_turn, block_size)
        - both are inward snaps so that the system prompt and the
          first kept turn never lose any KV
        - returns None if the resulting range is empty (turn was too
          short for this block size)

        last_turn_evicted is the 0-indexed turn index of the LAST turn
        included in this eviction (turn 0 = first user+assistant pair
        after the system prompt), in absolute terms (not relative to
        already-evicted turns).

        `effective_num_computed` overrides the safety clamp ceiling
        (`evict_end` is clamped to `effective_num_computed // block_size *
        block_size`). Default = `request.num_computed_tokens`, which is also
        what inline admission must use: the prefill kernel has not run yet, so
        tokens beyond the cached prefix are fresh prompt tail and must not be
        trimmed before the worker sees them.
        """
        positions = request.turn_end_positions
        # positions only tracks non-evicted content:
        #   positions[0] = end of system prompt
        #   positions[1] = end of user msg of (live) turn 1
        #   positions[2] = end of (live) turn 1
        #   positions[2*k] = end of (live) turn k
        live_turns = self._num_live_completed_turns(request)
        if live_turns < self._compaction_max_turns:
            return None
        stride = min(self._compaction_eviction_turn_stride, live_turns)
        if stride <= 0:
            return None

        # First live turn starts immediately after the system prompt,
        # i.e. at positions[0]. (Even after prior evictions, positions[0]
        # always marks the end of the system prompt — invariant.)
        turn_first_start_pos = positions[0]
        # The stride-th live turn ends at positions[2*stride].
        turn_last_end_pos = positions[2 * stride]

        evict_start = (
            (turn_first_start_pos + block_size - 1) // block_size
        ) * block_size
        if self._compaction_assume_aligned_turn_boundaries:
            # Client has padded each <|im_end|> so the next message's first
            # token lands on a block boundary. align_up(positions[2*stride])
            # == start of the next kept turn, so we can safely include the
            # padding of the last evicted turn in the evict range, avoiding
            # orphan tail tokens. The num_computed_tokens clamp below is
            # still load-bearing: if the model hasn't generated past the
            # snapped-up boundary yet, we cannot evict blocks without KV.
            evict_end = (
                (turn_last_end_pos + block_size - 1) // block_size
            ) * block_size
        else:
            # Default: inward snap. Safe without padding but leaves up to
            # block_size-1 orphan tokens from the tail of the last evicted
            # turn in the kept KV region.
            evict_end = (turn_last_end_pos // block_size) * block_size

        if evict_end <= evict_start:
            # Turns too short for this block size — bail out without
            # mutating state.
            logger.warning(
                "[COMPACT] turn-mode bail: req=%s turn_first_start=%d "
                "turn_last_end=%d block_size=%d -> empty range",
                request.request_id[:8],
                turn_first_start_pos, turn_last_end_pos, block_size,
            )
            return None

        # Safety: never evict past the KV that will exist when this
        # plan is consumed. For mid-gen this is num_computed_tokens;
        # for inline-admission Phase B2 the caller passes
        # num_prompt_tokens because the prefill completing this step
        # will write KV at all prompt positions.
        clamp_ceiling = (
            request.num_computed_tokens
            if effective_num_computed is None
            else effective_num_computed
        )
        evict_end = min(
            evict_end,
            (clamp_ceiling // block_size) * block_size,
        )
        if evict_end <= evict_start:
            return None

        last_turn_evicted = request.num_turns_evicted + stride - 1
        return (evict_start, evict_end, last_turn_evicted, stride)

    def _compute_new_user_fragment_len(
        self, request: Request, evict_end: int
    ) -> int:
        """Length of the in-progress turn's tail (the new_user_fragment
        whose K/V vLLM prefills under post-eviction attention in the
        single-forward in-step admission path). Emitted on
        CompactionEvent.new_user_fragment_len so the trainer's
        segmented_forward mirror can split each admission boundary at
        the same offset.

        Returns 0 when there's no fragment (e.g. all prompt tokens are
        completed turns, or turn-mode is disabled, or no turn boundaries
        have been scanned yet). The trainer treats 0 as "no split".
        """
        if self._compaction_max_turns <= 0:
            return 0
        positions = request.turn_end_positions
        live_completed = self._num_live_completed_turns(request)
        if not positions or live_completed == 0:
            return 0
        new_user_fragment_start = positions[-1]
        if new_user_fragment_start >= request.num_prompt_tokens:
            return 0
        if new_user_fragment_start <= evict_end:
            new_user_fragment_start = evict_end
        return max(0, request.num_prompt_tokens - new_user_fragment_start)

    def _phase4_trace_id(self, request: Request) -> str:
        if (
            not self._compaction_enabled
            or self._compaction_max_turns <= 0
            or not self.cache_config.enable_prefix_caching
            or request.sampling_params is None
        ):
            return ""
        extra_args = request.sampling_params.extra_args or {}
        trace_id = extra_args.get("kve_phase4_trace_id")
        if trace_id is None:
            return ""
        return str(trace_id)

    def _phase4_call_idx(self, request: Request) -> int | None:
        if request.sampling_params is None:
            return None
        extra_args = request.sampling_params.extra_args or {}
        raw_call_idx = extra_args.get("kve_phase4_call_idx")
        if raw_call_idx is None:
            return None
        try:
            return int(raw_call_idx)
        except (TypeError, ValueError):
            return None

    def _abort_waiting_phase4_request(
        self, request: Request, reason: str
    ) -> None:
        extra_args = (
            request.sampling_params.extra_args
            if request.sampling_params is not None
            else {}
        ) or {}
        trace_id = extra_args.get("kve_phase4_trace_id", "")
        call_idx = extra_args.get("kve_phase4_call_idx", None)
        restore_span_ids = extra_args.get("kve_restore_span_ids", None)
        offload_span_ids = extra_args.get("kve_offload_span_ids", None)
        request.status = RequestStatus.FINISHED_ERROR
        request_id = request.request_id
        self._clear_phase4_pin_consumed(trace_id, request_id)
        self._pending_admission_compaction_ids.discard(request_id)
        self._free_request(request)
        logger.error(
            "[PHASE4-REQUEST-ABORTED] req=%s trace=%s call=%s "
            "restore=%s offload=%s reason=%s",
            request_id[:8],
            trace_id,
            call_idx,
            restore_span_ids,
            offload_span_ids,
            reason,
        )
        self._queue_finished_request_output(request)

    def _queue_finished_request_output(self, request: Request) -> None:
        compaction_events = (
            list(request.compaction_events)
            if request.compaction_events
            else None
        )
        self._pending_engine_core_outputs[request.client_index].append(
            EngineCoreOutput(
                request_id=request.request_id,
                new_token_ids=[],
                finish_reason=request.get_finished_reason(),
                stop_reason=request.stop_reason,
                events=request.take_events(),
                trace_headers=request.trace_headers,
                num_cached_tokens=max(0, request.num_cached_tokens),
                num_external_computed_tokens=request.num_external_computed_tokens,
                num_nans_in_logits=request.num_nans_in_logits,
                compaction_events=compaction_events,
            )
        )

    def _phase4_pin_limit(self) -> int | None:
        raw_limit = os.environ.get("KVE_PHASE4_PIN_LIMIT")
        if raw_limit is not None:
            try:
                return max(0, int(raw_limit))
            except ValueError:
                pass
        return None

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    @staticmethod
    def _env_optional_int(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value >= 0 else None

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _phase4_pin_ttl_seconds(self) -> float:
        raw_ttl = os.environ.get("KVE_PHASE4_PIN_TTL_SECONDS")
        if raw_ttl is None:
            return 1800.0
        try:
            return max(0.0, float(raw_ttl))
        except ValueError:
            return 1800.0

    def _phase4_consumed_pin_grace_seconds(self) -> float:
        raw_grace = os.environ.get("KVE_PHASE4_CONSUMED_PIN_GRACE_SECONDS")
        if raw_grace is None:
            return 2.0
        try:
            return max(0.0, float(raw_grace))
        except ValueError:
            return 2.0

    def _release_phase4_pins(
        self, trace_id: str, reason: str, *, evict_prefix: bool = False
    ) -> None:
        if not trace_id:
            return
        pin = self._phase4_pinned_blocks.get(trace_id)
        if pin is None:
            return
        if self._phase4_pin_store_in_flight(trace_id, pin):
            pin.status = "expired"
            if os.environ.get("KVE_TRACE_PHASE4_PIN") == "1":
                logger.warning(
                    "[PHASE4-PIN-EXPIRE] trace=%s reason=%s "
                    "store_in_flight=True",
                    trace_id,
                    reason,
                )
            return
        if self._phase4_pin_load_in_flight(trace_id, pin):
            pin.status = "expired"
            if os.environ.get("KVE_TRACE_PHASE4_PIN") == "1":
                logger.warning(
                    "[PHASE4-PIN-EXPIRE] trace=%s reason=%s "
                    "load_in_flight=True",
                    trace_id,
                    reason,
                )
            return

        self._phase4_pinned_blocks.pop(trace_id, None)
        if pin.store_event_id is not None:
            self._managed_context_store_events_to_submit.pop(
                pin.store_event_id, None
            )
            getattr(
                self,
                "_phase4_pin_store_event_to_trace_id",
                {},
            ).pop(pin.store_event_id, None)
        if pin.load_event_id is not None:
            self._managed_context_load_events_to_submit.pop(
                pin.load_event_id, None
            )
            getattr(
                self,
                "_phase4_pin_load_event_to_trace_id",
                {},
            ).pop(pin.load_event_id, None)

        released_blocks = self._release_phase4_pin_gpu_entries(
            pin,
            evict_prefix=evict_prefix,
        )
        self._managed_context_free_cpu_block_ids(pin.cpu_block_ids_by_group)
        if evict_prefix or os.environ.get("KVE_TRACE_PHASE4_PIN") == "1":
            logger.warning(
                "[PHASE4-PIN-RELEASE] trace=%s reason=%s blocks=%d "
                "tokens=%d req=%s call=%s consumed_by=%s status=%s "
                "evict_prefix=%s",
                trace_id,
                reason,
                released_blocks,
                pin.token_count,
                pin.request_id,
                pin.call_idx,
                pin.consumed_by_request_id,
                pin.status,
                evict_prefix,
            )

    def _release_all_phase4_pins(self, reason: str) -> None:
        for trace_id in list(self._phase4_pinned_blocks):
            self._release_phase4_pins(trace_id, reason)

    def _phase4_pin_store_in_flight(
        self, trace_id: str, pin: Phase4Pin
    ) -> bool:
        event_id = pin.store_event_id
        return bool(
            pin.status == "store_pending"
            and event_id is not None
            and event_id not in self._managed_context_store_events_to_submit
            and event_id
            in getattr(self, "_phase4_pin_store_event_to_trace_id", {})
            and self._phase4_pin_store_event_to_trace_id[event_id] == trace_id
        )

    def _phase4_pin_load_in_flight(
        self, trace_id: str, pin: Phase4Pin
    ) -> bool:
        event_id = pin.load_event_id
        return bool(
            pin.status == "load_pending"
            and event_id is not None
            and event_id not in self._managed_context_load_events_to_submit
            and event_id
            in getattr(self, "_phase4_pin_load_event_to_trace_id", {})
            and self._phase4_pin_load_event_to_trace_id[event_id] == trace_id
        )

    def _release_phase4_pin_gpu_entries(
        self,
        pin: Phase4Pin,
        *,
        evict_prefix: bool = False,
    ) -> int:
        released_blocks = 0
        for manager, blocks in pin.entries:
            if not blocks:
                continue
            manager.block_pool.free_blocks(blocks)
            if evict_prefix:
                manager.block_pool.evict_blocks(
                    {
                        block.block_id
                        for block in blocks
                        if block.block_hash is not None
                    }
                )
            released_blocks += len(blocks)
        pin.entries = []
        return released_blocks

    def _start_phase4_pin_cpu_offload(
        self,
        trace_id: str,
        pin: Phase4Pin,
        *,
        reason: str,
    ) -> str | None:
        if not self._managed_context_cpu_archive_enabled:
            return "managed-context CPU archive is disabled"
        if pin.status in ("cpu_offloaded", "store_pending", "load_pending"):
            return None
        if pin.status != "gpu_pinned":
            return f"Phase4 pin cannot be offloaded from {pin.status}"
        if not pin.entries:
            return "Phase4 pin has no GPU blocks to offload"

        limit_error = self._managed_context_cpu_store_transfer_limit_error()
        if limit_error is not None:
            return limit_error

        logical_start_by_group: list[list[int]] = []
        total_blocks = 0
        for _manager, blocks in pin.entries:
            logical_starts = [int(block.logical_start) for block in blocks]
            logical_start_by_group.append(logical_starts)
            total_blocks += len(blocks)
        cpu_block_ids = self._alloc_managed_context_cpu_blocks(total_blocks)
        if cpu_block_ids is None:
            return (
                f"Phase4 pin needs {total_blocks} CPU blocks, "
                f"available={len(self._managed_context_cpu_free_block_ids)} "
                f"max={self._managed_context_cpu_max_blocks}"
            )

        cpu_by_group: list[list[int]] = []
        offset = 0
        for logical_starts in logical_start_by_group:
            count = len(logical_starts)
            cpu_by_group.append(cpu_block_ids[offset : offset + count])
            offset += count
        if offset != len(cpu_block_ids):
            self._managed_context_free_cpu_block_ids((cpu_block_ids,))
            return "Phase4 pin has inconsistent block metadata"

        event_id = self._next_managed_context_transfer_event_id()
        pin.status = "store_pending"
        pin.cpu_block_ids_by_group = tuple(cpu_by_group)
        pin.logical_start_by_group = tuple(logical_start_by_group)
        pin.store_event_id = event_id
        self._managed_context_store_events_to_submit[event_id] = (
            ManagedContextCopyEvent(
                event_id=event_id,
                gpu_block_ids=self._managed_context_gpu_block_ids(pin.entries),
                cpu_block_ids=cpu_block_ids,
            )
        )
        self._phase4_pin_store_event_to_trace_id[event_id] = trace_id
        logger.warning(
            "[PHASE4-PIN-OFFLOAD-SUBMIT] trace=%s reason=%s event=%d "
            "blocks=%d cpu_blocks=%s",
            trace_id,
            reason,
            event_id,
            total_blocks,
            cpu_by_group,
        )
        return None

    def _complete_phase4_pin_cpu_store(self, event_id: int) -> bool:
        trace_id = getattr(
            self,
            "_phase4_pin_store_event_to_trace_id",
            {},
        ).pop(event_id, None)
        if trace_id is None:
            return False
        pin = self._phase4_pinned_blocks.get(trace_id)
        if pin is None:
            return True
        released_blocks = self._release_phase4_pin_gpu_entries(pin)
        pin.store_event_id = None
        if pin.status == "expired":
            self._managed_context_free_cpu_block_ids(pin.cpu_block_ids_by_group)
            self._phase4_pinned_blocks.pop(trace_id, None)
            status = "expired"
        else:
            pin.status = "cpu_offloaded"
            status = pin.status
        logger.warning(
            "[PHASE4-PIN-STORE-DONE] trace=%s event=%d "
            "released_gpu_blocks=%d status=%s cpu_blocks=%s",
            trace_id,
            event_id,
            released_blocks,
            status,
            pin.cpu_block_ids_by_group,
        )
        return True

    def _start_phase4_pin_cpu_load(
        self,
        trace_id: str,
        pin: Phase4Pin,
        *,
        reason: str,
    ) -> str | None:
        if pin.status == "gpu_pinned":
            return None
        if pin.status in ("store_pending", "load_pending"):
            return f"Phase4 pin is waiting for {pin.status}"
        if pin.status != "cpu_offloaded":
            return f"Phase4 pin cannot be loaded from {pin.status}"
        if not pin.cpu_block_ids_by_group or not pin.logical_start_by_group:
            return "Phase4 pin has no CPU archive metadata"

        limit_error = self._managed_context_cpu_load_transfer_limit_error()
        if limit_error is not None:
            return limit_error

        managers = self.kv_cache_manager.coordinator.single_type_managers
        if len(managers) != len(pin.cpu_block_ids_by_group):
            return "Phase4 pin has incomplete manager metadata"
        free_blocks = min(
            manager.block_pool.get_num_free_blocks() for manager in managers
        )
        if free_blocks < pin.block_count:
            return (
                "Phase4 pin load is waiting for GPU blocks: "
                f"reload_blocks={pin.block_count} free={free_blocks}"
            )

        entries: list[tuple[Any, list[Any]]] = []
        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        try:
            for idx, manager in enumerate(managers):
                group_cpu_ids = pin.cpu_block_ids_by_group[idx]
                logical_starts = pin.logical_start_by_group[idx]
                if len(group_cpu_ids) != len(logical_starts):
                    return "Phase4 pin CPU block metadata mismatch"
                blocks = manager.block_pool.get_new_blocks(len(group_cpu_ids))
                for block, logical_start in zip(blocks, logical_starts):
                    block.logical_start = int(logical_start)
                entries.append((manager, blocks))
                gpu_block_ids.extend(int(block.block_id) for block in blocks)
                cpu_block_ids.extend(int(block_id) for block_id in group_cpu_ids)
        except ValueError as exc:
            for manager, blocks in entries:
                manager.block_pool.free_blocks(reversed(blocks))
            return f"insufficient GPU blocks for Phase4 pin load: {exc}"

        event_id = self._next_managed_context_transfer_event_id()
        self._managed_context_load_events_to_submit[event_id] = (
            ManagedContextCopyEvent(
                event_id=event_id,
                gpu_block_ids=gpu_block_ids,
                cpu_block_ids=cpu_block_ids,
            )
        )
        self._phase4_pin_load_event_to_trace_id[event_id] = trace_id
        pin.status = "load_pending"
        pin.entries = entries
        pin.load_event_id = event_id
        logger.warning(
            "[PHASE4-PIN-LOAD-SUBMIT] trace=%s reason=%s event=%d "
            "blocks=%d gpu_blocks=%s cpu_blocks=%s",
            trace_id,
            reason,
            event_id,
            pin.block_count,
            gpu_block_ids,
            cpu_block_ids,
        )
        return None

    def _complete_phase4_pin_cpu_load(self, event_id: int) -> bool:
        trace_id = getattr(
            self,
            "_phase4_pin_load_event_to_trace_id",
            {},
        ).pop(event_id, None)
        if trace_id is None:
            return False
        pin = self._phase4_pinned_blocks.get(trace_id)
        if pin is None:
            return True
        pin.load_event_id = None
        if pin.status == "expired":
            released_blocks = self._release_phase4_pin_gpu_entries(pin)
            self._managed_context_free_cpu_block_ids(pin.cpu_block_ids_by_group)
            self._phase4_pinned_blocks.pop(trace_id, None)
            logger.warning(
                "[PHASE4-PIN-LOAD-DONE] trace=%s event=%d "
                "status=expired released_gpu_blocks=%d",
                trace_id,
                event_id,
                released_blocks,
            )
            return True
        pin.status = "gpu_pinned"
        self._managed_context_free_cpu_block_ids(pin.cpu_block_ids_by_group)
        pin.cpu_block_ids_by_group = ()
        pin.logical_start_by_group = ()
        logger.warning(
            "[PHASE4-PIN-LOAD-DONE] trace=%s event=%d blocks=%d",
            trace_id,
            event_id,
            pin.block_count,
        )
        return True

    def _phase4_prefix_miss_reprefill_enabled(self) -> bool:
        raw = os.environ.get("KVE_PHASE4_PREFIX_MISS_REPREFILL", "0")
        return raw.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _phase4_pin_recovery_defers(reason: str) -> bool:
        return (
            reason.endswith("submitted")
            or " is waiting for " in reason
            or " transfer limit reached" in reason
            or reason.startswith("insufficient GPU blocks for Phase4 pin load")
        )

    def _phase4_try_load_pin_for_prefix_miss(
        self,
        trace_id: str,
        expected_cached_tokens: int,
        request: Request,
        reason: str,
    ) -> str | None:
        pin = self._phase4_pinned_blocks.get(trace_id)
        if pin is None:
            return "Phase4 pin is missing"
        if pin.token_count < expected_cached_tokens:
            return (
                "Phase4 pin is too short: "
                f"tokens={pin.token_count} expected={expected_cached_tokens}"
            )
        if pin.status == "gpu_pinned":
            return "Phase4 pin GPU blocks are unavailable"
        error = self._start_phase4_pin_cpu_load(
            trace_id,
            pin,
            reason=reason,
        )
        if error is None:
            return "Phase4 pin load submitted"
        return error

    def _mark_phase4_pin_consumed(
        self, trace_id: str, request_id: str
    ) -> None:
        pin = self._phase4_pinned_blocks.get(trace_id)
        if pin is None:
            return
        pin.consumed_by_request_id = request_id
        pin.consumed_at = time.monotonic()
        if os.environ.get("KVE_TRACE_PHASE4_PIN") == "1":
            logger.warning(
                "[PHASE4-PIN-CONSUMED] trace=%s req=%s pin_req=%s "
                "pin_call=%s tokens=%d blocks=%d",
                trace_id,
                request_id[:8],
                pin.request_id,
                pin.call_idx,
                pin.token_count,
                pin.block_count,
            )

    def _clear_phase4_pin_consumed(
        self, trace_id: str, request_id: str
    ) -> None:
        pin = self._phase4_pinned_blocks.get(trace_id)
        if pin is None or pin.consumed_by_request_id != request_id:
            return
        pin.consumed_by_request_id = None
        pin.consumed_at = None
        if os.environ.get("KVE_TRACE_PHASE4_PIN") == "1":
            logger.warning(
                "[PHASE4-PIN-UNCONSUMED] trace=%s req=%s pin_req=%s "
                "pin_call=%s tokens=%d blocks=%d",
                trace_id,
                request_id[:8],
                pin.request_id,
                pin.call_idx,
                pin.token_count,
                pin.block_count,
            )

    def _phase4_has_queued_successor(
        self, trace_id: str, pin: Phase4Pin
    ) -> bool:
        return bool(self._phase4_queued_successors(trace_id, pin))

    def _phase4_queued_successors(
        self, trace_id: str, pin: Phase4Pin
    ) -> list[Request]:
        if not trace_id:
            return []
        successors: list[Request] = []
        for request_queue in (
            getattr(self, "waiting", ()),
            getattr(self, "skipped_waiting", ()),
        ):
            for request in request_queue:
                if self._phase4_pin_release_reprefill_active(request):
                    continue
                if self._phase4_trace_id(request) != trace_id:
                    continue
                expected_cached_tokens = self._phase4_expected_cached_tokens(
                    request
                )
                if (
                    expected_cached_tokens is not None
                    and expected_cached_tokens <= pin.token_count
                ):
                    successors.append(request)
        return successors

    def _phase4_pin_has_live_demand(
        self,
        trace_id: str,
        pin: Phase4Pin,
        *,
        now: float,
        grace_seconds: float,
        queued_successors: list[Request] | None = None,
    ) -> bool:
        if pin.consumed_by_request_id is None:
            # The published retained-state pin has not been consumed by the
            # next Phase4 request yet. Dropping it would force an unsafe replay.
            return True
        if pin.consumed_by_request_id in self.requests:
            return True
        consumed_at = pin.consumed_at or pin.created_at
        if grace_seconds > 0 and now - consumed_at < grace_seconds:
            return True
        if queued_successors is None:
            queued_successors = self._phase4_queued_successors(trace_id, pin)
        return bool(queued_successors)

    def _phase4_pin_is_prunable(
        self,
        trace_id: str,
        pin: Phase4Pin,
        now: float,
        ttl_seconds: float,
    ) -> bool:
        if pin.consumed_by_request_id is not None:
            # A managed-context retry can arrive immediately after the first
            # pass consumes the Phase4 pin. Keep it while the consumer is
            # still live. After that, keep a short grace window for client-side
            # retry creation, but do not drop the pin just because no successor
            # is queued at this instant. Under high concurrency, the next
            # same-trace turn can arrive much later, and this pin is the only
            # recoverable retained prefix until a successor publishes a
            # replacement pin.
            if pin.consumed_by_request_id in self.requests:
                return False
            grace_seconds = self._phase4_consumed_pin_grace_seconds()
            consumed_at = pin.consumed_at or pin.created_at
            if grace_seconds > 0 and now - consumed_at < grace_seconds:
                return False
            if self._phase4_has_queued_successor(trace_id, pin):
                return False
            if ttl_seconds <= 0:
                return False
            return now - pin.created_at >= ttl_seconds
        if ttl_seconds <= 0:
            return False
        return now - pin.created_at >= ttl_seconds

    def _compact_phase4_pin_order(self) -> None:
        if not self._phase4_pin_order:
            return
        seen: set[str] = set()
        compacted: deque[str] = deque()
        for trace_id in reversed(self._phase4_pin_order):
            if trace_id in seen or trace_id not in self._phase4_pinned_blocks:
                continue
            seen.add(trace_id)
            compacted.appendleft(trace_id)
        self._phase4_pin_order = compacted

    def _prune_phase4_pins(self) -> None:
        if not self._phase4_pinned_blocks:
            self._phase4_pin_order.clear()
            return
        now = time.monotonic()
        ttl_seconds = self._phase4_pin_ttl_seconds()
        for trace_id in list(self._phase4_pin_order):
            pin = self._phase4_pinned_blocks.get(trace_id)
            if pin is None:
                continue
            if self._phase4_pin_is_prunable(
                trace_id, pin, now, ttl_seconds
            ):
                self._release_phase4_pins(trace_id, "ttl")

        self._compact_phase4_pin_order()
        limit = self._phase4_pin_limit()
        if limit is None or len(self._phase4_pinned_blocks) <= limit:
            return

        # Count limits are advisory for Phase4. Releasing live/unconsumed pins
        # breaks the retained-KV invariant, so only already-prunable pins are
        # candidates. If the budget is still saturated, allocation/backpressure
        # must handle it instead of silently corrupting Phase4.
        while len(self._phase4_pinned_blocks) > limit:
            if not self._phase4_pin_order:
                break
            trace_id = self._phase4_pin_order.popleft()
            pin = self._phase4_pinned_blocks.get(trace_id)
            if pin is None:
                continue
            if self._phase4_pin_is_prunable(
                trace_id, pin, now, ttl_seconds
            ):
                self._release_phase4_pins(trace_id, "limit")
                continue
            self._phase4_pin_order.append(trace_id)
            logger.warning(
                "[PHASE4-PIN-LIMIT-SATURATED] pins=%d limit=%d; "
                "keeping live pins to preserve retained KV",
                len(self._phase4_pinned_blocks),
                limit,
            )
            break

    def _release_phase4_pressure_pin(self, reason: str) -> bool:
        if not self._phase4_pinned_blocks:
            return False
        self._compact_phase4_pin_order()
        now = time.monotonic()
        grace_seconds = self._phase4_consumed_pin_grace_seconds()
        for trace_id in list(self._phase4_pin_order):
            pin = self._phase4_pinned_blocks.get(trace_id)
            if pin is None:
                continue
            queued_successors = self._phase4_queued_successors(trace_id, pin)
            live_demand = self._phase4_pin_has_live_demand(
                trace_id,
                pin,
                now=now,
                grace_seconds=grace_seconds,
                queued_successors=queued_successors,
            )
            if not live_demand and reason != "scheduler-stall-pressure":
                continue
            if pin.status != "gpu_pinned":
                continue
            offload_error = self._start_phase4_pin_cpu_offload(
                trace_id,
                pin,
                reason=reason,
            )
            if offload_error is not None:
                logger.warning(
                    "[PHASE4-PIN-OFFLOAD-DEFER] trace=%s reason=%s "
                    "queued_successors=%d live_demand=%s status=%s error=%s",
                    trace_id,
                    reason,
                    len(queued_successors),
                    live_demand,
                    pin.status,
                    offload_error,
                )
                continue
            if reason == "scheduler-stall-pressure":
                return True
            # The D2H copy will free GPU blocks only after the transfer
            # completion is observed, so allocation-pressure callers must
            # not immediately retry slot allocation.
            return False
        return False

    def _maybe_release_phase4_stall_pressure_pin(
        self, *, total_num_scheduled_tokens: int
    ) -> bool:
        if (
            total_num_scheduled_tokens != 0
            or self.running
            or not (self.waiting or self.skipped_waiting)
        ):
            return False
        before_pools = self._kve_gpu_block_pool_diag_summary()
        before_managed = self._kve_managed_context_diag_summary()
        release_limit = max(
            1,
            self._env_int("KVE_PHASE4_STALL_PRESSURE_RELEASE_MAX", 8),
        )
        released_hot_gpu_blocks = 0
        released_phase4_pins = 0
        for _ in range(release_limit):
            released_hot = self._managed_context_release_hot_gpu_pressure(
                "scheduler-stall-pressure"
            )
            if released_hot:
                released_hot_gpu_blocks += released_hot
                continue
            if self._release_phase4_pressure_pin("scheduler-stall-pressure"):
                released_phase4_pins += 1
                continue
            break
        if not released_hot_gpu_blocks and not released_phase4_pins:
            return False
        logger.warning(
            "[PHASE4-STALL-PRESSURE-RELEASE] hot_gpu_blocks=%d "
            "phase4_pins=%d limit=%d waiting=%d skipped=%d "
            "pools_before=%s managed_before=%s pools_after=%s "
            "managed_after=%s",
            released_hot_gpu_blocks,
            released_phase4_pins,
            release_limit,
            len(self.waiting),
            len(self.skipped_waiting),
            before_pools,
            before_managed,
            self._kve_gpu_block_pool_diag_summary(),
            self._kve_managed_context_diag_summary(),
        )
        return True

    def _pin_phase4_request_blocks(self, request: Request) -> None:
        trace_id = self._phase4_trace_id(request)
        if not trace_id or request.status in (
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_ERROR,
            RequestStatus.FINISHED_IGNORED,
        ):
            return

        entries: list[tuple[Any, list[Any]]] = []
        total_blocks = 0
        for manager in self.kv_cache_manager.coordinator.single_type_managers:
            req_blocks = manager.req_to_blocks.get(request.request_id)
            if not req_blocks:
                continue
            num_full_blocks = min(
                len(req_blocks),
                request.num_tokens // manager.block_size,
            )
            blocks = [
                block
                for block in req_blocks[:num_full_blocks]
                if not block.is_null
            ]
            if not blocks:
                continue
            manager.block_pool.touch(blocks)
            entries.append((manager, blocks))
            total_blocks += len(blocks)

        if not entries:
            return

        # The successor has finished successfully; replace the previous
        # retained-state pin only now. This keeps call N's pin available for
        # call N+1 preemption/retry until call N+1 can publish its own state.
        self._release_phase4_pins(trace_id, "replace")
        self._phase4_pinned_blocks[trace_id] = Phase4Pin(
            entries=entries,
            token_count=request.num_tokens,
            block_count=total_blocks,
            request_id=request.request_id[:8],
            call_idx=self._phase4_call_idx(request),
            created_at=time.monotonic(),
        )
        self._phase4_pin_order.append(trace_id)
        self._prune_phase4_pins()
        if os.environ.get("KVE_TRACE_PHASE4_PIN") == "1":
            logger.warning(
                "[PHASE4-PIN] trace=%s req=%s tokens=%d blocks=%d",
                trace_id,
                request.request_id[:8],
                request.num_tokens,
                total_blocks,
            )

    def _managed_context_trace_id(self, request: Request) -> str:
        if not self._managed_context_enabled:
            return ""
        return self._phase4_trace_id(request)

    def _next_managed_context_transfer_event_id(self) -> int:
        event_id = self._managed_context_next_transfer_event_id
        self._managed_context_next_transfer_event_id += 1
        return event_id

    def _managed_context_gpu_block_ids(
        self, entries: list[tuple[Any, list[Any]]]
    ) -> list[int]:
        return [
            int(block.block_id)
            for _manager, blocks in entries
            for block in blocks
        ]

    def _managed_context_free_cpu_block_ids(
        self, cpu_block_ids_by_group: tuple[list[int], ...]
    ) -> None:
        for cpu_ids in cpu_block_ids_by_group:
            self._managed_context_cpu_free_block_ids.extend(cpu_ids)

    def _managed_context_free_span_cpu_blocks(
        self, span: ManagedContextSpan
    ) -> None:
        if not span.cpu_block_ids_by_group:
            return
        self._managed_context_free_cpu_block_ids(span.cpu_block_ids_by_group)
        span.cpu_block_ids_by_group = ()

    def _release_managed_context_entries(
        self, entries: list[tuple[Any, list[Any]]]
    ) -> int:
        released_blocks = 0
        for manager, blocks in entries:
            if blocks:
                manager.block_pool.free_blocks(blocks)
                released_blocks += len(blocks)
        return released_blocks

    def _request_kv_swap_enabled(self) -> bool:
        return (
            os.environ.get("KVE_REQUEST_KV_SWAP", "0")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
            and self._managed_context_cpu_archive_enabled
            and not self._request_kv_swap_suspended
        )

    def _request_kv_swap_strict_preempt_enabled(self) -> bool:
        raw = os.environ.get("KVE_REQUEST_KV_SWAP_STRICT_PREEMPT", "1")
        return self._request_kv_swap_enabled() and raw.strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    def _request_kv_swap_resident_first_enabled(self) -> bool:
        raw = os.environ.get("KVE_REQUEST_KV_SWAP_RESIDENT_FIRST", "1")
        return self._request_kv_swap_enabled() and raw.strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    def _request_kv_swap_reload_starvation_seconds(self) -> float:
        value = self._env_float("KVE_REQUEST_KV_SWAP_RELOAD_STARVATION_SECONDS", 30.0)
        return max(0.0, value)

    def _request_kv_swap_usage_watermark(
        self,
        name: str,
    ) -> float | None:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        if not 0.0 < value < 1.0:
            return None
        return value

    def _request_kv_swap_reload_target_usage(self) -> float | None:
        return self._request_kv_swap_usage_watermark(
            "KVE_REQUEST_KV_SWAP_RELOAD_TARGET_USAGE"
        )

    def _request_kv_swap_offload_start_usage(self) -> float | None:
        return self._request_kv_swap_usage_watermark(
            "KVE_REQUEST_KV_SWAP_OFFLOAD_START_USAGE"
        )

    def _request_kv_swap_offload_stop_usage(self) -> float | None:
        explicit_stop = self._request_kv_swap_usage_watermark(
            "KVE_REQUEST_KV_SWAP_OFFLOAD_STOP_USAGE"
        )
        if explicit_stop is not None:
            return explicit_stop
        start = self._request_kv_swap_offload_start_usage()
        if start is None:
            return None
        return max(0.0, start - 0.10)

    def _request_kv_swap_pressure_largest_first_enabled(self) -> bool:
        raw = os.environ.get("KVE_REQUEST_KV_SWAP_PRESSURE_LARGEST_FIRST")
        if raw is None:
            return self._request_kv_swap_offload_start_usage() is not None
        return raw.strip().lower() not in ("0", "false", "no", "off")

    @staticmethod
    def _request_kv_swap_usage_free_blocks(
        total_blocks: int,
        usage: float,
    ) -> int:
        free_blocks = total_blocks * (1.0 - usage)
        return max(0, int(free_blocks + 0.999999))

    def _request_kv_swap_gpu_block_pool_stats(self) -> tuple[int, int]:
        managers = self.kv_cache_manager.coordinator.single_type_managers
        if not managers:
            return 0, 0
        total_blocks_by_group: list[int] = []
        free_blocks_by_group: list[int] = []
        for manager in managers:
            block_pool = manager.block_pool
            free_blocks = int(block_pool.get_num_free_blocks())
            total_blocks = int(getattr(block_pool, "num_gpu_blocks", 0) or 0)
            if total_blocks <= 0:
                total_blocks = free_blocks
            total_blocks_by_group.append(total_blocks)
            free_blocks_by_group.append(free_blocks)
        return min(total_blocks_by_group), min(free_blocks_by_group)

    def _request_kv_swap_preempt_error_defers(self, error: str) -> bool:
        if not self._request_kv_swap_strict_preempt_enabled():
            return False
        exactness_or_backpressure_errors = {
            "request KV swap is already active",
            "request KV swap store queue is full",
            "request has active managed-context hidden KV",
            "request has deferred managed-context hidden KV",
            "request has managed-context load in flight",
        }
        return (
            error in exactness_or_backpressure_errors
            or (
                error.startswith("request KV swap needs ")
                and " CPU blocks, " in error
            )
            or error.startswith("managed-context CPU store transfer limit reached")
        )

    def _request_kv_swap_max_pending_stores(self) -> int:
        return max(1, self._env_int("KVE_REQUEST_KV_SWAP_MAX_PENDING_STORES", 1))

    def _request_kv_swap_max_pending_loads(self) -> int:
        return max(1, self._env_int("KVE_REQUEST_KV_SWAP_MAX_PENDING_LOADS", 1))

    def _request_kv_swap_gpu_headroom_blocks(self) -> int:
        explicit_headroom = self._env_optional_int(
            "KVE_REQUEST_KV_SWAP_GPU_HEADROOM_BLOCKS"
        )
        if explicit_headroom is not None:
            return explicit_headroom
        return max(
            0, self._env_int("KVE_REQUEST_KV_SWAP_MIN_FREE_GPU_BLOCKS", 0)
        )

    def _request_kv_swap_gpu_pressure_blocks(self) -> int:
        start_usage = self._request_kv_swap_offload_start_usage()
        if start_usage is not None:
            total_blocks, _free_blocks = self._request_kv_swap_gpu_block_pool_stats()
            if total_blocks > 0:
                return max(
                    1,
                    self._request_kv_swap_usage_free_blocks(
                        total_blocks,
                        start_usage,
                    ),
                )
        explicit_pressure = self._env_optional_int(
            "KVE_REQUEST_KV_SWAP_GPU_PRESSURE_BLOCKS"
        )
        if explicit_pressure is not None:
            return explicit_pressure
        return self._request_kv_swap_gpu_headroom_blocks()

    def _request_kv_swap_pending_stores(self) -> int:
        return sum(
            1
            for swap in self._request_kv_swaps.values()
            if swap.status == "store_pending"
        )

    def _request_kv_swap_pending_loads(self) -> int:
        return sum(
            1
            for swap in self._request_kv_swaps.values()
            if swap.status == "load_pending"
        )

    def _request_kv_swap_pending_store_blocks(self) -> int:
        return sum(
            int(getattr(swap, "kv_block_count", 0) or 0)
            for swap in self._request_kv_swaps.values()
            if swap.status == "store_pending"
        )

    def _request_kv_swap_remove_ready(self, request_id: str) -> None:
        try:
            self._request_kv_swap_ready_queue.remove(request_id)
        except ValueError:
            pass

    def _request_kv_swap_ready_head(self) -> str | None:
        while self._request_kv_swap_ready_queue:
            request_id = self._request_kv_swap_ready_queue[0]
            swap = self._request_kv_swaps.get(request_id)
            request = self.requests.get(request_id)
            is_finished = (
                request.is_finished()
                if request is not None and hasattr(request, "is_finished")
                else False
            )
            if (
                swap is not None
                and swap.status == "swapped"
                and request is not None
                and not is_finished
            ):
                return request_id
            self._request_kv_swap_ready_queue.popleft()
        return None

    @staticmethod
    def _request_kv_swap_gpu_wait_error(error: str | None) -> bool:
        return bool(
            error
            and error.startswith(
                "request KV swap load is waiting for GPU blocks:"
            )
        )

    def _request_kv_swap_should_block_waiting_admission(
        self, request: Request
    ) -> bool:
        if not self._request_kv_swap_enabled() or not self.running:
            return False
        swap = self._request_kv_swaps.get(request.request_id)
        if swap is None or swap.status != "swapped":
            return False
        if self._request_kv_swap_ready_head() != request.request_id:
            return False
        return self._request_kv_swap_gpu_wait_error(swap.last_error)

    def _request_kv_swap_load_is_admissible(
        self,
        request: Request,
        swap: RequestKVSwap,
        *,
        token_budget: int | None = None,
    ) -> bool:
        if swap.status != "swapped":
            return False
        if self._request_kv_swap_pending_loads() >= (
            self._request_kv_swap_max_pending_loads()
        ):
            return False
        if self._managed_context_cpu_load_transfer_limit_error() is not None:
            return False

        managers = self.kv_cache_manager.coordinator.single_type_managers
        if len(managers) != len(getattr(swap, "logical_start_by_group", ())):
            return False
        if token_budget is None:
            token_budget = int(
                getattr(
                    self,
                    "max_num_scheduled_tokens",
                    getattr(request, "num_tokens", 0),
                )
                or 0
            )
        extra_required_gpu_blocks = (
            self._request_kv_swap_next_allocation_block_demand(
                request,
                swap,
                token_budget=token_budget,
            )
        )
        return (
            self._request_kv_swap_load_capacity_error(
                swap,
                extra_required_gpu_blocks=extra_required_gpu_blocks,
            )
            is None
        )

    def _request_kv_swap_load_capacity_error(
        self,
        swap: RequestKVSwap,
        *,
        extra_required_gpu_blocks: int,
    ) -> str | None:
        min_free = self._request_kv_swap_gpu_headroom_blocks()
        total_blocks, free_blocks = self._request_kv_swap_gpu_block_pool_stats()
        required_blocks = (
            int(getattr(swap, "kv_block_count", 0) or 0)
            + max(0, extra_required_gpu_blocks)
            + min_free
        )
        if free_blocks < required_blocks:
            return (
                "request KV swap load is waiting for GPU blocks: "
                f"reload_blocks={swap.kv_block_count} free={free_blocks} "
                f"extra_blocks={extra_required_gpu_blocks} "
                f"min_free={min_free} required={required_blocks}"
            )

        target_usage = self._request_kv_swap_reload_target_usage()
        if target_usage is None or total_blocks <= 0:
            return None

        reload_blocks = int(getattr(swap, "kv_block_count", 0) or 0)
        extra_blocks = max(0, extra_required_gpu_blocks)
        used_blocks = max(0, total_blocks - free_blocks)
        projected_used_blocks = used_blocks + reload_blocks + extra_blocks
        target_used_blocks = int(total_blocks * target_usage)
        if projected_used_blocks <= target_used_blocks:
            return None

        return (
            "request KV swap load is parked by GPU watermark: "
            f"reload_blocks={reload_blocks} used={used_blocks} "
            f"free={free_blocks} total={total_blocks} "
            f"extra_blocks={extra_blocks} min_free={min_free} "
            f"target_usage={target_usage:.3f} "
            f"target_used={target_used_blocks} "
            f"projected_used={projected_used_blocks}"
        )

    def _request_kv_swap_should_park_waiting_request(
        self,
        request: Request,
        *,
        token_budget: int | None = None,
    ) -> bool:
        if not self._request_kv_swap_resident_first_enabled():
            return False
        if request.status != RequestStatus.WAITING_FOR_REMOTE_KVS:
            return False
        swap = self._request_kv_swaps.get(request.request_id)
        if swap is None:
            return False
        if swap.status == "load_pending":
            return (
                request.request_id
                not in self._request_kv_swap_finished_load_req_ids
            )
        if swap.status == "store_pending":
            return bool(self.running or self.waiting)
        if swap.status != "swapped":
            return False

        # If no GPU-resident or ordinary waiting work can make progress, this
        # parked request is the work. Let the normal load path try and report
        # the exact admission reason.
        if not self.running and not self.waiting:
            return False

        age = time.monotonic() - swap.created_at
        if age >= self._request_kv_swap_reload_starvation_seconds() and (
            self._request_kv_swap_load_is_admissible(
                request,
                swap,
                token_budget=token_budget,
            )
        ):
            return False

        if token_budget is not None and token_budget <= 0:
            return True
        swap.last_error = (
            "request KV swap load parked behind GPU-resident work: "
            f"age={age:.3f}s"
        )
        return True

    def _request_kv_swap_should_prefer_waiting_queue(self) -> bool:
        if (
            not self._request_kv_swap_resident_first_enabled()
            or not self.waiting
            or not self.skipped_waiting
        ):
            return False
        request = self.skipped_waiting.peek_request()
        if request.status != RequestStatus.WAITING_FOR_REMOTE_KVS:
            return False
        swap = self._request_kv_swaps.get(request.request_id)
        if swap is None:
            return False
        if swap.status == "load_pending":
            return (
                request.request_id
                not in self._request_kv_swap_finished_load_req_ids
            )
        if swap.status == "store_pending":
            return True
        if swap.status != "swapped":
            return False
        age = time.monotonic() - swap.created_at
        return age < self._request_kv_swap_reload_starvation_seconds()

    def _request_kv_swap_prioritize_ready_waiting(self) -> None:
        if (
            not self._request_kv_swap_enabled()
            or self.policy != SchedulingPolicy.FCFS
        ):
            return
        ready_request_id = self._request_kv_swap_ready_head()
        if ready_request_id is None:
            return

        for queue in (self.skipped_waiting, self.waiting):
            for request in tuple(queue):
                if request.request_id != ready_request_id:
                    continue
                if (
                    queue is self.skipped_waiting
                    and self.skipped_waiting
                    and self.skipped_waiting.peek_request() is request
                ):
                    return
                queue.remove_request(request)
                self.skipped_waiting.prepend_request(request)
                return

    def _request_kv_swap_gpu_block_count(self, request_id: str) -> int:
        total_blocks = 0
        for manager in self.kv_cache_manager.coordinator.single_type_managers:
            total_blocks += len(manager.req_to_blocks.get(request_id, []))
        return total_blocks

    def _request_kv_swap_pressure_candidate_error(
        self,
        request: Request,
        protected_request_ids: set[str],
    ) -> str | None:
        request_id = request.request_id
        if request_id in protected_request_ids:
            return "request is protected in this scheduler step"
        if request.status != RequestStatus.RUNNING:
            return f"request is not running: {request.status}"
        if request.padding_pending or request.num_output_placeholders:
            return "request has pending padding/output placeholders"
        if request_id in self._request_kv_swaps:
            return "request KV swap is already active"
        if request_id in self._managed_context_active_restores:
            return "request has active managed-context hidden KV"
        if request_id in self._managed_context_deferred_restores:
            return "request has deferred managed-context hidden KV"
        if request_id in self._managed_context_pending_loads:
            return "request has managed-context load in flight"
        if self._request_kv_swap_gpu_block_count(request_id) <= 0:
            return "request has no GPU KV blocks"
        return None

    def _request_kv_swap_pressure_candidates(
        self,
        protected_request_ids: set[str],
    ) -> list[Request]:
        candidates = [
            request
            for request in self.running
            if self._request_kv_swap_pressure_candidate_error(
                request,
                protected_request_ids,
            )
            is None
        ]
        if self.policy == SchedulingPolicy.PRIORITY:
            candidates.sort(
                key=lambda request: (request.priority, request.arrival_time),
                reverse=True,
            )
        else:
            candidates = list(reversed(candidates))
        if self._request_kv_swap_pressure_largest_first_enabled():
            candidates.sort(
                key=lambda request: self._request_kv_swap_gpu_block_count(
                    request.request_id
                ),
                reverse=True,
            )
            return candidates
        return candidates

    @staticmethod
    def _request_kv_swap_pressure_global_error(error: str | None) -> bool:
        if error is None:
            return False
        return (
            error == "request KV swap store queue is full"
            or error.startswith("request KV swap needs ")
            or error.startswith("managed-context CPU store transfer limit reached")
        )

    def _preempt_request_for_kv_swap_pressure(
        self,
        timestamp: float,
        *,
        reason: str,
        protected_request_ids: set[str],
    ) -> Request | None:
        if not self._request_kv_swap_enabled():
            return None
        pressure_blocks = self._request_kv_swap_gpu_pressure_blocks()
        if pressure_blocks <= 0:
            return None
        total_blocks, free_blocks = self._request_kv_swap_gpu_block_pool_stats()
        if free_blocks >= pressure_blocks:
            return None
        stop_usage = self._request_kv_swap_offload_stop_usage()
        if stop_usage is not None and total_blocks > 0:
            target_free_blocks = self._request_kv_swap_usage_free_blocks(
                total_blocks,
                stop_usage,
            )
            projected_free_blocks = (
                free_blocks + self._request_kv_swap_pending_store_blocks()
            )
            if projected_free_blocks >= target_free_blocks:
                return None

        last_error: str | None = None
        for candidate in self._request_kv_swap_pressure_candidates(
            protected_request_ids
        ):
            try:
                candidate_index = self.running.index(candidate)
            except ValueError:
                continue
            preempted_req = self.running.pop(candidate_index)
            result = self._preempt_request_for_kv_swap(
                preempted_req,
                timestamp,
                reason=reason,
            )
            if result.kind == _PREEMPTION_ASYNC_PENDING:
                logger.warning(
                    "[REQUEST-KV-SWAP-PRESSURE] req=%s reason=%s "
                    "free=%d pressure=%d blocks=%d",
                    preempted_req.request_id[:8],
                    reason,
                    free_blocks,
                    pressure_blocks,
                    self._request_kv_swaps[
                        preempted_req.request_id
                    ].kv_block_count,
                )
                return preempted_req

            self.running.insert(candidate_index, preempted_req)
            last_error = result.error
            if self._request_kv_swap_pressure_global_error(result.error):
                break

        if (
            last_error is not None
            and os.environ.get("KVE_TRACE_REQUEST_KV_SWAP") == "1"
        ):
            logger.warning(
                "[REQUEST-KV-SWAP-PRESSURE-SKIP] reason=%s free=%d "
                "pressure=%d error=%s",
                reason,
                free_blocks,
                pressure_blocks,
                last_error,
            )
        return None

    def _request_kv_swap_next_allocation_block_demand(
        self,
        request: Request,
        swap: RequestKVSwap,
        *,
        token_budget: int,
    ) -> int:
        if token_budget <= 0:
            return 0
        managers = self.kv_cache_manager.coordinator.single_type_managers
        if not managers:
            return 0
        block_size = max(1, int(getattr(managers[0], "block_size", 1)))
        computed_tokens = max(0, swap.num_computed_tokens)
        tokens_with_spec = getattr(request, "num_tokens_with_spec", None)
        if tokens_with_spec is None:
            tokens_with_spec = getattr(request, "num_tokens", 0) + len(
                getattr(request, "spec_token_ids", []) or []
            )
        num_new_tokens = (
            int(tokens_with_spec)
            + int(getattr(request, "num_output_placeholders", 0))
            - computed_tokens
        )
        if num_new_tokens <= 0:
            return 0
        long_prefill_threshold = int(
            getattr(
                getattr(self, "scheduler_config", None),
                "long_prefill_token_threshold",
                0,
            )
            or 0
        )
        if 0 < long_prefill_threshold < num_new_tokens:
            num_new_tokens = long_prefill_threshold
        num_new_tokens = min(num_new_tokens, token_budget)
        if num_new_tokens <= 0:
            return 0

        max_sched_len = int(getattr(self, "max_model_len", 0) or 0)
        if max_sched_len > 0 and not getattr(request, "padding_pending", False):
            max_sched_len -= 1
        tokens_needing_slots = (
            computed_tokens
            + num_new_tokens
            + max(0, int(getattr(self, "num_lookahead_tokens", 0) or 0))
        )
        if max_sched_len > 0:
            tokens_needing_slots = min(tokens_needing_slots, max_sched_len)
        required_blocks = (tokens_needing_slots + block_size - 1) // block_size
        return max(0, required_blocks - max(0, swap.kv_block_count))

    def _release_request_kv_swap_entries(
        self, request_id: str, entries: list[tuple[Any, list[Any]]]
    ) -> int:
        released_blocks = 0
        for manager, blocks in entries:
            if hasattr(manager, "req_to_blocks"):
                manager.req_to_blocks.pop(request_id, None)
            if hasattr(manager, "num_cached_block"):
                manager.num_cached_block.pop(request_id, None)
            if blocks:
                manager.block_pool.free_blocks(reversed(blocks))
                released_blocks += len(blocks)
        return released_blocks

    def _start_request_kv_swap_out(
        self, request: Request, reason: str
    ) -> str | None:
        if not self._request_kv_swap_enabled():
            return "request KV swap is disabled"
        request_id = request.request_id
        if request_id in self._request_kv_swaps:
            return "request KV swap is already active"
        if request.padding_pending or request.num_output_placeholders:
            return "request has pending padding/output placeholders"
        if request_id in self._managed_context_active_restores:
            return "request has active managed-context hidden KV"
        if request_id in self._managed_context_deferred_restores:
            return "request has deferred managed-context hidden KV"
        if request_id in self._managed_context_pending_loads:
            return "request has managed-context load in flight"
        if self._request_kv_swap_pending_stores() >= (
            self._request_kv_swap_max_pending_stores()
        ):
            return "request KV swap store queue is full"
        limit_error = self._managed_context_cpu_store_transfer_limit_error()
        if limit_error is not None:
            return limit_error

        managers = self.kv_cache_manager.coordinator.single_type_managers
        if len(managers) != 1:
            return "request KV swap currently supports one KV cache group"

        entries: list[tuple[Any, list[Any]]] = []
        logical_start_by_group: list[list[int]] = []
        total_blocks = 0
        for manager in managers:
            blocks = list(manager.req_to_blocks.get(request_id, []))
            if not blocks:
                return "request has no GPU KV blocks"
            entries.append((manager, blocks))
            logical_start_by_group.append(
                [int(block.logical_start) for block in blocks]
            )
            total_blocks += len(blocks)

        cpu_block_ids = self._alloc_managed_context_cpu_blocks(total_blocks)
        if cpu_block_ids is None:
            return (
                f"request KV swap needs {total_blocks} CPU blocks, "
                f"available={len(self._managed_context_cpu_free_block_ids)} "
                f"max={self._managed_context_cpu_max_blocks}"
            )

        cpu_by_group: list[list[int]] = []
        offset = 0
        for starts in logical_start_by_group:
            count = len(starts)
            cpu_by_group.append(cpu_block_ids[offset : offset + count])
            offset += count
        if offset != len(cpu_block_ids):
            self._managed_context_free_cpu_block_ids((cpu_block_ids,))
            return "request KV swap has inconsistent block metadata"

        event_id = self._next_managed_context_transfer_event_id()
        swap = RequestKVSwap(
            request_id=request_id,
            cpu_block_ids_by_group=tuple(cpu_by_group),
            logical_start_by_group=tuple(logical_start_by_group),
            kv_block_count=total_blocks,
            num_computed_tokens=request.num_computed_tokens,
            position_offset=request.position_offset,
            status="store_pending",
            created_at=time.monotonic(),
            entries=entries,
            store_event_id=event_id,
        )
        self._request_kv_swaps[request_id] = swap
        self._request_kv_swap_store_event_to_request_id[event_id] = request_id
        self._managed_context_store_events_to_submit[event_id] = (
            ManagedContextCopyEvent(
                event_id=event_id,
                gpu_block_ids=self._managed_context_gpu_block_ids(entries),
                cpu_block_ids=cpu_block_ids,
            )
        )
        if os.environ.get("KVE_TRACE_REQUEST_KV_SWAP") == "1":
            logger.warning(
                "[REQUEST-KV-SWAP-OUT-SUBMIT] req=%s event=%d reason=%s "
                "blocks=%d computed=%d position_offset=%d cpu_blocks=%s",
                request_id[:8],
                event_id,
                reason,
                total_blocks,
                request.num_computed_tokens,
                request.position_offset,
                cpu_by_group,
            )
        return None

    def _complete_request_kv_swap_store(self, event_id: int) -> None:
        request_id = self._request_kv_swap_store_event_to_request_id.pop(
            event_id, None
        )
        if request_id is None:
            return
        swap = self._request_kv_swaps.get(request_id)
        if swap is None or swap.status not in ("store_pending", "expired"):
            return
        released_blocks = self._release_request_kv_swap_entries(
            request_id, swap.entries
        )
        swap.entries = []
        swap.store_event_id = None
        if swap.status == "expired":
            self._managed_context_free_cpu_block_ids(swap.cpu_block_ids_by_group)
            self._request_kv_swaps.pop(request_id, None)
            self._request_kv_swap_remove_ready(request_id)
            self.requests.pop(request_id, None)
            return
        swap.status = "swapped"
        self._request_kv_swap_remove_ready(request_id)
        self._request_kv_swap_ready_queue.append(request_id)
        if os.environ.get("KVE_TRACE_REQUEST_KV_SWAP") == "1":
            logger.warning(
                "[REQUEST-KV-SWAP-STORE-DONE] req=%s event=%d "
                "released_gpu_blocks=%d cpu_blocks=%s",
                request_id[:8],
                event_id,
                released_blocks,
                swap.cpu_block_ids_by_group,
            )

    def _start_request_kv_swap_load(
        self,
        request: Request,
        *,
        extra_required_gpu_blocks: int = 0,
    ) -> str | None:
        request_id = request.request_id
        swap = self._request_kv_swaps.get(request_id)
        if swap is None:
            return "request KV swap state is missing"
        if swap.status != "swapped":
            return f"request KV swap is not ready to load: {swap.status}"
        if self._request_kv_swap_pending_loads() >= (
            self._request_kv_swap_max_pending_loads()
        ):
            return "request KV swap load queue is full"
        limit_error = self._managed_context_cpu_load_transfer_limit_error()
        if limit_error is not None:
            return limit_error

        managers = self.kv_cache_manager.coordinator.single_type_managers
        if len(managers) != len(swap.logical_start_by_group):
            return "request KV swap has incomplete manager metadata"

        capacity_error = self._request_kv_swap_load_capacity_error(
            swap,
            extra_required_gpu_blocks=extra_required_gpu_blocks,
        )
        if capacity_error is not None:
            return capacity_error
        min_free = self._request_kv_swap_gpu_headroom_blocks()

        entries: list[tuple[Any, list[Any]]] = []
        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        try:
            for idx, manager in enumerate(managers):
                logical_starts = swap.logical_start_by_group[idx]
                cpu_ids = swap.cpu_block_ids_by_group[idx]
                if len(logical_starts) != len(cpu_ids):
                    return "request KV swap block metadata mismatch"
                blocks = manager.block_pool.get_new_blocks(len(cpu_ids))
                for block, logical_start in zip(blocks, logical_starts):
                    block.logical_start = int(logical_start)
                manager.req_to_blocks[request_id] = blocks
                manager.num_cached_block.pop(request_id, None)
                entries.append((manager, blocks))
                gpu_block_ids.extend(int(block.block_id) for block in blocks)
                cpu_block_ids.extend(int(block_id) for block_id in cpu_ids)
        except ValueError as exc:
            for manager, blocks in entries:
                manager.req_to_blocks.pop(request_id, None)
                manager.block_pool.free_blocks(reversed(blocks))
            return f"insufficient GPU blocks for request KV swap load: {exc}"

        event_id = self._next_managed_context_transfer_event_id()
        self._managed_context_load_events_to_submit[event_id] = (
            ManagedContextCopyEvent(
                event_id=event_id,
                gpu_block_ids=gpu_block_ids,
                cpu_block_ids=cpu_block_ids,
            )
        )
        self._request_kv_swap_load_event_to_request_id[event_id] = request_id
        swap.status = "load_pending"
        swap.entries = entries
        swap.load_event_id = event_id
        self._request_kv_swap_remove_ready(request_id)
        if os.environ.get("KVE_TRACE_REQUEST_KV_SWAP") == "1":
            logger.warning(
                "[REQUEST-KV-SWAP-IN-SUBMIT] req=%s event=%d blocks=%d "
                "extra_blocks=%d min_free=%d gpu_blocks=%s cpu_blocks=%s",
                request_id[:8],
                event_id,
                swap.kv_block_count,
                extra_required_gpu_blocks,
                min_free,
                gpu_block_ids,
                cpu_block_ids,
            )
        return None

    def _complete_request_kv_swap_load(self, request: Request) -> None:
        request_id = request.request_id
        swap = self._request_kv_swaps.pop(request_id, None)
        if swap is None:
            return
        self._request_kv_swap_finished_load_req_ids.discard(request_id)
        if swap.load_event_id is not None:
            self._request_kv_swap_load_event_to_request_id.pop(
                swap.load_event_id, None
            )
        self._managed_context_free_cpu_block_ids(swap.cpu_block_ids_by_group)
        request.num_computed_tokens = swap.num_computed_tokens
        request.position_offset = swap.position_offset
        request.num_external_computed_tokens = 0
        request.num_cached_tokens = request.num_computed_tokens
        request.needs_rebuild = True
        if os.environ.get("KVE_TRACE_REQUEST_KV_SWAP") == "1":
            logger.warning(
                "[REQUEST-KV-SWAP-LOAD-DONE] req=%s blocks=%d "
                "computed=%d position_offset=%d",
                request_id[:8],
                swap.kv_block_count,
                request.num_computed_tokens,
                request.position_offset,
            )

    def _release_request_kv_swap(
        self,
        request_id: str,
        reason: str,
        *,
        release_gpu_entries: bool,
    ) -> bool:
        swap = self._request_kv_swaps.get(request_id)
        if swap is None:
            return True

        store_in_flight = (
            swap.status == "store_pending"
            and swap.store_event_id is not None
            and swap.store_event_id
            not in self._managed_context_store_events_to_submit
        )
        load_in_flight = (
            swap.status == "load_pending"
            and swap.load_event_id is not None
            and swap.load_event_id
            not in self._managed_context_load_events_to_submit
            and request_id not in self._request_kv_swap_finished_load_req_ids
        )
        if store_in_flight or load_in_flight:
            swap.status = "expired"
            self._request_kv_swap_remove_ready(request_id)
            if os.environ.get("KVE_TRACE_REQUEST_KV_SWAP") == "1":
                logger.warning(
                    "[REQUEST-KV-SWAP-EXPIRE] req=%s reason=%s "
                    "store_in_flight=%s load_in_flight=%s",
                    request_id[:8],
                    reason,
                    store_in_flight,
                    load_in_flight,
                )
            return False

        self._request_kv_swaps.pop(request_id, None)
        self._request_kv_swap_remove_ready(request_id)
        if swap.store_event_id is not None:
            self._managed_context_store_events_to_submit.pop(
                swap.store_event_id, None
            )
            self._request_kv_swap_store_event_to_request_id.pop(
                swap.store_event_id, None
            )
        if swap.load_event_id is not None:
            self._managed_context_load_events_to_submit.pop(
                swap.load_event_id, None
            )
            self._request_kv_swap_load_event_to_request_id.pop(
                swap.load_event_id, None
            )
            self._request_kv_swap_finished_load_req_ids.discard(request_id)
        if release_gpu_entries:
            self._release_request_kv_swap_entries(request_id, swap.entries)
        self._managed_context_free_cpu_block_ids(swap.cpu_block_ids_by_group)
        if os.environ.get("KVE_TRACE_REQUEST_KV_SWAP") == "1":
            logger.warning(
                "[REQUEST-KV-SWAP-RELEASE] req=%s reason=%s status=%s",
                request_id[:8],
                reason,
                swap.status,
            )
        return True

    def _release_request_kv_swap_for_finish(
        self, request_id: str, reason: str
    ) -> tuple[bool, bool]:
        swap = self._request_kv_swaps.get(request_id)
        if swap is None:
            return False, False

        if swap.status == "store_pending":
            event_queued = (
                swap.store_event_id in self._managed_context_store_events_to_submit
            )
            self._release_request_kv_swap(
                request_id,
                reason,
                release_gpu_entries=False,
            )
            if event_queued:
                return False, False
            return True, True

        self._release_request_kv_swap(
            request_id,
            reason,
            release_gpu_entries=True,
        )
        return True, True

    def _try_progress_request_kv_swap(
        self,
        request: Request,
        *,
        token_budget: int | None = None,
    ) -> bool:
        request_id = request.request_id
        swap = self._request_kv_swaps.get(request_id)
        if swap is None:
            return False
        if swap.status == "store_pending":
            return False
        if swap.status == "swapped":
            ready_head = self._request_kv_swap_ready_head()
            if ready_head is not None and ready_head != request_id:
                swap.last_error = (
                    "request KV swap load is waiting behind ready request: "
                    f"head={ready_head[:8]}"
                )
                return False
            extra_required_gpu_blocks = (
                self._request_kv_swap_next_allocation_block_demand(
                    request,
                    swap,
                    token_budget=(
                        int(
                            getattr(
                                self,
                                "max_num_scheduled_tokens",
                                getattr(request, "num_tokens", 0),
                            )
                            or 0
                        )
                        if token_budget is None
                        else token_budget
                    ),
                )
            )
            error = self._start_request_kv_swap_load(
                request,
                extra_required_gpu_blocks=extra_required_gpu_blocks,
            )
            if error is not None:
                swap.last_error = error
                if os.environ.get("KVE_TRACE_REQUEST_KV_SWAP") == "1":
                    logger.warning(
                        "[REQUEST-KV-SWAP-IN-DEFER] req=%s %s",
                        request_id[:8],
                        error,
                    )
            return False
        if swap.status == "load_pending":
            if request_id not in self._request_kv_swap_finished_load_req_ids:
                return False
            self._complete_request_kv_swap_load(request)
            request.status = RequestStatus.PREEMPTED
            return True
        if swap.status == "expired":
            return False
        return False

    def _release_managed_context_span_gpu_entries(
        self, span: ManagedContextSpan
    ) -> int:
        released_blocks = self._release_managed_context_entries(span.entries)
        span.entries = []
        return released_blocks

    @staticmethod
    def _managed_context_span_key(
        span: ManagedContextSpan,
    ) -> tuple[str, str]:
        return (span.trace_id, span.span_id)

    def _reserve_managed_context_restore_spans(
        self,
        request_id: str,
        spans: list[ManagedContextSpan],
    ) -> None:
        self._reserve_managed_context_restore_span_keys(
            request_id,
            {self._managed_context_span_key(span) for span in spans},
        )

    def _reserve_managed_context_restore_span_keys(
        self,
        request_id: str,
        keys: set[tuple[str, str]],
    ) -> None:
        if keys:
            self._managed_context_restore_reservations[request_id] = keys

    def _reserve_managed_context_restore_request(self, request: Request) -> None:
        if not self._managed_context_enabled:
            return
        span_ids = self._managed_context_restore_span_ids(request)
        if not span_ids:
            return
        trace_id = self._managed_context_trace_id(request)
        if not trace_id:
            return
        self._reserve_managed_context_restore_span_keys(
            request.request_id,
            {(trace_id, span_id) for span_id in span_ids},
        )
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            logger.warning(
                "[MANAGED-CONTEXT-RESTORE-RESERVE] req=%s trace=%s spans=%s",
                request.request_id[:8],
                trace_id,
                span_ids,
            )

    def _release_managed_context_restore_reservation(
        self,
        request_id: str,
        reason: str,
    ) -> None:
        keys = self._managed_context_restore_reservations.pop(request_id, None)
        if (
            keys
            and os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1"
        ):
            logger.warning(
                "[MANAGED-CONTEXT-RESTORE-RESERVE-RELEASE] req=%s "
                "reason=%s spans=%s",
                request_id[:8],
                reason,
                sorted(f"{trace}:{span}" for trace, span in keys),
            )

    @staticmethod
    def _managed_context_restore_error_is_unavailable(
        error: str | None,
    ) -> bool:
        if error is None:
            return False
        return "is not available for this trace" in error

    @staticmethod
    def _managed_context_restore_error_is_oversize(
        error: str | None,
    ) -> bool:
        if error is None:
            return False
        return error.startswith("restore would exceed max_model_len:")

    def _drop_unavailable_managed_context_restore(
        self,
        request: Request,
        reason: str,
    ) -> bool:
        unavailable = self._managed_context_restore_error_is_unavailable(reason)
        oversize = self._managed_context_restore_error_is_oversize(reason)
        drop_unavailable = (
            os.environ.get(
                "KVE_MANAGED_CONTEXT_DROP_UNAVAILABLE_RESTORE",
                "0",
            )
            .lower()
            in ("1", "true", "yes", "on")
        )
        drop_oversize = (
            os.environ.get(
                "KVE_MANAGED_CONTEXT_DROP_OVERSIZE_RESTORE",
                "1",
            )
            .lower()
            in ("1", "true", "yes", "on")
        )
        if not (
            (unavailable and drop_unavailable)
            or (oversize and drop_oversize)
        ):
            return False
        if request.sampling_params is None:
            return False
        extra_args = dict(request.sampling_params.extra_args or {})
        restore_span_ids = extra_args.get("kve_restore_span_ids")
        if "kve_restore_span_ids" not in extra_args:
            return False

        for key in (
            "kve_restore_span_ids",
            "kve_restore_defer_until_prefill",
            "kve_restore_after_visible_tokens",
        ):
            extra_args.pop(key, None)
        request.sampling_params.extra_args = extra_args or None
        request.managed_context_defer_restore_until_prefill = False
        request_id = request.request_id
        if request_id:
            self._release_managed_context_restore_reservation(
                request_id,
                "restore-dropped",
            )
            self._release_managed_context_deferred_restore(
                request_id,
                "restore-dropped",
            )
        logger.warning(
            "[MANAGED-CONTEXT-RESTORE-DROP] req=%s restore=%s reason=%s",
            request_id[:8],
            restore_span_ids,
            reason,
        )
        return True

    def _managed_context_cpu_archive_protected_keys(
        self,
        extra: set[tuple[str, str]] | None = None,
    ) -> set[tuple[str, str]]:
        protected = set(extra or ())
        for keys in self._managed_context_restore_reservations.values():
            protected.update(keys)
        for key, span in self._managed_context_archive.items():
            if span.pending_load_count > 0:
                protected.add(key)
        return protected

    def _managed_context_remove_hot_gpu_key(
        self,
        key: tuple[str, str],
    ) -> None:
        try:
            self._managed_context_hot_gpu_order.remove(key)
        except ValueError:
            pass

    def _managed_context_touch_hot_gpu_span(
        self,
        span: ManagedContextSpan,
    ) -> None:
        key = self._managed_context_span_key(span)
        self._managed_context_remove_hot_gpu_key(key)
        self._managed_context_hot_gpu_order.append(key)

    def _release_managed_context_hot_gpu_span(
        self,
        span: ManagedContextSpan,
        reason: str,
        *,
        expire: bool = False,
    ) -> int:
        key = self._managed_context_span_key(span)
        self._managed_context_remove_hot_gpu_key(key)
        released_blocks = self._release_managed_context_span_gpu_entries(span)
        if released_blocks:
            stats = self._managed_context_hot_gpu_stats
            stats.resident_blocks = max(0, stats.resident_blocks - released_blocks)
            stats.evictions += 1
        if span.status == "cpu_hot":
            span.status = "expired" if expire else "cpu_offloaded"
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1" and released_blocks:
            logger.warning(
                "[MANAGED-CONTEXT-HOT-GPU-RELEASE] trace=%s span=%s "
                "reason=%s blocks=%d resident=%d budget=%d status=%s",
                span.trace_id,
                span.span_id,
                reason,
                released_blocks,
                self._managed_context_hot_gpu_stats.resident_blocks,
                self._managed_context_hot_gpu_stats.budget_blocks,
                span.status,
            )
        return released_blocks

    def _managed_context_evict_hot_gpu_blocks(
        self,
        needed_blocks: int,
        *,
        protected: set[tuple[str, str]] | None = None,
        reason: str,
    ) -> None:
        protected = protected or set()
        stats = self._managed_context_hot_gpu_stats
        if stats.budget_blocks <= 0:
            return
        while (
            stats.resident_blocks + max(0, needed_blocks) > stats.budget_blocks
            and self._managed_context_hot_gpu_order
        ):
            key = self._managed_context_hot_gpu_order.popleft()
            if key in protected:
                self._managed_context_hot_gpu_order.append(key)
                if all(item in protected for item in self._managed_context_hot_gpu_order):
                    break
                continue
            span = self._managed_context_archive.get(key)
            if span is None or span.status != "cpu_hot":
                continue
            self._release_managed_context_hot_gpu_span(span, reason)

    def _managed_context_free_hot_gpu_for_blocks(
        self,
        needed_blocks: int,
        *,
        protected: set[tuple[str, str]] | None = None,
    ) -> int:
        protected = protected or set()
        if needed_blocks <= 0:
            return 0
        managers = self.kv_cache_manager.coordinator.single_type_managers
        if not managers:
            return 0
        released_blocks = 0
        while self._managed_context_hot_gpu_order:
            free_blocks = min(
                manager.block_pool.get_num_free_blocks() for manager in managers
            )
            if free_blocks >= needed_blocks:
                return released_blocks
            key = self._managed_context_hot_gpu_order.popleft()
            if key in protected:
                self._managed_context_hot_gpu_order.append(key)
                if all(item in protected for item in self._managed_context_hot_gpu_order):
                    return released_blocks
                continue
            span = self._managed_context_archive.get(key)
            if span is None or span.status != "cpu_hot":
                continue
            released_blocks += self._release_managed_context_hot_gpu_span(
                span, "gpu-free-block-pressure"
            )
        return released_blocks

    def _managed_context_release_hot_gpu_pressure(self, reason: str) -> int:
        if not self._managed_context_cpu_archive_enabled:
            return 0
        free_blocks = self._managed_context_min_free_gpu_blocks()
        target_free_blocks = max(free_blocks + 1, self.block_size)
        return self._managed_context_free_hot_gpu_for_blocks(
            target_free_blocks,
            protected=self._managed_context_cpu_archive_protected_keys(),
        )

    def _retry_managed_context_gpu_pinned_offloads(self, reason: str) -> int:
        if not (
            self._managed_context_enabled
            and self._managed_context_cpu_offload_immediate
        ):
            return 0
        started = 0
        for key in list(self._managed_context_archive_order):
            if self._managed_context_cpu_store_transfer_limit_error() is not None:
                break
            span = self._managed_context_archive.get(key)
            if span is None or span.status != "gpu_pinned":
                continue
            if (
                not self._managed_context_cpu_evict_on_capacity
                and len(self._managed_context_cpu_free_block_ids)
                < span.kv_block_count
            ):
                continue
            error = self._start_managed_context_cpu_offload(span, reason)
            if error is None:
                started += 1
                continue
            if "store transfer limit" in error:
                break
            if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
                logger.warning(
                    "[MANAGED-CONTEXT-CPU-RETRY-SKIP] trace=%s span=%s %s",
                    span.trace_id,
                    span.span_id,
                    error,
                )
        return started

    def _promote_managed_context_hot_gpu_span(
        self,
        span: ManagedContextSpan,
        entries: list[tuple[Any, list[Any]]],
    ) -> bool:
        stats = self._managed_context_hot_gpu_stats
        if (
            stats.budget_blocks <= 0
            or span.status != "cpu_offloaded"
            or not span.cpu_block_ids_by_group
            or not entries
        ):
            return False
        block_count = sum(len(blocks) for _manager, blocks in entries)
        if block_count <= 0 or block_count > stats.budget_blocks:
            return False

        key = self._managed_context_span_key(span)
        self._managed_context_evict_hot_gpu_blocks(
            block_count,
            protected={key},
            reason="hot-gpu-budget",
        )
        if stats.resident_blocks + block_count > stats.budget_blocks:
            return False

        span.entries = entries
        span.status = "cpu_hot"
        self._managed_context_remove_hot_gpu_key(key)
        self._managed_context_hot_gpu_order.append(key)
        stats.resident_blocks += block_count
        stats.promotions += 1
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            logger.warning(
                "[MANAGED-CONTEXT-HOT-GPU-PROMOTE] trace=%s span=%s "
                "blocks=%d resident=%d budget=%d",
                span.trace_id,
                span.span_id,
                block_count,
                stats.resident_blocks,
                stats.budget_blocks,
            )
        return True

    def _release_managed_context_pending_load_span_refs(
        self, pending: ManagedContextPendingLoad
    ) -> None:
        for span in pending.spans:
            if span.span_id not in pending.restored_entries_by_span:
                continue
            span.pending_load_count = max(0, span.pending_load_count - 1)
            if span.status == "expired" and span.pending_load_count == 0:
                self._managed_context_free_span_cpu_blocks(span)

    def _alloc_managed_context_cpu_blocks(
        self,
        num_blocks: int,
        *,
        protected: set[tuple[str, str]] | None = None,
    ) -> list[int] | None:
        if (
            not self._managed_context_cpu_archive_enabled
            or num_blocks <= 0
            or num_blocks > self._managed_context_cpu_max_blocks
        ):
            return None

        protected = self._managed_context_cpu_archive_protected_keys(protected)
        if (
            not self._managed_context_cpu_evict_on_capacity
            and len(self._managed_context_cpu_free_block_ids) < num_blocks
        ):
            return None
        while len(self._managed_context_cpu_free_block_ids) < num_blocks:
            if not self._managed_context_archive_order:
                return None
            key = self._managed_context_archive_order.popleft()
            if key in protected:
                self._managed_context_archive_order.append(key)
                if all(
                    item in protected
                    for item in self._managed_context_archive_order
                ):
                    return None
                continue
            if key in self._managed_context_archive:
                self._release_managed_context_span(key, "cpu-block-limit")

        if len(self._managed_context_cpu_free_block_ids) < num_blocks:
            return None
        contiguous = _pop_contiguous_managed_context_cpu_blocks(
            self._managed_context_cpu_free_block_ids,
            num_blocks,
        )
        if contiguous is not None:
            return contiguous
        return [
            int(self._managed_context_cpu_free_block_ids.popleft())
            for _ in range(num_blocks)
        ]

    def _managed_context_cpu_store_transfer_limit_error(self) -> str | None:
        max_pending = max(0, self._managed_context_cpu_max_pending_store_events)
        if max_pending <= 0:
            return None
        pending = len(self._managed_context_store_event_to_span) + len(
            getattr(self, "_request_kv_swap_store_event_to_request_id", {})
        ) + len(
            getattr(self, "_phase4_pin_store_event_to_trace_id", {})
        )
        if pending < max_pending:
            return None
        return (
            "managed-context CPU store transfer limit reached: "
            f"pending={pending} max={max_pending}"
        )

    def _managed_context_cpu_load_transfer_limit_error(self) -> str | None:
        max_pending = max(0, self._managed_context_cpu_max_pending_load_events)
        if max_pending <= 0:
            return None
        pending = len(self._managed_context_load_event_to_request_id) + len(
            getattr(self, "_request_kv_swap_load_event_to_request_id", {})
        ) + len(
            getattr(self, "_phase4_pin_load_event_to_trace_id", {})
        )
        if pending < max_pending:
            return None
        return (
            "managed-context CPU load transfer limit reached: "
            f"pending={pending} max={max_pending}"
        )

    def _start_managed_context_cpu_offload(
        self,
        span: ManagedContextSpan,
        reason: str,
    ) -> str | None:
        if not self._managed_context_cpu_archive_enabled:
            return "managed-context CPU archive is disabled"
        key = self._managed_context_span_key(span)
        if span.status == "cpu_offloaded" or span.status == "offload_pending":
            return None
        if span.status == "cpu_hot":
            self._release_managed_context_hot_gpu_span(span, reason)
            return None
        if span.status != "gpu_pinned":
            return f"span {span.span_id!r} cannot be offloaded from {span.status}"
        if not span.entries:
            return f"span {span.span_id!r} has no GPU blocks to offload"

        limit_error = self._managed_context_cpu_store_transfer_limit_error()
        if limit_error is not None:
            return limit_error

        cpu_block_ids = self._alloc_managed_context_cpu_blocks(
            span.kv_block_count,
            protected={key},
        )
        if cpu_block_ids is None:
            return (
                f"span {span.span_id!r} needs {span.kv_block_count} CPU "
                f"blocks, exceeds available managed-context CPU capacity "
                f"{self._managed_context_cpu_max_blocks}"
            )

        cpu_by_group: list[list[int]] = []
        offset = 0
        for logical_starts in span.logical_start_by_group:
            count = len(logical_starts)
            cpu_by_group.append(cpu_block_ids[offset : offset + count])
            offset += count
        if offset != len(cpu_block_ids):
            self._managed_context_free_cpu_block_ids((cpu_block_ids,))
            return f"span {span.span_id!r} has inconsistent block metadata"

        event_id = self._next_managed_context_transfer_event_id()
        span.status = "offload_pending"
        span.cpu_block_ids_by_group = tuple(cpu_by_group)
        span.offload_event_id = event_id
        self._managed_context_store_events_to_submit[event_id] = (
            ManagedContextCopyEvent(
                event_id=event_id,
                gpu_block_ids=self._managed_context_gpu_block_ids(span.entries),
                cpu_block_ids=cpu_block_ids,
            )
        )
        self._managed_context_store_event_to_span[event_id] = span
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            logger.warning(
                "[MANAGED-CONTEXT-OFFLOAD-SUBMIT] trace=%s span=%s event=%d "
                "reason=%s gpu_blocks=%s cpu_blocks=%s",
                span.trace_id,
                span.span_id,
                event_id,
                reason,
                self._managed_context_gpu_block_ids(span.entries),
                cpu_block_ids,
            )
        return None

    def _drain_managed_context_transfer_metadata(
        self,
    ) -> ManagedContextTransferMetadata | None:
        if (
            not self._managed_context_store_events_to_submit
            and not self._managed_context_load_events_to_submit
        ):
            return None
        metadata = ManagedContextTransferMetadata(
            store_events=list(self._managed_context_store_events_to_submit.values()),
            load_events=list(self._managed_context_load_events_to_submit.values()),
        )
        self._managed_context_store_events_to_submit.clear()
        self._managed_context_load_events_to_submit.clear()
        return None if metadata.is_empty() else metadata

    def _release_managed_context_span(
        self, key: tuple[str, str], reason: str
    ) -> None:
        span = self._managed_context_archive.pop(key, None)
        if span is None:
            return
        released_blocks = 0
        event_id = span.offload_event_id
        if span.status == "cpu_hot":
            released_blocks += self._release_managed_context_hot_gpu_span(
                span, reason, expire=True
            )
            if span.pending_load_count == 0:
                self._managed_context_free_span_cpu_blocks(span)
        elif span.status == "offload_pending" and event_id is not None:
            if event_id in self._managed_context_store_events_to_submit:
                self._managed_context_store_events_to_submit.pop(event_id, None)
                self._managed_context_store_event_to_span.pop(event_id, None)
                released_blocks += self._release_managed_context_span_gpu_entries(
                    span
                )
                self._managed_context_free_span_cpu_blocks(span)
            else:
                # The worker may still be reading the source GPU blocks. Keep
                # them alive until the completion event comes back.
                span.status = "expired"
        else:
            released_blocks += self._release_managed_context_span_gpu_entries(
                span
            )
            if span.pending_load_count == 0:
                self._managed_context_free_span_cpu_blocks(span)
        span.status = "expired"
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            logger.warning(
                "[MANAGED-CONTEXT-RELEASE] trace=%s span=%s reason=%s "
                "blocks=%d tokens=%d turns=%d..%d",
                span.trace_id,
                span.span_id,
                reason,
                released_blocks,
                len(span.token_ids),
                span.absolute_turn_start,
                span.absolute_turn_end,
            )

    def _release_all_managed_context_spans(self, reason: str) -> None:
        self._managed_context_restore_reservations.clear()
        for key in list(self._managed_context_archive):
            self._release_managed_context_span(key, reason)
        self._managed_context_archive_order.clear()

    def _release_managed_context_active_restore(
        self, request_id: str, reason: str
    ) -> None:
        restore = self._managed_context_active_restores.pop(request_id, None)
        if restore is None:
            return
        released_blocks = 0
        for manager, blocks in restore.entries:
            if blocks:
                manager.block_pool.free_blocks(blocks)
                released_blocks += len(blocks)
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            logger.warning(
                "[MANAGED-CONTEXT-RESTORE-RELEASE] req=%s reason=%s "
                "spans=%s blocks=%d hidden_tokens=%d",
                request_id[:8],
                reason,
                restore.span_ids,
                released_blocks,
                restore.num_tokens,
            )

    def _managed_context_defer_restore_until_prefill(
        self, request: Request
    ) -> bool:
        if request.sampling_params is None:
            request.managed_context_defer_restore_until_prefill = False
            return False
        extra_args = request.sampling_params.extra_args or {}
        raw = extra_args.get("kve_restore_defer_until_prefill")
        if raw is None:
            # Restored hidden KVs are older context. Visible prompt tokens
            # added by the current request must be prefetched without being
            # able to attend to those restored blocks.
            enabled = "kve_restore_span_ids" in extra_args
        elif isinstance(raw, str):
            enabled = raw.lower() in ("1", "true", "yes", "on")
        else:
            enabled = bool(raw)
        request.managed_context_defer_restore_until_prefill = enabled
        return enabled

    def _managed_context_deferred_restore_ready_tokens(
        self, request: Request
    ) -> int:
        """Visible tokens to prefill before activating hidden restore.

        By default, the final visible prompt token runs with restored hidden
        K/V so its logits produce the first answer token in the restored
        context. Callers can set ``kve_restore_after_visible_tokens`` to attach
        hidden K/V earlier, for example immediately after the visible retrieve
        JSON and before the restored-memory answer-control suffix.
        """
        if request.sampling_params is not None:
            extra_args = request.sampling_params.extra_args or {}
            raw_ready = extra_args.get("kve_restore_after_visible_tokens")
            if raw_ready is not None:
                try:
                    ready_tokens = int(raw_ready)
                except (TypeError, ValueError):
                    ready_tokens = request.num_prompt_tokens - 1
                return max(
                    0,
                    min(ready_tokens, max(0, request.num_prompt_tokens - 1)),
                )
        return max(0, request.num_prompt_tokens - 1)

    def _cap_managed_context_deferred_prefill_tokens(
        self,
        request: Request,
        *,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        if (
            num_new_tokens <= 0
            or request.request_id in self._managed_context_active_restores
            or not self._managed_context_defer_restore_until_prefill(request)
        ):
            return num_new_tokens
        ready_tokens = self._managed_context_deferred_restore_ready_tokens(
            request
        )
        if ready_tokens <= 0 or num_computed_tokens >= ready_tokens:
            return num_new_tokens
        if num_computed_tokens + num_new_tokens <= ready_tokens:
            return num_new_tokens
        capped = ready_tokens - num_computed_tokens
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            logger.warning(
                "[MANAGED-CONTEXT-PREFILL-HOLDBACK] req=%s "
                "scheduled=%d->%d computed=%d prompt=%d",
                request.request_id[:8],
                num_new_tokens,
                capped,
                num_computed_tokens,
                request.num_prompt_tokens,
            )
        return max(0, capped)

    def _set_managed_context_deferred_restore(
        self,
        request: Request,
        spans: list[ManagedContextSpan],
        *,
        restored_entries_by_span: dict[
            str, list[tuple[Any, list[Any]]]
        ] | None = None,
        skip_hot_hit_span_ids: set[str] | None = None,
    ) -> None:
        if not spans:
            return
        request_id = request.request_id
        self._release_managed_context_deferred_restore(request_id, "replace")
        self._reserve_managed_context_restore_spans(request_id, spans)
        restored_entries_by_span = restored_entries_by_span or {}
        skip_hot_hit_span_ids = skip_hot_hit_span_ids or set()
        deferred = ManagedContextDeferredRestore(
            span_ids=[span.span_id for span in spans],
            spans=list(spans),
            restored_entries_by_span=restored_entries_by_span,
            skip_hot_hit_span_ids=set(skip_hot_hit_span_ids),
            created_at=time.monotonic(),
        )
        self._managed_context_deferred_restores[request_id] = deferred
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            logger.warning(
                "[MANAGED-CONTEXT-RESTORE-DEFER] req=%s spans=%s "
                "computed=%d prompt=%d loaded_spans=%s",
                request_id[:8],
                deferred.span_ids,
                request.num_computed_tokens,
                request.num_prompt_tokens,
                list(restored_entries_by_span),
            )

    def _release_managed_context_deferred_restore(
        self, request_id: str, reason: str
    ) -> None:
        deferred = self._managed_context_deferred_restores.pop(request_id, None)
        if deferred is None:
            return
        self._release_managed_context_restore_reservation(request_id, reason)
        released_blocks = 0
        for entries in deferred.restored_entries_by_span.values():
            released_blocks += self._release_managed_context_entries(entries)
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            logger.warning(
                "[MANAGED-CONTEXT-RESTORE-DEFER-RELEASE] req=%s "
                "reason=%s spans=%s released_blocks=%d",
                request_id[:8],
                reason,
                deferred.span_ids,
                released_blocks,
            )

    def _activate_deferred_managed_context_restore_if_ready(
        self,
        request: Request,
        *,
        computed_tokens: int | None = None,
    ) -> bool:
        request_id = request.request_id
        if request_id in self._managed_context_active_restores:
            return False
        if not self._managed_context_defer_restore_until_prefill(request):
            return False
        ready_tokens = self._managed_context_deferred_restore_ready_tokens(
            request
        )
        current_computed_tokens = (
            request.num_computed_tokens
            if computed_tokens is None
            else int(computed_tokens)
        )
        if current_computed_tokens < ready_tokens:
            return False

        deferred = self._managed_context_deferred_restores.pop(request_id, None)
        if deferred is None:
            spans, error = self._validate_managed_context_restore_request(request)
            if error is not None:
                if self._drop_unavailable_managed_context_restore(request, error):
                    return False
                logger.error(
                    "[MANAGED-CONTEXT-DEFER-ACTIVATE-ABORT] req=%s %s",
                    request_id[:8],
                    error,
                )
                self.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)
                return False
            if not spans:
                return False
            deferred = ManagedContextDeferredRestore(
                span_ids=[span.span_id for span in spans],
                spans=list(spans),
                restored_entries_by_span={},
                skip_hot_hit_span_ids=set(),
                created_at=time.monotonic(),
            )

        self._release_managed_context_restore_reservation(request_id, "activate")
        self._activate_managed_context_restore(
            request,
            deferred.spans,
            restored_entries_by_span=deferred.restored_entries_by_span,
            skip_hot_hit_span_ids=deferred.skip_hot_hit_span_ids,
        )
        request.needs_rebuild = True
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            restore = self._managed_context_active_restores.get(request_id)
            logger.warning(
                "[MANAGED-CONTEXT-RESTORE-DEFER-ACTIVE] req=%s spans=%s "
                "hidden_tokens=%d computed=%d prompt=%d ready=%d",
                request_id[:8],
                deferred.span_ids,
                restore.num_tokens if restore is not None else 0,
                current_computed_tokens,
                request.num_prompt_tokens,
                ready_tokens,
            )
        return True

    def _release_managed_context_pending_load(
        self, request_id: str, reason: str, *, only_if_safe: bool = True
    ) -> bool:
        pending = self._managed_context_pending_loads.get(request_id)
        if pending is None:
            return True

        event_id = pending.event_id
        event_not_submitted = event_id in self._managed_context_load_events_to_submit
        event_finished = request_id in self._managed_context_finished_load_req_ids
        if only_if_safe and not event_not_submitted and not event_finished:
            return False

        self._managed_context_pending_loads.pop(request_id, None)
        self._managed_context_load_events_to_submit.pop(event_id, None)
        self._managed_context_load_event_to_request_id.pop(event_id, None)
        self._managed_context_finished_load_req_ids.discard(request_id)

        self._release_managed_context_pending_load_span_refs(pending)
        released_blocks = 0
        for entries in pending.restored_entries_by_span.values():
            released_blocks += self._release_managed_context_entries(entries)
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            logger.warning(
                "[MANAGED-CONTEXT-LOAD-RELEASE] req=%s reason=%s "
                "spans=%s blocks=%d event=%d",
                request_id[:8],
                reason,
                pending.span_ids,
                released_blocks,
                event_id,
            )
        return True

    def _managed_context_active_hidden_kv(
        self, request_id: str
    ) -> tuple[tuple[list[int], ...], int, list[str]]:
        restore = self._managed_context_active_restores.get(request_id)
        if restore is None:
            return (), 0, []
        return restore.block_ids, restore.num_tokens, list(restore.span_ids)

    def _managed_context_min_free_gpu_blocks(self) -> int:
        managers = self.kv_cache_manager.coordinator.single_type_managers
        if not managers:
            return 0
        return min(manager.block_pool.get_num_free_blocks() for manager in managers)

    def _managed_context_visible_allocation_demand(
        self,
        request: Request,
        *,
        num_new_tokens: int,
        num_new_computed_tokens: int,
        new_computed_blocks: KVCacheBlocks,
        num_lookahead_tokens: int,
        num_external_computed_tokens: int,
        num_encoder_tokens: int,
    ) -> int:
        """Estimate visible-request GPU block demand without mutating state."""
        if num_new_tokens <= 0 and num_external_computed_tokens <= 0:
            return 0

        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        raw_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        total_computed_tokens = min(
            raw_total_computed_tokens,
            self.max_model_len,
        )
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens,
            self.max_model_len,
        )

        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_blocks.blocks,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=raw_total_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

    def _align_managed_context_restore_position(
        self, request: Request, spans: list[ManagedContextSpan]
    ) -> bool:
        """Seed a non-deferred retry request's RoPE frame after hidden KV.

        Managed-context restore mirrors Phase4's retained-KV invariant: cached
        K/V keeps the original RoPE frame, and newly written retry K/V must be
        positioned after the retained/restored frame. The scheduler must do
        this before allocate_slots so freshly allocated blocks get logical_start
        metadata matching the positions the worker will use.

        Deferred restore is different: the visible prompt is prefetched first
        in its normal Phase4 frame, then old hidden K/V is attached as sideband
        evidence. In that mode, callers must not use restored spans to mutate
        the visible prompt's position_offset.
        """
        if not spans or not self._managed_context_align_positions:
            return False

        max_hidden_logical_end = -1
        for span in spans:
            for logical_starts in span.logical_start_by_group:
                for logical_start in logical_starts:
                    if logical_start >= 0:
                        max_hidden_logical_end = max(
                            max_hidden_logical_end,
                            int(logical_start) + self.block_size,
                        )
        if max_hidden_logical_end < 0:
            return False

        protected_prefix_len = self._worker_protected_prefix_len(request)
        required_offset = max(
            0,
            max_hidden_logical_end - protected_prefix_len,
        )
        if required_offset <= request.position_offset:
            return False

        old_offset = request.position_offset
        request.position_offset = required_offset
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            logger.warning(
                "[MANAGED-CONTEXT-ALIGN] req=%s position_offset %d -> %d "
                "hidden_logical_end=%d protected_prefix_len=%d spans=%s",
                request.request_id[:8],
                old_offset,
                required_offset,
                max_hidden_logical_end,
                protected_prefix_len,
                [span.span_id for span in spans],
            )
        return True

    def _managed_context_total_archive_blocks(self) -> int:
        return sum(
            span.kv_block_count
            for span in self._managed_context_archive.values()
            if span.status in ("gpu_pinned", "offload_pending", "cpu_hot")
        )

    def _prune_managed_context_archive(self) -> None:
        if not self._managed_context_archive:
            return

        ttl_seconds = max(0.0, self._managed_context_archive_ttl_seconds)
        now = time.monotonic()
        protected = self._managed_context_cpu_archive_protected_keys()
        for key in list(self._managed_context_archive_order):
            if key in protected:
                continue
            span = self._managed_context_archive.get(key)
            if span is None:
                continue
            if ttl_seconds > 0 and now - span.created_at >= ttl_seconds:
                self._release_managed_context_span(key, "ttl")

        max_blocks = self._managed_context_archive_max_blocks
        if max_blocks is None:
            return
        while (
            self._managed_context_archive
            and self._managed_context_total_archive_blocks() > max_blocks
        ):
            if not self._managed_context_archive_order:
                break
            key = self._managed_context_archive_order.popleft()
            if key in protected:
                self._managed_context_archive_order.append(key)
                if all(
                    item in protected
                    for item in self._managed_context_archive_order
                ):
                    break
                continue
            if key in self._managed_context_archive:
                self._release_managed_context_span(key, "block-limit")

    def _archive_managed_context_span(
        self,
        request: Request,
        *,
        compaction_mgr: CompactingKVCacheManager,
        evict_start: int,
        evict_end: int,
        explicit_block_range: tuple[int, int] | None,
        last_turn_evicted: int,
        stride_used: int,
    ) -> list[str] | None:
        """Pin the blocks that turn-mode compaction is about to evict.

        The archive owns exactly one extra ref per captured block. The normal
        compaction path still deletes the blocks from the active request and
        decrements the request's refs as before.
        """
        if (
            not self._managed_context_enabled
            or explicit_block_range is None
            or evict_end <= evict_start
        ):
            return []
        trace_id = self._managed_context_trace_id(request)
        if not trace_id:
            return []

        start_block, end_block = explicit_block_range
        if start_block == end_block:
            return []

        collected: list[tuple[Any, list[Any], list[int]]] = []
        total_blocks = 0
        for manager in (compaction_mgr,):
            req_blocks = manager.req_to_blocks.get(request.request_id)
            if not req_blocks or end_block > len(req_blocks):
                return []
            blocks = [
                block
                for block in req_blocks[start_block:end_block]
                if not block.is_null
            ]
            if not blocks:
                continue
            collected.append(
                (manager, blocks, [int(block.logical_start) for block in blocks])
            )
            total_blocks += len(blocks)

        if not collected:
            return []
        max_blocks = self._managed_context_archive_max_blocks
        if max_blocks is not None and total_blocks > max_blocks:
            logger.warning(
                "[MANAGED-CONTEXT-SKIP] req=%s trace=%s evict=[%d,%d) "
                "blocks=%d exceeds archive_max_blocks=%d",
                request.request_id[:8],
                trace_id,
                evict_start,
                evict_end,
                total_blocks,
                max_blocks,
            )
            return []
        if (
            self._managed_context_skip_compaction_on_cpu_capacity
            and self._managed_context_cpu_offload_immediate
            and not self._managed_context_cpu_evict_on_capacity
            and total_blocks > len(self._managed_context_cpu_free_block_ids)
        ):
            cpu_free = len(self._managed_context_cpu_free_block_ids)
            skip_state = (evict_start, evict_end, total_blocks, cpu_free)
            if (
                getattr(
                    request,
                    "_kve_managed_context_last_capacity_skip",
                    None,
                )
                != skip_state
            ):
                request._kve_managed_context_last_capacity_skip = skip_state  # type: ignore[attr-defined]
                logger.warning(
                    "[MANAGED-CONTEXT-COMPACT-SKIP-CAPACITY] req=%s trace=%s "
                    "evict=[%d,%d) blocks=%d cpu_free=%d cpu_max=%d "
                    "gpu_pinned_blocks=%d managed=%s",
                    request.request_id[:8],
                    trace_id,
                    evict_start,
                    evict_end,
                    total_blocks,
                    cpu_free,
                    self._managed_context_cpu_max_blocks,
                    sum(
                        span.kv_block_count
                        for span in self._managed_context_archive.values()
                        if span.status == "gpu_pinned"
                    ),
                    self._kve_managed_context_diag_summary(),
                )
            return None

        entries: list[tuple[Any, list[Any]]] = []
        logical_start_by_group: list[list[int]] = []
        for manager, blocks, logical_starts in collected:
            manager.block_pool.touch(blocks)
            entries.append((manager, blocks))
            logical_start_by_group.append(logical_starts)

        next_id = self._managed_context_next_span_by_trace[trace_id] + 1
        self._managed_context_next_span_by_trace[trace_id] = next_id
        span_id = f"T{next_id:04d}"
        key = (trace_id, span_id)
        if last_turn_evicted >= 0 and stride_used > 0:
            absolute_turn_start = max(0, last_turn_evicted - stride_used + 1)
            absolute_turn_end = last_turn_evicted
        else:
            absolute_turn_start = -1
            absolute_turn_end = -1
        span = ManagedContextSpan(
            span_id=span_id,
            trace_id=trace_id,
            request_id=request.request_id[:8],
            absolute_turn_start=absolute_turn_start,
            absolute_turn_end=absolute_turn_end,
            token_ids=list(request._all_token_ids[evict_start:evict_end]),
            entries=entries,
            kv_block_count=total_blocks,
            logical_start_by_group=logical_start_by_group,
            position_offset_frame=int(request.position_offset),
            evict_start=evict_start,
            evict_end=evict_end,
            created_at=time.monotonic(),
        )
        self._managed_context_archive[key] = span
        self._managed_context_archive_order.append(key)
        if self._managed_context_cpu_offload_immediate:
            offload_error = self._start_managed_context_cpu_offload(
                span,
                "archive-immediate",
            )
            if offload_error is not None:
                logger.warning(
                    "[MANAGED-CONTEXT-CPU-SKIP] req=%s trace=%s span=%s "
                    "%s; keeping GPU-pinned",
                    request.request_id[:8],
                    trace_id,
                    span_id,
                    offload_error,
                )
        self._prune_managed_context_archive()
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            archive_block_ids = [
                [int(block.block_id) for block in blocks]
                for _manager, blocks in entries
            ]
            logger.warning(
                "[MANAGED-CONTEXT-ARCHIVE] trace=%s span=%s req=%s "
                "evict=[%d,%d) turns=%d..%d blocks=%d tokens=%d "
                "block_ids=%s logical_starts=%s token_head=%s token_tail=%s",
                trace_id,
                span_id,
                request.request_id[:8],
                evict_start,
                evict_end,
                absolute_turn_start,
                absolute_turn_end,
                total_blocks,
                len(span.token_ids),
                archive_block_ids,
                logical_start_by_group,
                span.token_ids[:8],
                span.token_ids[-8:],
            )
        return [span_id] if key in self._managed_context_archive else []

    def _managed_context_restore_needs_cpu_load(
        self, spans: list[ManagedContextSpan]
    ) -> bool:
        return any(span.status == "cpu_offloaded" for span in spans)

    def _start_managed_context_cpu_load(
        self,
        request: Request,
        spans: list[ManagedContextSpan],
        *,
        extra_required_gpu_blocks: int = 0,
    ) -> ManagedContextCPULoadStart:
        if not self._managed_context_restore_needs_cpu_load(spans):
            return ManagedContextCPULoadStart()
        if request.request_id in self._managed_context_pending_loads:
            return ManagedContextCPULoadStart()

        limit_error = self._managed_context_cpu_load_transfer_limit_error()
        if limit_error is not None:
            return ManagedContextCPULoadStart(limit_error, retryable=True)

        manager_to_index = {
            id(manager): i
            for i, manager in enumerate(
                self.kv_cache_manager.coordinator.single_type_managers
            )
        }
        if len(manager_to_index) != 1:
            return ManagedContextCPULoadStart(
                "managed-context CPU reload currently supports one KV cache group"
            )

        restored_entries_by_span: dict[str, list[tuple[Any, list[Any]]]] = {}
        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        groups: list[list[Any]] = [[] for _ in manager_to_index]
        load_block_count = _managed_context_cpu_reload_block_demand(spans)
        self._managed_context_free_hot_gpu_for_blocks(
            load_block_count + max(0, extra_required_gpu_blocks),
            protected={self._managed_context_span_key(span) for span in spans},
        )
        if self._managed_context_scheduler_accounted_restore:
            wait_reason = _managed_context_gpu_reload_wait_reason(
                reload_blocks=load_block_count,
                extra_blocks=extra_required_gpu_blocks,
                free_blocks=self._managed_context_min_free_gpu_blocks(),
            )
            if wait_reason is not None:
                return ManagedContextCPULoadStart(
                    wait_reason,
                    retryable=True,
                )
        self._managed_context_hot_gpu_stats.misses += sum(
            1 for span in spans if span.status == "cpu_offloaded"
        )

        def _fail(
            message: str,
            *,
            retryable: bool = False,
        ) -> ManagedContextCPULoadStart:
            for entries in restored_entries_by_span.values():
                self._release_managed_context_entries(entries)
            return ManagedContextCPULoadStart(message, retryable=retryable)

        try:
            for span in sorted(
                spans, key=lambda s: (s.absolute_turn_start, s.span_id)
            ):
                if span.status != "cpu_offloaded":
                    continue
                if not span.cpu_block_ids_by_group:
                    return _fail(f"span {span.span_id!r} has no CPU archive slots")
                span_entries: list[tuple[Any, list[Any]]] = []
                for manager in self.kv_cache_manager.coordinator.single_type_managers:
                    idx = manager_to_index[id(manager)]
                    try:
                        span_cpu_ids = span.cpu_block_ids_by_group[idx]
                        logical_starts = span.logical_start_by_group[idx]
                    except IndexError:
                        return _fail(
                            f"span {span.span_id!r} has incomplete CPU "
                            "position metadata"
                        )
                    if len(span_cpu_ids) != len(logical_starts):
                        return _fail(
                            f"span {span.span_id!r} CPU block metadata mismatch"
                        )
                    blocks = manager.block_pool.get_new_blocks(len(span_cpu_ids))
                    for block, logical_start in zip(blocks, logical_starts):
                        block.logical_start = int(logical_start)
                    span_entries.append((manager, blocks))
                    groups[idx].extend(blocks)
                    gpu_block_ids.extend(int(block.block_id) for block in blocks)
                    cpu_block_ids.extend(int(block_id) for block_id in span_cpu_ids)
                restored_entries_by_span[span.span_id] = span_entries
        except ValueError as exc:
            return _fail(
                f"insufficient GPU blocks for managed-context CPU reload: {exc}",
                retryable=self._managed_context_scheduler_accounted_restore,
            )

        if not gpu_block_ids:
            return ManagedContextCPULoadStart()

        event_id = self._next_managed_context_transfer_event_id()
        self._managed_context_load_events_to_submit[event_id] = (
            ManagedContextCopyEvent(
                event_id=event_id,
                gpu_block_ids=gpu_block_ids,
                cpu_block_ids=cpu_block_ids,
            )
        )
        self._managed_context_load_event_to_request_id[event_id] = request.request_id
        pending = ManagedContextPendingLoad(
            request_id=request.request_id,
            span_ids=[span.span_id for span in spans],
            spans=list(spans),
            restored_entries_by_span=restored_entries_by_span,
            block_ids=tuple(
                [int(block.block_id) for block in group] for group in groups
            ),
            num_tokens=sum(len(span.token_ids) for span in spans),
            event_id=event_id,
            created_at=time.monotonic(),
        )
        self._managed_context_pending_loads[request.request_id] = pending
        for span in spans:
            if span.span_id in restored_entries_by_span:
                span.pending_load_count += 1
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            logger.warning(
                "[MANAGED-CONTEXT-LOAD-SUBMIT] req=%s spans=%s event=%d "
                "gpu_blocks=%s cpu_blocks=%s logical_starts=%s",
                request.request_id[:8],
                [span.span_id for span in spans],
                event_id,
                gpu_block_ids,
                cpu_block_ids,
                {
                    span.span_id: [
                        list(group) for group in span.logical_start_by_group
                    ]
                    for span in spans
                },
            )
        return ManagedContextCPULoadStart()

    def _complete_managed_context_pending_load(self, request: Request) -> None:
        pending = self._managed_context_pending_loads.pop(
            request.request_id, None
        )
        if pending is None:
            return
        self._managed_context_finished_load_req_ids.discard(request.request_id)
        self._managed_context_load_event_to_request_id.pop(pending.event_id, None)
        self._release_managed_context_pending_load_span_refs(pending)
        active_entries_by_span: dict[str, list[tuple[Any, list[Any]]]] = {}
        promoted_span_ids: set[str] = set()
        for span in pending.spans:
            entries = pending.restored_entries_by_span.get(span.span_id)
            if entries is None:
                continue
            if self._promote_managed_context_hot_gpu_span(span, entries):
                promoted_span_ids.add(span.span_id)
            else:
                active_entries_by_span[span.span_id] = entries
        if (
            self._managed_context_defer_restore_until_prefill(request)
            and request.num_computed_tokens < request.num_prompt_tokens
        ):
            self._set_managed_context_deferred_restore(
                request,
                pending.spans,
                restored_entries_by_span=active_entries_by_span,
                skip_hot_hit_span_ids=promoted_span_ids,
            )
            return
        self._activate_managed_context_restore(
            request,
            pending.spans,
            restored_entries_by_span=active_entries_by_span,
            skip_hot_hit_span_ids=promoted_span_ids,
        )

    def _activate_managed_context_restore(
        self,
        request: Request,
        spans: list[ManagedContextSpan],
        restored_entries_by_span: dict[
            str, list[tuple[Any, list[Any]]]
        ] | None = None,
        skip_hot_hit_span_ids: set[str] | None = None,
    ) -> None:
        if not spans:
            return
        restored_entries_by_span = restored_entries_by_span or {}
        skip_hot_hit_span_ids = skip_hot_hit_span_ids or set()
        manager_to_index = {
            id(manager): i
            for i, manager in enumerate(
                self.kv_cache_manager.coordinator.single_type_managers
            )
        }
        if len(manager_to_index) != 1:
            raise RuntimeError(
                "managed-context restore currently supports one KV cache group"
            )
        groups: list[list[Any]] = [[] for _ in manager_to_index]
        span_ids: list[str] = []
        num_tokens = 0
        for span in sorted(spans, key=lambda s: (s.absolute_turn_start, s.span_id)):
            span_ids.append(span.span_id)
            num_tokens += len(span.token_ids)
            entries = restored_entries_by_span.get(span.span_id, span.entries)
            should_touch = span.span_id not in restored_entries_by_span
            for manager, blocks in entries:
                idx = manager_to_index.get(id(manager))
                if idx is None:
                    raise RuntimeError(
                        f"managed-context span {span.span_id} belongs to an "
                        "unknown KV cache manager"
                    )
                if should_touch:
                    manager.block_pool.touch(blocks)
                groups[idx].extend(blocks)
            if span.status == "cpu_hot" and should_touch:
                self._managed_context_touch_hot_gpu_span(span)
                if span.span_id not in skip_hot_hit_span_ids:
                    self._managed_context_hot_gpu_stats.hits += 1
                    if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
                        logger.warning(
                            "[MANAGED-CONTEXT-HOT-GPU-HIT] trace=%s span=%s "
                            "resident=%d budget=%d hits=%d misses=%d",
                            span.trace_id,
                            span.span_id,
                            self._managed_context_hot_gpu_stats.resident_blocks,
                            self._managed_context_hot_gpu_stats.budget_blocks,
                            self._managed_context_hot_gpu_stats.hits,
                            self._managed_context_hot_gpu_stats.misses,
                        )

        entries: list[tuple[Any, list[Any]]] = []
        for manager in self.kv_cache_manager.coordinator.single_type_managers:
            idx = manager_to_index[id(manager)]
            blocks = groups[idx]
            if not blocks:
                continue
            entries.append((manager, blocks))

        block_ids = tuple(
            [int(block.block_id) for block in group] for group in groups
        )
        self._release_managed_context_active_restore(
            request.request_id, "replace"
        )
        self._managed_context_active_restores[request.request_id] = (
            ManagedContextActiveRestore(
                span_ids=span_ids,
                entries=entries,
                block_ids=block_ids,
                num_tokens=num_tokens,
                created_at=time.monotonic(),
            )
        )
        if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
            span_block_ids: dict[str, list[list[int]]] = {}
            span_logical_starts: dict[str, list[list[int]]] = {}
            span_token_heads: dict[str, list[int]] = {}
            span_token_tails: dict[str, list[int]] = {}
            for span in spans:
                entries = restored_entries_by_span.get(span.span_id, span.entries)
                span_block_ids[span.span_id] = [
                    [int(block.block_id) for block in blocks]
                    for _manager, blocks in entries
                ]
                span_logical_starts[span.span_id] = [
                    list(group) for group in span.logical_start_by_group
                ]
                span_token_heads[span.span_id] = span.token_ids[:8]
                span_token_tails[span.span_id] = span.token_ids[-8:]
            logger.warning(
                "[MANAGED-CONTEXT-RESTORE-ACTIVE] req=%s spans=%s "
                "hidden_tokens=%d blocks=%d block_ids=%s "
                "span_block_ids=%s span_logical_starts=%s "
                "span_token_heads=%s span_token_tails=%s",
                request.request_id[:8],
                span_ids,
                num_tokens,
                sum(len(group) for group in groups),
                block_ids,
                span_block_ids,
                span_logical_starts,
                span_token_heads,
                span_token_tails,
            )

    def _managed_context_extra_span_ids(
        self,
        request: Request,
        key: str,
        *,
        max_spans: int,
    ) -> list[str] | None:
        if request.sampling_params is None:
            return []
        extra_args = request.sampling_params.extra_args or {}
        raw = extra_args.get(key)
        if raw is None:
            return []
        if not isinstance(raw, (list, tuple)):
            return None
        seen: set[str] = set()
        out: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item:
                return None
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        if len(out) > max_spans:
            return None
        return out

    def _managed_context_restore_span_ids(
        self, request: Request
    ) -> list[str] | None:
        return self._managed_context_extra_span_ids(
            request,
            "kve_restore_span_ids",
            max_spans=self._managed_context_recall_max_spans,
        )

    def _managed_context_offload_span_ids(
        self, request: Request
    ) -> list[str] | None:
        return self._managed_context_extra_span_ids(
            request,
            "kve_offload_span_ids",
            max_spans=self._managed_context_recall_max_spans,
        )

    def _validate_managed_context_offload_request(
        self, request: Request
    ) -> tuple[list[ManagedContextSpan], str | None]:
        span_ids = self._managed_context_offload_span_ids(request)
        if span_ids == []:
            return [], None
        if span_ids is None:
            return [], "malformed or over-budget kve_offload_span_ids"
        if not self._managed_context_enabled:
            return [], "managed context is disabled"
        if not self._managed_context_cpu_archive_enabled:
            return [], "managed-context CPU archive is disabled"
        trace_id = self._managed_context_trace_id(request)
        if not trace_id:
            return [], "offload requires a valid Phase4 trace id"
        self._prune_managed_context_archive()
        spans: list[ManagedContextSpan] = []
        for span_id in span_ids:
            span = self._managed_context_archive.get((trace_id, span_id))
            if span is None or span.status not in (
                "gpu_pinned",
                "offload_pending",
                "cpu_offloaded",
                "cpu_hot",
            ):
                return [], f"span {span_id!r} is not available for this trace"
            spans.append(span)
        return spans, None

    def _start_managed_context_cpu_offloads(
        self,
        spans: list[ManagedContextSpan],
        reason: str,
    ) -> str | None:
        for span in spans:
            error = self._start_managed_context_cpu_offload(span, reason)
            if error is not None:
                return error
        return None

    def _validate_managed_context_restore_request(
        self, request: Request
    ) -> tuple[list[ManagedContextSpan], str | None]:
        span_ids = self._managed_context_restore_span_ids(request)
        if span_ids == []:
            return [], None
        if span_ids is None:
            return [], "malformed or over-budget kve_restore_span_ids"
        if not self._managed_context_enabled:
            return [], "managed context is disabled"
        trace_id = self._managed_context_trace_id(request)
        if not trace_id:
            return [], "restore requires a valid Phase4 trace id"
        if len(self.kv_cache_manager.coordinator.single_type_managers) != 1:
            return [], "managed-context restore currently supports one KV cache group"
        self._prune_managed_context_archive()
        spans: list[ManagedContextSpan] = []
        total_blocks = 0
        total_tokens = 0
        for span_id in span_ids:
            span = self._managed_context_archive.get((trace_id, span_id))
            if span is None or span.status not in (
                "gpu_pinned",
                "offload_pending",
                "cpu_offloaded",
                "cpu_hot",
            ):
                return [], f"span {span_id!r} is not available for this trace"
            if span.status == "offload_pending" and not span.entries:
                return [], f"span {span_id!r} is still offloading"
            if span.status == "cpu_offloaded" and not span.cpu_block_ids_by_group:
                return [], f"span {span_id!r} has no CPU archive slots"
            if span.status == "cpu_hot" and not span.entries:
                return [], f"span {span_id!r} has no hot GPU blocks"
            spans.append(span)
            total_blocks += span.kv_block_count
            total_tokens += len(span.token_ids)
        max_blocks = self._managed_context_recall_max_kv_blocks
        if max_blocks is not None and total_blocks > max_blocks:
            return [], (
                f"restore needs {total_blocks} KV blocks, exceeds "
                f"KVE_MANAGED_CONTEXT_RECALL_MAX_KV_BLOCKS={max_blocks}"
            )
        if request.num_tokens + total_tokens > self.max_model_len:
            return [], (
                f"restore would exceed max_model_len: visible={request.num_tokens} "
                f"hidden={total_tokens} max={self.max_model_len}"
            )
        return spans, None

    def _phase4_pinned_cache_blocks(
        self, trace_id: str, expected_cached_tokens: int
    ) -> tuple[KVCacheBlocks, int, int] | None:
        if not trace_id or expected_cached_tokens <= 0:
            return None
        entries = self._phase4_pinned_blocks.get(trace_id)
        if entries is None:
            return None
        if entries.status != "gpu_pinned":
            return None

        by_manager_id = {
            id(manager): blocks for manager, blocks in entries.entries
        }
        groups: list[list[Any]] = []
        min_cached_tokens = expected_cached_tokens
        for manager in self.kv_cache_manager.coordinator.single_type_managers:
            blocks = by_manager_id.get(id(manager))
            if blocks is None:
                return None
            num_blocks = expected_cached_tokens // manager.block_size
            if len(blocks) < num_blocks:
                return None
            groups.append(list(blocks[:num_blocks]))
            min_cached_tokens = min(
                min_cached_tokens, num_blocks * manager.block_size
            )

        inherited_offset = 0
        first_group = groups[0] if groups else []
        block_size = (
            self.kv_cache_manager.coordinator.single_type_managers[0].block_size
            if self.kv_cache_manager.coordinator.single_type_managers
            else max(1, self._compaction_block_size)
        )
        for block_idx, block in enumerate(first_group):
            if block.logical_start < 0:
                continue
            block_offset = block.logical_start - block_idx * block_size
            if block_offset != 0:
                inherited_offset = block_offset
                break

        return (
            self.kv_cache_manager.create_kv_cache_blocks(tuple(groups)),
            min_cached_tokens,
            inherited_offset,
        )

    def _phase4_expected_cached_tokens(
        self, request: Request
    ) -> int | None:
        """Return the rollout-local Phase4 prefix length, when supplied.

        This is production metadata, not only a debug aid. The scheduler's
        prefix cache can validly hit a longer global prefix than the current
        rollout-local state, but the trainer mirror only owns the retained
        Phase4 prefix sent by the orchestrator.
        """
        if (
            not self._compaction_enabled
            or self._compaction_max_turns <= 0
            or not self.cache_config.enable_prefix_caching
        ):
            return None
        if request.sampling_params is None:
            return None
        extra_args = request.sampling_params.extra_args or {}
        raw_expected = extra_args.get("kve_phase4_expected_cached_tokens")
        if raw_expected is None:
            return None
        try:
            expected = int(raw_expected)
        except (TypeError, ValueError):
            return None
        if expected <= 0:
            return None
        return min(expected, request.num_prompt_tokens)

    def _apply_trim(
        self,
        request: Request,
        *,
        evict_start: int,
        evict_end: int,
        total_evicted: int,
        stride_used: int,
        num_turns_evicted_after: int,
        trim_prompt_token_ids: bool,
    ) -> tuple[int, int]:
        """Mutate request state to reflect eviction of [evict_start, evict_end).

        Shared between mid-generation compaction (`_compact_request`
        called from `update_from_output`, where the to-be-evicted KV
        blocks have already been physically populated by prior steps)
        and in-step admission compaction (`_compact_request` called
        via `_run_admission_eviction_loop` from
        `_apply_inline_admission_eviction`, where blocks are freed at
        the END of `schedule()` BEFORE the worker's prefill kernel
        runs).

        When trim_prompt_token_ids=True, the raw prompt_token_ids list is
        also trimmed — required for admission so the worker prefills
        the shortened prompt. Mid-generation callers pass False because
        prompt_token_ids is a historical artifact at that point and
        mutating it would confuse downstream consumers (segmented_forward
        reconstructs pre-eviction prompt from the original list).

        Returns (prompt_tokens_evicted, output_tokens_evicted).
        """
        prompt_len = request.num_prompt_tokens
        prompt_tokens_evicted = max(
            0, min(prompt_len, evict_end) - evict_start
        )
        output_tokens_evicted = total_evicted - prompt_tokens_evicted

        # 1. Trim all_token_ids at the eviction boundary.
        del request._all_token_ids[evict_start:evict_end]
        request.all_token_ids = ConstantList(request._all_token_ids)

        # 1b. Whenever prompt tokens are evicted, also trim the raw
        # prompt_token_ids list so it stays consistent with
        # num_prompt_tokens (decremented in step 3). Required for:
        #   - Pre-prefill admission (old code path).
        #   - Post-prefill deferred admission (Path 2: evict-after-prefill).
        #   - Mid-gen eviction whose range overlaps the prompt (this can
        #     fire during chunked prefill before output starts).
        # The `trim_prompt_token_ids` flag is preserved for API
        # compatibility but is now effectively unused — we infer from
        # prompt_tokens_evicted > 0 whether to trim the list. Without this
        # trim, `len(prompt_token_ids) > num_prompt_tokens` and the
        # scheduler may try to re-prefill the un-trimmed tail (whose KV
        # blocks were just evicted), causing engine stall or wrong logits.
        if prompt_tokens_evicted > 0:
            assert request.prompt_token_ids is not None, (
                "prompt-overlapping eviction requires prompt_token_ids"
            )
            del request.prompt_token_ids[
                evict_start : evict_start + prompt_tokens_evicted
            ]

        # 2. Trim output_token_ids for any evicted output tokens.
        if output_tokens_evicted > 0:
            out_start = max(0, evict_start - prompt_len)
            del request._output_token_ids[
                out_start : out_start + output_tokens_evicted
            ]
        request.output_token_ids = ConstantList(request._output_token_ids)

        # 3. Adjust num_prompt_tokens if prompt tokens were evicted.
        if prompt_tokens_evicted > 0:
            request.num_prompt_tokens -= prompt_tokens_evicted

        # 4 & 5. Decrement num_computed_tokens by the OVERLAP between the
        # cached K/V prefix [0, num_computed) and the eviction range
        # [evict_start, evict_end). Shift position_offset by total_evicted
        # so post-eviction physical positions rotate at the same absolute
        # RoPE positions they had pre-eviction.
        #
        # Cases:
        #   - Cold cache (num_computed == 0): no KV exists. Skip both —
        #     the trimmed prompt is freshly prefilled with positions
        #     [0, post_prompt_len) and position_offset stays 0. Trainer
        #     mirror also fresh-prefills the trimmed prompt.
        #   - Partial prefix-cache hit (num_computed <= evict_start):
        #     overlap == 0. Cached K/V at physical [0, num_computed)
        #     survives the splice unchanged (those blocks are not in the
        #     evict range). num_computed_tokens stays the same;
        #     position_offset bumps by total_evicted so subsequent
        #     prefill of physical [num_computed, post_prompt_len) gets
        #     RoPE for the original-absolute positions of the kept-
        #     suffix tokens.
        #   - Cross-range cache (evict_start < num_computed < evict_end):
        #     cached K/V at physical [evict_start, num_computed) was in
        #     freed blocks. overlap = num_computed - evict_start.
        #     Decrement reduces num_computed to evict_start.
        #   - Warm cache (num_computed >= evict_end), e.g. post-prefill
        #     deferred admission or mid-gen eviction:
        #     overlap == total_evicted. Decrement matches the old "always
        #     subtract total_evicted" behavior; shift keeps decode-Q in
        #     the original absolute frame.
        had_kv = request.num_computed_tokens > 0
        if had_kv:
            overlap = max(
                0,
                min(request.num_computed_tokens, evict_end) - evict_start,
            )
            request.num_computed_tokens -= overlap
            request.position_offset += total_evicted

        # 6. Turn mode: drop the markers for the evicted turns and shift
        # the rest left by total_evicted. Index-based — robust against
        # inward-snap edge cases where a turn-end marker happens to land
        # exactly on the snapped boundary.
        #
        # Original positions layout (e=num_turns_evicted_before_this_event):
        #   [0]      end of system prompt (KEEP)
        #   [1]      end of U_{e+1}                  ──┐
        #   [2]      end of A_{e+1} = end of turn e+1   │ DROP these
        #   ...                                          │ (2*stride entries
        #   [2*s-1]  end of U_{e+s}                      │  total)
        #   [2*s]    end of A_{e+s} = end of turn e+s  ──┘
        #   [2*s+1]  end of U_{e+s+1} (SHIFT by -total_evicted)
        #   ...
        # Invariant: positions[0] (end of system prompt) is preserved
        # bit-for-bit — system prompt is never evicted.
        if self._compaction_max_turns > 0:
            old = request.turn_end_positions
            kept_tail = [p - total_evicted for p in old[2 * stride_used + 1:]]
            request.turn_end_positions = [old[0]] + kept_tail
            request.last_turn_scan_pos = len(request._all_token_ids)
            request.num_turns_evicted = num_turns_evicted_after

        # 7. Prefix-cache rebuild. After eviction, request.block_hashes is
        # stale (refers to the pre-eviction token sequence + evicted parent
        # hashes) and the kept blocks are registered in
        # cached_block_hash_to_block under those stale hashes — so a future
        # request whose prompt matches the post-eviction sequence would
        # compute a fresh hash chain from NONE_HASH and miss every kept
        # block. Rebuild the chain and re-register the survivors under
        # their new hashes so prefix caching can actually hit on the kept
        # window. No-op when prefix caching is disabled.
        self._rehash_after_eviction(request)

        return prompt_tokens_evicted, output_tokens_evicted

    def _rehash_after_eviction(self, request: Request) -> None:
        """Rebuild the block-hash chain after eviction trimmed
        request._all_token_ids, and re-register surviving blocks in the
        prefix-cache map under their new hashes.

        Required for prefix caching to hit on a future request whose
        prompt matches the post-eviction sequence. Without this:

          1. request.block_hashes encodes the pre-eviction chain. Block
             N's hash references h(block_{N-1}) — but block_{N-1} may
             have been freed by this eviction, and h(block_{N-1}) is
             not what a fresh request walking the chain from NONE_HASH
             would compute over the post-eviction tokens.
          2. KVCacheBlocks for surviving blocks still carry their
             pre-eviction hashes in `block_hash` and entries in
             `cached_block_hash_to_block` — so even if a future request
             SOMEHOW computed the same stale hash, it would be matching
             on a chain that no longer reflects the physical state.

        Implementation:
          a. For each single-type manager with caching enabled, pop the
             stale hash entries for the request's surviving blocks from
             `cached_block_hash_to_block`, reset each block's stored
             hash to None, and zero the manager's
             `num_cached_block[req_id]` counter so the next
             `cache_blocks()` call re-registers them.
          b. Clear `request.block_hashes` and call `update_block_hashes`
             to rebuild the chain from scratch over the post-eviction
             token sequence.

        No-op when prefix caching is globally disabled (the common
        case today; this method exists to unlock enabling it).
        """
        if not self.cache_config.enable_prefix_caching:
            return
        for mgr in self.kv_cache_manager.coordinator.single_type_managers:
            block_pool = getattr(mgr, "block_pool", None)
            if block_pool is None or not block_pool.enable_caching:
                continue
            kept_blocks = mgr.req_to_blocks.get(request.request_id, [])
            for blk in kept_blocks:
                if blk.block_hash is not None:
                    # pop is a no-op if the entry doesn't match the
                    # block_id (defensive against hash collisions).
                    block_pool.cached_block_hash_to_block.pop(
                        blk.block_hash, blk.block_id
                    )
                    blk.reset_hash()
            # Tell the manager that nothing is cached for this request
            # — so its next cache_blocks() call re-registers the kept
            # blocks under the freshly-computed hashes.
            if request.request_id in mgr.num_cached_block:
                mgr.num_cached_block[request.request_id] = 0
        # Rebuild the request's hash chain over the (post-trim)
        # all_token_ids. block_hashes is a typed wrapper around a list;
        # use .clear() / .extend() rather than rebinding the attribute.
        request.block_hashes.clear()
        request.update_block_hashes()

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            self._clear_reprefill_after_flush_if_prompt_ready(
                request, reason="update-after-schedule"
            )
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        # Mid-call admission eviction: at this point session.num_computed_tokens
        # is the count of tokens whose K already lives in cache from the
        # previous call, and session.num_prompt_tokens has just grown to
        # include the newly-appended content. Fire admission eviction NOW —
        # the splice mutates block_table + position_offset BEFORE the new
        # content is scheduled for prefill on the next step, so the new
        # content's K vectors are written under the post-eviction state.
        # See plans/connect_admission_events_to_trainer.md "Operative intent".
        session.session_prefill_boundary = session.num_computed_tokens
        if self._compaction_max_turns > 0:
            self._run_admission_eviction_loop(session)

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()
        rebuild_req_ids: set[str] = set()
        position_offsets: dict[str, int] = {}
        prompt_lengths: dict[str, int] = {}
        protected_prefix_lens: dict[str, int] = {}
        hidden_kv_num_tokens: dict[str, int] = {}

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)

            # Compaction rebuild: MUST come before scheduled_in_prev_step check
            # to ensure trimmed all_token_ids and full block_ids are sent.
            if req.needs_rebuild:
                rebuild_req_ids.add(req_id)
                position_offsets[req_id] = req.position_offset
                prompt_lengths[req_id] = req.num_prompt_tokens
                # Ship the request's protected_prefix_len so the worker
                # can apply the 2-piece position rule (Phase D). Static
                # per request after admission, so we only send it on
                # rebuild events (= when admission fired or other state
                # mutations require a worker resync).
                if self._compaction_enabled:
                    protected_prefix_lens[req_id] = (
                        self._worker_protected_prefix_len(req)
                    )
                all_token_ids[req_id] = req.all_token_ids.copy()
                # Send full block_ids (not just new) for rebuild.
                full_block_ids = tuple(
                    [blk.block_id for blk in group]
                    for group in self.kv_cache_manager.coordinator.get_blocks(
                        req_id
                    )
                )
                hidden_block_ids, hidden_tokens, _ = (
                    self._managed_context_active_hidden_kv(
                        req_id
                    )
                )
                if hidden_tokens:
                    hidden_kv_num_tokens[req_id] = hidden_tokens
                if hidden_block_ids:
                    managers = (
                        self.kv_cache_manager.coordinator.single_type_managers
                    )
                    full_block_ids = tuple(
                        list(
                            full_block_ids[i][
                                : min(
                                    len(full_block_ids[i]),
                                    (
                                        self._worker_protected_prefix_len(req)
                                        + managers[i].block_size
                                        - 1
                                    )
                                    // managers[i].block_size,
                                )
                            ]
                        )
                        + list(hidden_block_ids[i])
                        + list(
                            full_block_ids[i][
                                min(
                                    len(full_block_ids[i]),
                                    (
                                        self._worker_protected_prefix_len(req)
                                        + managers[i].block_size
                                        - 1
                                    )
                                    // managers[i].block_size,
                                ) :
                            ]
                        )
                        for i in range(len(full_block_ids))
                    )
                new_block_ids.append(full_block_ids)
                req.needs_rebuild = False
            else:
                scheduled_in_prev_step = (
                    req_id in self.prev_step_scheduled_req_ids
                )
                if idx >= num_running_reqs:
                    assert not scheduled_in_prev_step
                    resumed_req_ids.add(req_id)
                if not scheduled_in_prev_step:
                    all_token_ids[req_id] = req.all_token_ids.copy()
                new_block_ids.append(
                    req_to_new_blocks[req_id].get_block_ids(allow_none=True)
                )

            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
            rebuild_req_ids=rebuild_req_ids,
            position_offsets=position_offsets,
            prompt_lengths=prompt_lengths,
            protected_prefix_lens=protected_prefix_lens,
            hidden_kv_num_tokens=hidden_kv_num_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        managed_context_transfer_output = (
            model_runner_output.managed_context_transfer_output
        )
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        if self._pending_engine_core_outputs:
            for client_index, pending_outputs in (
                self._pending_engine_core_outputs.items()
            ):
                outputs[client_index].extend(pending_outputs)
            self._pending_engine_core_outputs.clear()
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        if managed_context_transfer_output:
            self._update_from_managed_context_transfer_finished(
                managed_context_transfer_output
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            routed_experts = None
            finish_reason = None
            padding_token_ids_for_output: list[int] | None = None
            if stopped:
                routed_experts = self._get_routed_experts(request)

                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()

                # Compaction finish commit: when enabled, defer the actual
                # free + worker-finish-notification by one step so the next
                # `schedule()` iter can forward any still-uncomputed final
                # sampled token and optional filler padding. This ensures
                # blocks are not published to the prefix cache until their
                # K/V has really been materialized.
                needs_final_token_forward = (
                    request.num_computed_tokens < request.num_tokens
                )
                needs_block_align_forward = (
                    request.num_tokens % self._compaction_block_size != 0
                    if self._compaction_block_size > 0
                    else False
                )
                auto_pad_active = (
                    self._compaction_block_aligned_finish
                    and self.cache_config.enable_prefix_caching
                    and self._compaction_block_size > 0
                    and not request.padding_pending
                    and status_before_stop == RequestStatus.RUNNING
                    and (
                        needs_final_token_forward
                        or needs_block_align_forward
                    )
                    and not request.streaming_queue
                )
                if auto_pad_active:
                    pad_len = (
                        self._compaction_block_size
                        - (request.num_tokens % self._compaction_block_size)
                    ) % self._compaction_block_size
                    pre_pad_num_tokens = request.num_tokens
                    if pad_len > 0:
                        request.append_padding_token_ids(
                            self._compaction_filler_token_id,
                            pad_len,
                        )
                        # Surface the padding ids on this step's output.
                        # The orchestrator forwards them to the trainer
                        # (which appends them to its pre-trim K-cache
                        # contribution for THIS call) and to vLLM's prefix
                        # cache lookup for the NEXT call (which inherits
                        # these blocks).
                        padding_token_ids_for_output = [
                            self._compaction_filler_token_id
                        ] * pad_len
                    else:
                        padding_token_ids_for_output = []
                    request._pending_auto_pad_finish_reason = finish_reason
                    request._pending_auto_pad_stop_reason = request.stop_reason
                    request._pending_auto_pad_routed_experts = routed_experts
                    request._pending_auto_pad_padding_token_ids = (
                        padding_token_ids_for_output
                    )
                    # Do not let the client observe final completion until
                    # the internal tail/padding forward has completed and the
                    # final blocks are prefix-cache visible. The output
                    # processor still consumes this step's sampled token ids
                    # and logprobs, but the OpenAI response is held open until
                    # the finalize branch emits the real finish_reason.
                    finish_reason = None
                    padding_token_ids_for_output = None
                    # Stash the original finish status so we can restore
                    # it after the padding step (FINISHED_STOPPED vs
                    # FINISHED_LENGTH_CAPPED is user-visible).
                    request._pending_finish_status = request.status
                    request.padding_pending = True
                    request.status = RequestStatus.RUNNING
                    # CRITICAL: force the next schedule() iter to ship
                    # the UPDATED all_token_ids (incl. the padding ids
                    # we just appended) to the worker. Without this,
                    # `_make_cached_request_data` treats the request
                    # as already-scheduled (it WAS scheduled in this
                    # step) and skips the all_token_ids copy, so the
                    # worker's token_ids_cpu still holds stale slots
                    # at [pre_pad_num_tokens..num_tokens) and the
                    # auto-pad forward writes K for the wrong tokens.
                    # The K-dump diagnostic (debug/compare_kv.py)
                    # confirmed this: V's K at the auto-pad slots was
                    # not the filler token's K at any plausible
                    # position. Setting needs_rebuild=True routes
                    # through the rebuild branch in
                    # `_make_cached_request_data`, which always
                    # re-ships all_token_ids verbatim.
                    request.needs_rebuild = True
                    self.prev_step_scheduled_req_ids.discard(
                        request.request_id
                    )
                    logger.info(
                        "[COMPACT/auto-pad] req=%s appended %d filler "
                        "tokens (pre=%d post=%d, block=%d); next step "
                        "will forward the pending tail before publishing "
                        "the final block in the prefix cache",
                        request.request_id[:8],
                        pad_len,
                        pre_pad_num_tokens,
                        request.num_tokens,
                        self._compaction_block_size,
                    )
                    # Defer final output via routed_experts /
                    # finish_reason captured above. Skip
                    # _handle_stopped_request and _free_request — the
                    # request stays in self.running for the next schedule()
                    # iter to forward the pending tail. Don't add to
                    # stopped_running_reqs (we want it kept). Don't add to
                    # finished_req_ids (we want the worker to keep its
                    # input_batch entry).
                else:
                    finished = self._handle_stopped_request(request)
                    if finished:
                        kv_transfer_params = self._free_request(request)

                    if status_before_stop == RequestStatus.RUNNING:
                        stopped_running_reqs.add(request)
                    else:
                        stopped_preempted_reqs.add(request)

            # Finalize compaction auto-pad: if the request is in the
            # padding step AND num_computed_tokens has now caught up to
            # num_tokens (i.e. the filler forward completed this step),
            # transition to the original FINISHED status and free. The
            # EngineCoreOutput for this request was already emitted at
            # the prior (stop) step — there is nothing to add here, so
            # skip the rest of the loop iteration to avoid running
            # turn-boundary scans / compaction passes on a now-freed
            # request (whose blocks list is empty and would assert).
            if (
                request.padding_pending
                and request.num_computed_tokens >= request.num_tokens
            ):
                final_finish_reason = getattr(
                    request,
                    "_pending_auto_pad_finish_reason",
                    request.get_finished_reason(),
                )
                final_stop_reason = getattr(
                    request,
                    "_pending_auto_pad_stop_reason",
                    request.stop_reason,
                )
                final_routed_experts = getattr(
                    request,
                    "_pending_auto_pad_routed_experts",
                    None,
                )
                final_padding_token_ids = getattr(
                    request,
                    "_pending_auto_pad_padding_token_ids",
                    None,
                )
                logger.info(
                    "[COMPACT/auto-pad] req=%s finalizing: num_computed=%d "
                    "num_tokens=%d; restoring finish status",
                    request.request_id[:8],
                    request.num_computed_tokens,
                    request.num_tokens,
                )
                request.padding_pending = False
                if request._pending_finish_status is not None:
                    request.status = request._pending_finish_status
                    request._pending_finish_status = None
                else:
                    request.status = RequestStatus.FINISHED_STOPPED
                compaction_events = (
                    list(request.compaction_events)
                    if request.compaction_events
                    else None
                )
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=[],
                        finish_reason=final_finish_reason,
                        stop_reason=final_stop_reason,
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        num_external_computed_tokens=request.num_external_computed_tokens,
                        routed_experts=final_routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                        compaction_events=compaction_events,
                        padding_token_ids=final_padding_token_ids,
                    )
                )
                for attr in (
                    "_pending_auto_pad_finish_reason",
                    "_pending_auto_pad_stop_reason",
                    "_pending_auto_pad_routed_experts",
                    "_pending_auto_pad_padding_token_ids",
                ):
                    if hasattr(request, attr):
                        delattr(request, attr)
                finished = self._handle_stopped_request(request)
                if finished:
                    self._free_request(request)
                stopped_running_reqs.add(request)
                continue
            if request.padding_pending:
                logger.info(
                    "[COMPACT/auto-pad] req=%s still pending: "
                    "num_computed=%d num_tokens=%d (waiting for forward)",
                    request.request_id[:8],
                    request.num_computed_tokens,
                    request.num_tokens,
                )

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                if not ok:
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

            # --- Turn-boundary scan (turn-mode compaction) ---
            # Keep `request.turn_end_positions` in sync with newly
            # appended tokens. Previously this scan was triggered as a
            # side-effect of `_should_compact` in the mid-gen block
            # below; the Phase-D `_pending_admission_compaction_ids`
            # gate on that block prevents it from firing for reqs that
            # never get a plan emitted (e.g. live_turns < max_turns).
            # Call directly so the scan happens regardless of which
            # compaction branch (or neither) fires.
            if self._compaction_enabled and self._compaction_max_turns > 0:
                self._scan_new_turn_boundaries(request)

            # KV cache compaction: admission eviction has already fired
            # inline inside `schedule()` via
            # `_apply_inline_admission_eviction`. Drain any residual
            # pending ids whose prefill has now completed without a plan
            # being emitted (e.g. live_turns dropped below max_turns
            # mid-flight), so the mid-gen branch below — gated on
            # `not in _pending` — can fire for them later.
            if (
                request.request_id in self._pending_admission_compaction_ids
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                self._pending_admission_compaction_ids.discard(request.request_id)

            # --- KV cache compaction: mid-gen sliding window ---
            # After tokens are appended and stop is checked, compact if needed.
            # Must be AFTER stop check (don't compact finished requests).
            # Excluded for reqs still in _pending_admission_compaction_ids:
            # those will be handled by the admission path on the step
            # where their plan finally emits. Also excluded for reqs
            # that were just finalized in the auto-pad path: their
            # status is FINISHED_* and their blocks are already freed,
            # so any compaction attempt would index out of bounds.
            if (
                not stopped
                and not request.is_finished()
                and self._compaction_enabled
                and request.num_output_placeholders == 0
                and request.request_id not in self._pending_admission_compaction_ids
            ):
                while self._should_compact(request):
                    tokens_evicted = self._compact_request(request)
                    if tokens_evicted == 0:
                        break
                    request.needs_rebuild = True
                if request.needs_rebuild:
                    # Ensure _make_cached_request_data sends trimmed
                    # all_token_ids by removing from prev_step set.
                    self.prev_step_scheduled_req_ids.discard(
                        request.request_id
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                # Add EngineCoreOutput for this Request.
                # Send the full cumulative compaction_events list (overwrite
                # semantics at the client). Events are append-only and the
                # list is short (one per stride's worth of generation), so
                # the per-step overhead is negligible. Only included when
                # non-empty to keep non-compaction outputs unchanged.
                compaction_events = (
                    list(request.compaction_events)
                    if request.compaction_events
                    else None
                )
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        num_external_computed_tokens=request.num_external_computed_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                        compaction_events=compaction_events,
                        padding_token_ids=padding_token_ids_for_output,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                if os.environ.get("KVE_DIAG_OUTPUT_ORPHANS") == "1":
                    emitted_req_ids = {
                        output.request_id for output in outputs.get(client_index, ())
                    }
                    finished_without_output = finished_set - emitted_req_ids
                    if finished_without_output:
                        logger.warning(
                            "[KVE-OUTPUT-ORPHAN-SCHED] client=%d "
                            "finished_without_output=%s finished=%s outputs=%s "
                            "still_tracked=%s",
                            client_index,
                            sorted(finished_without_output),
                            sorted(finished_set),
                            sorted(emitted_req_ids),
                            sorted(
                                req_id
                                for req_id in finished_without_output
                                if req_id in self.requests
                            ),
                        )
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            if self._request_kv_swap_should_prefer_waiting_queue():
                return self.waiting
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    def _get_routed_experts(self, request: Request) -> np.ndarray | None:
        if not self.vllm_config.model_config.enable_return_routed_experts:
            return None

        kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
        block_ids = kv_blocks.get_block_ids()[self.routed_experts_attn_gid]
        num_tokens = request.num_tokens - 1

        # compute slot mapping using attention group's block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        num_blocks = len(block_ids)
        attn_group = self.kv_cache_config.kv_cache_groups[self.routed_experts_attn_gid]
        block_size = attn_group.kv_cache_spec.block_size

        # generate block offsets
        block_offsets = np.arange(0, block_size)

        # compute slot mapping: slot = block_id * block_size + offset
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_array.reshape((num_blocks, 1)) * block_size
        ).flatten()[:num_tokens]

        return self.routed_experts_reader.get_routed_experts(indices=slot_mapping)

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            # Mark the request for in-step admission eviction at its
            # prefill-completing step. `_apply_inline_admission_eviction`
            # in `schedule()` consumes the pending set: when prefill
            # completes this step AND `live_turns >= max_turns`, it
            # frees the to-be-evicted blocks BEFORE the worker's prefill
            # kernel fires, so `new_user_fragment` K/V is computed under
            # attention over the kept context only. Single forward.
            if (
                self._compaction_max_turns > 0
                and request.prompt_token_ids is not None
            ):
                self._pending_admission_compaction_ids.add(request.request_id)
            self._reserve_managed_context_restore_request(request)
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            drop_request_after_free = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                if request.request_id in self._request_kv_swaps:
                    (
                        delay_free_blocks,
                        drop_request_after_free,
                    ) = self._release_request_kv_swap_for_finish(
                        request.request_id,
                        "finish-waiting-request",
                    )
                elif request.request_id in self._managed_context_pending_loads:
                    pending = self._managed_context_pending_loads[
                        request.request_id
                    ]
                    event_not_submitted = (
                        pending.event_id
                        in self._managed_context_load_events_to_submit
                    )
                    event_finished = (
                        request.request_id
                        in self._managed_context_finished_load_req_ids
                    )
                    delay_free_blocks = (
                        not event_not_submitted and not event_finished
                    )
                    if not delay_free_blocks:
                        self._release_managed_context_pending_load(
                            request.request_id,
                            "finish-waiting-request",
                            only_if_safe=False,
                        )
                else:
                    delay_free_blocks = (
                        request.request_id not in self.finished_recving_kv_req_ids
                    )
                    self.finished_recving_kv_req_ids.discard(request.request_id)
                    self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)
            if drop_request_after_free:
                self.requests.pop(request.request_id, None)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self._release_managed_context_restore_reservation(
            request_id, "free-request"
        )
        self._release_managed_context_deferred_restore(
            request_id, "free-request"
        )
        if request_id in self._managed_context_pending_loads:
            pending_load_released = self._release_managed_context_pending_load(
                request_id,
                "free-request",
                only_if_safe=True,
            )
            delay_free_blocks |= not pending_load_released
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)
        else:
            self._release_managed_context_active_restore(
                request_id, "delay-free-request"
            )

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self._release_managed_context_restore_reservation(
            request.request_id, "request-finished"
        )
        self._release_managed_context_pending_load(
            request.request_id, "request-finished", only_if_safe=False
        )
        self._release_managed_context_deferred_restore(
            request.request_id, "request-finished"
        )
        self._release_managed_context_active_restore(
            request.request_id, "request-finished"
        )
        # KV compaction: ensure any block that became full on the request's
        # final decode step gets registered in the prefix-cache pool before
        # its blocks are freed. Otherwise the last block — whose state
        # transitioned from partial to full via the in-step kernel write
        # — is lost because `allocate_slots` for that step computed
        # `num_full_blocks` from the pre-decode `request.num_tokens` and
        # the request finishes before a subsequent `allocate_slots`
        # observes the completed block. Manifests as future calls within
        # the same rollout chain failing to prefix-cache-hit the last
        # block of the prior writer (see find_longest_cache_hit TRACE
        # HASH-MISS at block_idx = N-1 where N is the writer's final
        # block count). The auto-pad path inadvertently registered the
        # block via its extra scheduling iter, masking this for non-
        # block-aligned writers; block-aligned writers (auto-pad no-op)
        # exposed the bug. Safe: cache_blocks is idempotent when blocks
        # are already cached and only fires when prefix caching is on.
        if self.cache_config.enable_prefix_caching and request.num_tokens > 0:
            if os.environ.get("KVE_TRACE_CACHE_COMMIT") == "1":
                logger.warning(
                    "[TRACE-FREE-CACHE-BEFORE-FREE] req=%s num_tokens=%d "
                    "num_computed=%d num_prompt=%d position_offset=%d "
                    "padding_pending=%d",
                    request.request_id[:8],
                    request.num_tokens,
                    request.num_computed_tokens,
                    request.num_prompt_tokens,
                    request.position_offset,
                    int(getattr(request, "padding_pending", False)),
                )
            self.kv_cache_manager.cache_blocks(
                request, request.num_tokens
            )
            self._pin_phase4_request_blocks(request)
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            swap_suspended = self._request_kv_swap_suspended
            self._request_kv_swap_suspended = True
            try:
                while self.running:
                    request = self.running.pop()
                    self._preempt_request(request, timestamp)
                    if request.is_finished():
                        continue
                    # NOTE(zhuohan): For async scheduling, we need to discard the latest
                    # output token on the fly to avoid a redundant repetitive output token.
                    request.num_output_placeholders = 0
                    request.discard_latest_async_tokens = True
            finally:
                self._request_kv_swap_suspended = swap_suspended

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        if self._managed_context_active_restores and not reset_running_requests:
            logger.warning(
                "[MANAGED-CONTEXT] refusing prefix-cache reset with active "
                "restored requests: %s",
                list(self._managed_context_active_restores),
            )
            return False
        if self._managed_context_pending_loads:
            logger.warning(
                "[MANAGED-CONTEXT] refusing prefix-cache reset with pending "
                "CPU reloads: %s",
                list(self._managed_context_pending_loads),
            )
            return False
        if self._request_kv_swaps:
            logger.warning(
                "[REQUEST-KV-SWAP] refusing prefix-cache reset with request "
                "KV swaps: %s",
                list(self._request_kv_swaps),
            )
            return False
        in_flight_store_events = [
            event_id
            for event_id in self._managed_context_store_event_to_span
            if event_id not in self._managed_context_store_events_to_submit
        ]
        if in_flight_store_events:
            logger.warning(
                "[MANAGED-CONTEXT] refusing prefix-cache reset with pending "
                "CPU archive stores: %s",
                in_flight_store_events,
            )
            return False

        # Phase4 pins are scheduler-local references into the prefix cache. A
        # prefix-cache reset invalidates the carried retained state, so release
        # all pins before resetting the normal cache.
        for request_id in list(self._managed_context_active_restores):
            self._release_managed_context_active_restore(
                request_id, "reset-prefix-cache"
            )
        for request_id in list(self._managed_context_deferred_restores):
            self._release_managed_context_deferred_restore(
                request_id, "reset-prefix-cache"
            )
        self._release_all_managed_context_spans("reset-prefix-cache")
        self._release_all_phase4_pins("reset-prefix-cache")
        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting) + len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            encoder_cache_usage=self._get_encoder_cache_usage(),
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def _get_encoder_cache_usage(self) -> float:
        """Get encoder cache usage as a fraction (0.0 to 1.0)."""
        ecm = self.encoder_cache_manager
        if ecm.cache_size == 0:
            return 0.0
        used_slots = ecm.cache_size - ecm.num_free_slots
        return used_slots / ecm.cache_size

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

            # Count the number of prefix cached tokens.
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = request.num_computed_tokens

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(
        self,
        request: Request,
        *,
        token_budget: int | None = None,
    ) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            if request.request_id in self._request_kv_swaps:
                return self._try_progress_request_kv_swap(
                    request,
                    token_budget=token_budget,
                )
            if request.request_id in self._managed_context_pending_loads:
                if (
                    request.request_id
                    not in self._managed_context_finished_load_req_ids
                ):
                    return False
                self._complete_managed_context_pending_load(request)
                if request.num_preemptions:
                    request.status = RequestStatus.PREEMPTED
                else:
                    request.status = RequestStatus.WAITING
                return True

            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not (structured_output_req and structured_output_req.grammar):
                return False
            request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_managed_context_transfer_finished(
        self, output: ManagedContextTransferOutput
    ) -> None:
        for event_id in output.completed_store_event_ids:
            if self._complete_phase4_pin_cpu_store(event_id):
                continue
            span = self._managed_context_store_event_to_span.pop(event_id, None)
            if span is None:
                self._complete_request_kv_swap_store(event_id)
                continue
            released_blocks = self._release_managed_context_span_gpu_entries(span)
            span.offload_event_id = None
            if span.status == "expired":
                self._managed_context_free_span_cpu_blocks(span)
            else:
                span.status = "cpu_offloaded"
            if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
                logger.warning(
                    "[MANAGED-CONTEXT-STORE-DONE] trace=%s span=%s event=%d "
                    "released_gpu_blocks=%d status=%s cpu_blocks=%s",
                    span.trace_id,
                    span.span_id,
                    event_id,
                    released_blocks,
                    span.status,
                    span.cpu_block_ids_by_group,
                )
        if output.completed_store_event_ids:
            self._retry_managed_context_gpu_pinned_offloads("store-done")

        for event_id in output.completed_load_event_ids:
            if self._complete_phase4_pin_cpu_load(event_id):
                continue
            request_id = self._managed_context_load_event_to_request_id.pop(
                event_id, None
            )
            if request_id is None:
                request_id = self._request_kv_swap_load_event_to_request_id.pop(
                    event_id, None
                )
                if request_id is not None:
                    self._request_kv_swap_finished_load_req_ids.add(request_id)
                    swap = self._request_kv_swaps.get(request_id)
                    if swap is not None and swap.status == "expired":
                        self._release_request_kv_swap(
                            request_id,
                            "expired-load-done",
                            release_gpu_entries=True,
                        )
                continue
            self._managed_context_finished_load_req_ids.add(request_id)
            request = self.requests.get(request_id)
            if request is None:
                self._release_managed_context_pending_load(
                    request_id, "load-done-missing-request", only_if_safe=False
                )
                continue
            if RequestStatus.is_finished(request.status):
                self._free_blocks(request)
            elif request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                if os.environ.get("KVE_TRACE_MANAGED_CONTEXT") == "1":
                    logger.warning(
                        "[MANAGED-CONTEXT-LOAD-DONE] req=%s event=%d",
                        request_id[:8],
                        event_id,
                    )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens
                request.num_external_computed_tokens -= num_affected_tokens
                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
