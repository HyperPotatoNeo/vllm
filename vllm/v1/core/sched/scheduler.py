# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
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
    AttentionMatchingCompactionResult,
    DraftTokenIds,
    KVConnectorOutput,
    ModelRunnerOutput,
    NoiseControlResult,
    ShuffleControlResult,
)
from vllm.v1.core.compaction.am_manager import AttentionMatchingKVCacheManager
from vllm.v1.core.compaction.am_runtime import (
    DEFAULT_TURN_SEPARATOR_TOKEN_ID_SEQUENCE,
    advance_attention_matching_turn_boundary,
)
from vllm.v1.core.compaction.am_prefix_cache import (
    AttentionMatchingPrefixCacheReplay,
    build_attention_matching_prefix_cache_key,
    build_attention_matching_turn_prefix_cache_replay,
    hash_attention_matching_tokens,
)
from vllm.v1.core.compaction.manager import CompactingKVCacheManager
from vllm.v1.core.compaction.shuffle_control import (
    NoiseEvent,
    ShuffleEvent,
    chunk_permutation,
)
from vllm.v1.core.compaction.types import CompactionEvent
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.utils import ConstantList
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


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
        self.prev_step_scheduled_req_ids: set[str] = set()

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
            if self.cache_config.compaction_window_size > 0:
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
        # AM compressed-prefix cache metadata needed for faithful trainer
        # replay. Prefix-cache blocks only store KV; the trainer also needs
        # the discrete OMP atom choices that produced synthetic KV.
        self.attention_matching_prefix_cache_selected_indices: dict[
            str, list[list[list[int]]]
        ] = {}
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
            compaction_strategy=self.cache_config.compaction_strategy,
            prefix_caching_mode=self.cache_config.prefix_caching_mode,
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
                "[COMPACT] enabled window=%d stride=%d",
                self.cache_config.compaction_window_size,
                self.cache_config.compaction_stride,
            )

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self._attention_matching_enabled = (
            self.cache_config.compaction_strategy == "attention_matching"
            and self.cache_config.compaction_window_size > 0
        )
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
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

        self.kv_cache_manager.new_step_starts()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

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
            # This is necessary when using spec decoding. Hidden AM tail
            # finalization is filler-only prefill whose sampled token is
            # discarded by the worker, so it may safely compute the final
            # position at max_model_len - 1 instead of reserving one more slot
            # for a visible sampled token.
            max_input_tokens = (
                self.max_model_len
                if request.prefix_cache_tail_finalizing
                else self.max_model_len - 1
            )
            num_new_tokens = min(
                num_new_tokens, max_input_tokens - request.num_computed_tokens
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
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
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
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
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

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
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

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    self._refresh_attention_matching_protected_prompt_len(request)
                    self._maybe_prepare_attention_matching_cross_turn_candidate(
                        request
                    )
                    while True:
                        # Get locally-cached tokens. AM-full cross-turn
                        # admission may retry progressively shallower replay
                        # states until it finds the deepest cached AM state.
                        new_computed_blocks, num_new_local_computed_tokens = (
                            self.kv_cache_manager.get_computed_blocks(request)
                        )
                        if not (
                            self._maybe_finalize_attention_matching_cross_turn_candidate(
                                request, num_new_local_computed_tokens
                            )
                        ):
                            break

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
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
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
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
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
                    break

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

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                self._maybe_privatize_attention_matching_blocks(
                    request, num_computed_tokens
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
                if (
                    self.cache_config.enable_prefix_caching
                    and request.num_cached_tokens > 0
                    and not request.skip_reading_prefix_cache
                    and not request.prefix_cache_read_hit_logged
                ):
                    logger.warning(
                        "[PrefixCache] read hit request %s cached_tokens=%d "
                        "prompt_tokens=%d mode=%s "
                        "attention_matching_enabled=%s am_keyed=%s key=%s",
                        request.request_id,
                        request.num_cached_tokens,
                        request.num_prompt_tokens,
                        self.cache_config.prefix_caching_mode,
                        self._attention_matching_enabled,
                        request.attention_matching_prefix_cache_key is not None,
                        (request.attention_matching_prefix_cache_key or "")[:16],
                    )
                    request.prefix_cache_read_hit_logged = True
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

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
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

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        # FIFO-compacted requests remain unrecoverable. Attention-matching
        # requests become resumable once the worker can snapshot the compacted
        # live KV state and later restore it onto fresh blocks.
        if (
            request.position_offset > 0
            and not (
                request.attention_matching_active
                and request.attention_matching_snapshot_version is not None
            )
        ):
            logger.warning(
                "Attempted to preempt compacted request %s "
                "(position_offset=%d) — aborting request instead.",
                request.request_id, request.position_offset,
            )
            self.kv_cache_manager.free(request)
            self.encoder_cache_manager.free(request)
            request.status = RequestStatus.FINISHED_ABORTED
            self.finished_req_ids.add(request.request_id)
            return
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        if request.attention_matching_active:
            logger.warning(
                "[AM] preempting compacted request %s at computed=%d "
                "(offset=%d version=%s)",
                request.request_id,
                request.num_computed_tokens,
                request.position_offset,
                request.attention_matching_snapshot_version,
            )
            request.attention_matching_restore_pending = True
        else:
            request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    # --- Compaction helpers ---

    def _should_compact(self, request: Request) -> bool:
        """Check if any KV cache group needs compaction for this request."""
        for mgr in self.kv_cache_manager.coordinator.single_type_managers:
            if isinstance(mgr, CompactingKVCacheManager) and mgr.needs_compaction(
                request.request_id,
                request.num_computed_tokens,
                request.num_prompt_tokens,
            ):
                return True
        return False

    def _compact_request(self, request: Request) -> int:
        """Compact a request: splice blocks, trim tokens, update state.

        After this, the request looks like a shorter sequence to all consumers.
        """
        total_evicted = 0
        compaction_mgr: "CompactingKVCacheManager | None" = None
        for mgr in self.kv_cache_manager.coordinator.single_type_managers:
            if not isinstance(mgr, CompactingKVCacheManager):
                continue
            tokens_evicted = mgr.compact_request(
                request.request_id, request.num_prompt_tokens
            )
            if tokens_evicted > 0:
                total_evicted = tokens_evicted
                compaction_mgr = mgr
                break  # Only one KV group for standard models

        if total_evicted == 0:
            return 0

        # Use the same block_size the compaction manager used to free physical
        # blocks. Under hybrid / multi-group KV caches, self.block_size (which
        # comes from a potentially different group) could diverge from the
        # compaction group's block_size, silently mis-aligning the trim.
        assert compaction_mgr is not None
        block_size = compaction_mgr.block_size

        # Record event BEFORE mutating state.
        event = CompactionEvent(
            num_output_tokens_at_compaction=request.num_total_generated,
            tokens_evicted=total_evicted,
            position_offset_after=request.position_offset + total_evicted,
            num_prompt_tokens=request.num_prompt_tokens,
            compaction_strategy="fifo",
        )
        request.compaction_events.append(event)

        # --- Mutate request to look like a shorter sequence ---
        # Physical block-level eviction in CompactingKVCacheManager drops blocks
        # starting at index prompt_blocks = ceil(prompt_len / block_size). That
        # block's first slot is at physical position prompt_aligned_len =
        # prompt_blocks * block_size, NOT at prompt_len. When prompt_len is not a
        # multiple of block_size, the last prompt block is partially filled with
        # prompt tokens; generation then fills the remainder of that same block
        # (vLLM allocates blocks via cdiv(num_tokens, block_size) and reuses the
        # last partial block for the first `block_size - (prompt_len % block_size)`
        # generated tokens). Those tokens share a block with the prompt tail and
        # are therefore NEVER evicted. Trimming the logical token lists from
        # prompt_len would delete those retained-forever tokens AND fail to
        # delete the actually-evicted ones; the lists would still match in count
        # but not in identity, leaving the logical view out of sync with the
        # physical KV cache (breaking GPU sample kernels that index all_token_ids
        # positionally: penalties, bad_words, prompt_logprob).
        prompt_len = request.num_prompt_tokens
        prompt_aligned_len = (
            (prompt_len + block_size - 1) // block_size
        ) * block_size
        # How many generated tokens live in the tail of the last prompt block.
        # These are never evicted and must be preserved across the trim.
        gen_tail_in_prompt_block = prompt_aligned_len - prompt_len

        # 1. Trim all_token_ids at the block-aligned boundary so the removed
        # identities match the physically evicted block range.
        del request._all_token_ids[
            prompt_aligned_len : prompt_aligned_len + total_evicted
        ]
        request.all_token_ids = ConstantList(request._all_token_ids)

        # 2. Trim output_token_ids at the same range, expressed in output
        # (post-prompt) coordinates. The first gen_tail_in_prompt_block entries
        # of _output_token_ids correspond to generated tokens living in the last
        # prompt block and are retained.
        del request._output_token_ids[
            gen_tail_in_prompt_block : gen_tail_in_prompt_block + total_evicted
        ]
        request.output_token_ids = ConstantList(request._output_token_ids)

        # 3. Reduce num_computed_tokens (guard against underflow)
        assert request.num_computed_tokens >= total_evicted, (
            f"Compaction underflow: num_computed={request.num_computed_tokens}, "
            f"evicting={total_evicted}"
        )
        request.num_computed_tokens -= total_evicted

        # 4. Update position offset
        request.position_offset += total_evicted

        return total_evicted

    @staticmethod
    def _hash_attention_matching_tokens(token_ids: list[int]) -> str:
        """Stable compact hash for token regions summarized by AM."""
        return hash_attention_matching_tokens(token_ids)

    @classmethod
    def _attention_matching_compacted_tokens_hash(
        cls,
        request: Request,
        *,
        protected_prefix_len: int,
        source_len: int,
        exact_kept_tokens: int,
    ) -> str:
        compact_start = protected_prefix_len
        compact_end = source_len - exact_kept_tokens
        return cls._hash_attention_matching_tokens(
            request._all_token_ids[compact_start:compact_end]
        )

    @classmethod
    def _build_attention_matching_prefix_cache_key(
        cls,
        request: Request,
        result: AttentionMatchingCompactionResult,
        *,
        source_len: int,
        target_len: int,
        position_offset_after: int,
        snapshot_version: int,
    ) -> str:
        """Build an AM-specific prefix-cache namespace for compacted KV.

        The visible synthetic prompt IDs are placeholders, so the hash must
        include the source history and AM parameters that produced the KV.
        """
        compacted_tokens_hash = cls._attention_matching_compacted_tokens_hash(
            request,
            protected_prefix_len=result.protected_prefix_len,
            source_len=source_len,
            exact_kept_tokens=result.exact_kept_tokens,
        )
        tail_signature = None
        if result.query_source != "random_queries":
            tail_signature = (
                snapshot_version,
                source_len,
                target_len,
            )
        return build_attention_matching_prefix_cache_key(
            cache_salt=request.cache_salt,
            protected_prefix_len=result.protected_prefix_len,
            synthetic_prefix_len=result.synthetic_prefix_len,
            compacted_tokens_hash=compacted_tokens_hash,
            query_source=result.query_source,
            max_queries_per_kv_head=result.max_queries_per_kv_head,
            query_seed=result.query_seed,
            zerobeta=result.zerobeta,
            parent_key=request.attention_matching_prefix_cache_key,
            parent_key_start=request.attention_matching_prefix_cache_key_start,
            position_offset_before=request.position_offset,
            position_offset_after=position_offset_after,
            tail_signature=tail_signature,
            forget_gate_enabled=result.forget_gate_enabled,
            forget_gate_alpha=result.forget_gate_alpha,
        )

    def _remember_attention_matching_selected_indices(
        self, result: AttentionMatchingCompactionResult
    ) -> None:
        if result.prefix_cache_key is not None and result.selected_indices is not None:
            self.attention_matching_prefix_cache_selected_indices[
                result.prefix_cache_key
            ] = result.selected_indices
        for replay_step in result.replay_steps or ():
            if not isinstance(replay_step, dict):
                continue
            key = replay_step.get("attention_matching_prefix_cache_key")
            selected = replay_step.get("attention_matching_selected_indices")
            if isinstance(key, str) and selected is not None:
                self.attention_matching_prefix_cache_selected_indices[key] = selected

    def _attention_matching_step_payload(
        self,
        *,
        source_len: int,
        target_len: int,
        protected_prefix_len: int,
        synthetic_prefix_len: int,
        exact_kept_tokens: int,
        query_seed: int,
        prefix_cache_key: str | None = None,
        selected_indices: list[list[list[int]]] | None = None,
        forget_gate_enabled: bool = False,
        forget_gate_alpha: float = 0.5,
        forget_gate_applied: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_len": source_len,
            "target_len": target_len,
            "protected_prefix_len": protected_prefix_len,
            "synthetic_prefix_len": synthetic_prefix_len,
            "exact_kept_tokens": exact_kept_tokens,
            "attention_matching_query_seed": query_seed,
        }
        if forget_gate_enabled:
            payload["attention_matching_forget_gate_enabled"] = True
            payload["attention_matching_forget_gate_alpha"] = float(
                forget_gate_alpha
            )
            payload["attention_matching_forget_gate_applied"] = bool(
                forget_gate_applied
            )
        if prefix_cache_key is not None:
            payload["attention_matching_prefix_cache_key"] = prefix_cache_key
        if selected_indices is not None:
            payload["attention_matching_selected_indices"] = selected_indices
        return payload

    @staticmethod
    def _attention_matching_event_has_selected_indices(
        event: CompactionEvent,
    ) -> bool:
        replay_steps = event.attention_matching_replay_steps or []
        if replay_steps:
            return all(
                isinstance(step, dict)
                and bool(step.get("attention_matching_selected_indices"))
                for step in replay_steps
            )
        return bool(event.attention_matching_selected_indices)

    def _apply_attention_matching_compaction(
        self,
        request: Request,
        result: AttentionMatchingCompactionResult,
    ) -> int:
        """Apply a worker-side AM compaction result to the scheduler state."""
        total_evicted = result.position_offset_delta
        target_len = (
            result.protected_prefix_len
            + result.synthetic_prefix_len
            + result.exact_kept_tokens
        )
        # Pre-sample AM reports the current computed boundary and returns no
        # sampled token. Legacy post-sample AM may report the KV cache that
        # existed before the sampled next token was appended, so scheduler state
        # can be one token ahead. Always use worker-reported source_len for
        # event semantics and exact-token slicing.
        source_len = result.source_len
        scheduler_source_len = request.num_computed_tokens
        if source_len > scheduler_source_len:
            raise RuntimeError(
                f"AM source_len exceeds scheduler num_computed for request "
                f"{request.request_id}: result.source_len={source_len}, "
                f"num_computed={scheduler_source_len}"
            )
        if total_evicted <= 0 or target_len >= source_len:
            return 0

        if result.selected_indices is None:
            raise RuntimeError(
                "AM compaction result is missing selected OMP indices for "
                f"request {request.request_id}; refusing to emit a "
                "non-faithful replay event."
            )
        replay_steps = result.replay_steps
        if replay_steps is None:
            replay_steps = [
                self._attention_matching_step_payload(
                    source_len=result.source_len,
                    target_len=result.target_len,
                    protected_prefix_len=result.protected_prefix_len,
                    synthetic_prefix_len=result.synthetic_prefix_len,
                    exact_kept_tokens=result.exact_kept_tokens,
                    query_seed=result.query_seed,
                    prefix_cache_key=result.prefix_cache_key,
                    selected_indices=result.selected_indices,
                    forget_gate_enabled=result.forget_gate_enabled,
                    forget_gate_alpha=result.forget_gate_alpha,
                    forget_gate_applied=result.forget_gate_applied,
                )
            ]

        self._remember_attention_matching_selected_indices(result)

        compaction_mgrs: list[AttentionMatchingKVCacheManager] = []
        for mgr in self.kv_cache_manager.coordinator.single_type_managers:
            if isinstance(mgr, AttentionMatchingKVCacheManager):
                compaction_mgrs.append(mgr)

        assert compaction_mgrs
        if self.cache_config.enable_prefix_caching:
            if self.cache_config.prefix_caching_mode == "am_unsafe":
                logger.warning(
                    "[PrefixCache][AM][UNSAFE] preserving existing prefix-cache "
                    "hash mappings before AM overwrite for request %s",
                    request.request_id,
                )
            else:
                request_block_ids_by_group = self.kv_cache_manager.get_block_ids(
                    request.request_id
                )
                request_block_ids = {
                    block_id
                    for group_block_ids in request_block_ids_by_group
                    for block_id in group_block_ids
                }
                if request_block_ids:
                    self.kv_cache_manager.evict_blocks(request_block_ids)
                    cleared_hashes = sum(
                        compaction_mgr.clear_request_block_hashes(
                            request.request_id
                        )
                        for compaction_mgr in compaction_mgrs
                    )
                    logger.warning(
                        "[PrefixCache][AM] evicted %d request block ids and "
                        "cleared %d block hashes from prefix cache before AM "
                        "finalize for request %s",
                        len(request_block_ids),
                        cleared_hashes,
                        request.request_id,
                    )
        for compaction_mgr in compaction_mgrs:
            compaction_mgr.finalize_compaction(request.request_id, target_len)
        for mgr in self.kv_cache_manager.coordinator.single_type_managers:
            if isinstance(mgr, AttentionMatchingKVCacheManager):
                continue
            mgr.finalize_attention_matching_compaction(
                request.request_id,
                target_len,
            )
        if self.cache_config.prefix_caching_mode == "am_unsafe":
            logger.warning(
                "[PrefixCache][AM][UNSAFE] preserving cached-block bookkeeping "
                "after AM overwrite for request %s",
                request.request_id,
            )
        else:
            for compaction_mgr in compaction_mgrs:
                compaction_mgr.set_cached_prefix_blocks(request.request_id, 0)

        source_logical_boundary = (
            source_len + request.position_offset - request.logical_prompt_len
        )
        if source_logical_boundary < 0:
            raise RuntimeError(
                f"AM logical compaction boundary is negative for request "
                f"{request.request_id}: source_len={source_len}, "
                f"position_offset={request.position_offset}, "
                f"logical_prompt_len={request.logical_prompt_len}"
            )
        if result.pre_sample:
            logprob_boundary = source_logical_boundary
            if request.num_total_generated != source_logical_boundary:
                raise RuntimeError(
                    f"Pre-sample AM must compact exactly at the current "
                    f"generated-token boundary for request {request.request_id}: "
                    f"num_generated={request.num_total_generated}, "
                    f"source_boundary={source_logical_boundary}, "
                    f"source_len={source_len}."
                )
        else:
            logprob_boundary = request.num_total_generated

        if (not result.pre_sample) and logprob_boundary <= source_logical_boundary:
            raise RuntimeError(
                f"AM logprob boundary must be after the compacted source "
                f"boundary for request {request.request_id}: "
                f"logprob_boundary={logprob_boundary}, "
                f"source_boundary={source_logical_boundary}, "
                f"source_len={source_len}, num_generated="
                f"{request.num_total_generated}."
            )

        version_delta = max(len(result.replay_steps or ()), 1)
        next_version = (
            version_delta
            if request.attention_matching_snapshot_version is None
            else request.attention_matching_snapshot_version + version_delta
        )
        am_prefix_cache_key: str | None = None
        if (
            self.cache_config.enable_prefix_caching
            and self.cache_config.prefix_caching_mode == "am_full"
        ):
            am_prefix_cache_key = result.prefix_cache_key
            if am_prefix_cache_key is None:
                am_prefix_cache_key = self._build_attention_matching_prefix_cache_key(
                    request,
                    result,
                    source_len=source_len,
                    target_len=target_len,
                    position_offset_after=request.position_offset + total_evicted,
                    snapshot_version=next_version,
                )

        event = CompactionEvent(
            # New AM events are pre-sample: vLLM discards the provisional
            # sample and recomputes the boundary token under the compacted KV.
            # Older post-sample events remain supported for compatibility.
            num_output_tokens_at_compaction=logprob_boundary,
            tokens_evicted=total_evicted,
            position_offset_after=request.position_offset + total_evicted,
            num_prompt_tokens=request.logical_prompt_len,
            compaction_strategy="attention_matching",
            source_len=result.source_len,
            target_len=result.target_len,
            protected_prefix_len=result.protected_prefix_len,
            synthetic_prefix_len=result.synthetic_prefix_len,
            exact_kept_tokens=result.exact_kept_tokens,
            attention_matching_query_source=result.query_source,
            attention_matching_max_queries_per_kv_head=(
                result.max_queries_per_kv_head
            ),
            attention_matching_query_seed=result.query_seed,
            attention_matching_zerobeta=result.zerobeta,
            attention_matching_pre_sample=result.pre_sample,
            attention_matching_replay_steps=replay_steps,
            attention_matching_selected_indices=result.selected_indices,
            attention_matching_forget_gate_enabled=result.forget_gate_enabled,
            attention_matching_forget_gate_alpha=result.forget_gate_alpha,
            attention_matching_forget_gate_applied=result.forget_gate_applied,
        )
        request.compaction_events.append(event)

        kept_exact_start = source_len - result.exact_kept_tokens
        if result.physical_token_ids is not None:
            retained_token_ids = list(result.physical_token_ids)
            prompt_cut = result.protected_prefix_len + result.synthetic_prefix_len
            protected_prompt_token_ids = retained_token_ids[: result.protected_prefix_len]
            kept_exact_token_ids = retained_token_ids[prompt_cut:]
        else:
            protected_prompt_token_ids = request._all_token_ids[
                : result.protected_prefix_len
            ]
            kept_exact_token_ids = request._all_token_ids[kept_exact_start:source_len]
        uncomputed_suffix = request._all_token_ids[source_len:]
        synthetic_prompt = [0] * result.synthetic_prefix_len
        prompt_token_ids = list(protected_prompt_token_ids) + synthetic_prompt

        request.prompt_token_ids = prompt_token_ids
        request.prompt_embeds = None
        request.num_prompt_tokens = len(prompt_token_ids)
        request._output_token_ids = list(kept_exact_token_ids) + list(uncomputed_suffix)
        request.output_token_ids = ConstantList(request._output_token_ids)
        request._all_token_ids = prompt_token_ids + request._output_token_ids
        request.all_token_ids = ConstantList(request._all_token_ids)
        request.num_computed_tokens = (
            max(target_len - 1, 0) if result.pre_sample else target_len
        )
        request.position_offset += total_evicted
        request.needs_rebuild = True
        request.attention_matching_active = True
        if am_prefix_cache_key is not None:
            request.attention_matching_prefix_cache_key = am_prefix_cache_key
            request.attention_matching_prefix_cache_key_start = (
                result.prefix_cache_key_start or result.protected_prefix_len
            )
            request.attention_matching_prefix_cache_hash_start = 0
            request.attention_matching_synthetic_prefix_len = (
                result.synthetic_prefix_len
            )
            request.skip_reading_prefix_cache = False
            request.skip_writing_prefix_cache = False
            request.prefix_cache_skip_reason = ""
            request.prefix_cache_write_skip_logged = False
            request._prompt_embeds_per_block_hashes.clear()
            request.block_hashes = []
            request.update_block_hashes()
            logger.warning(
                "[PrefixCache][AM] assigned AM prefix-cache key %s for "
                "request %s over compacted prefix from token 0 "
                "(protected_prefix=%d)",
                am_prefix_cache_key[:16],
                request.request_id,
                result.protected_prefix_len,
            )
        else:
            request.attention_matching_prefix_cache_key = None
            request.attention_matching_prefix_cache_key_start = 0
            request.attention_matching_prefix_cache_hash_start = 0
            request.attention_matching_synthetic_prefix_len = 0
            if (
                self.cache_config.enable_prefix_caching
                and self.cache_config.prefix_caching_mode == "am_unsafe"
            ):
                request._prompt_embeds_per_block_hashes.clear()
                request.block_hashes = []
                request.update_block_hashes()
                logger.warning(
                    "[PrefixCache][AM][UNSAFE] request %s will cache post-AM "
                    "synthetic blocks under ordinary visible-token hashes",
                    request.request_id,
                )
        if (
            self.cache_config.enable_prefix_caching
            and self.cache_config.prefix_caching_mode
            not in ("am_full", "am_unsafe")
        ):
            request.skip_reading_prefix_cache = True
            request.skip_writing_prefix_cache = True
            request.prefix_cache_skip_reason = "attention_matching_active"
            logger.warning(
                "[PrefixCache][AM] disabled prefix-cache reads/writes for "
                "request %s after AM activation",
                request.request_id,
            )
        request.attention_matching_target_len = target_len
        request.attention_matching_restore_pending = False
        request.attention_matching_snapshot_version = next_version
        logger.warning(
            "[AM] compacted request %s source_len=%d target_len=%d evicted=%d "
            "(scheduler_source_len=%d source_boundary=%d logprob_boundary=%d "
            "protected_prefix=%d "
            "synthetic_prefix=%d pre_sample=%s version=%d offset=%d)",
            request.request_id,
            source_len,
            target_len,
            total_evicted,
            scheduler_source_len,
            source_logical_boundary,
            logprob_boundary,
            result.protected_prefix_len,
            result.synthetic_prefix_len,
            result.pre_sample,
            next_version,
            request.position_offset,
        )
        return total_evicted

    def _apply_shuffle_control_result(
        self,
        request: Request,
        result: ShuffleControlResult,
    ) -> int:
        chunk_len = result.chunk_end - result.chunk_start
        if chunk_len <= 1:
            return 0
        if result.chunk_end > request.num_computed_tokens:
            raise RuntimeError(
                f"shuffle_control chunk exceeds computed tokens for request "
                f"{request.request_id}: chunk_end={result.chunk_end}, "
                f"num_computed={request.num_computed_tokens}"
            )
        if not result.kv_only:
            permutation = chunk_permutation(
                base_seed=self.cache_config.shuffle_control_seed,
                request_id=request.request_id,
                chunk_index=result.chunk_index,
                chunk_len=chunk_len,
            )
            computed_prefix = list(request._all_token_ids[:request.num_computed_tokens])
            chunk_tokens = computed_prefix[result.chunk_start:result.chunk_end]
            computed_prefix[result.chunk_start:result.chunk_end] = [
                chunk_tokens[i] for i in permutation
            ]
            suffix = list(request._all_token_ids[request.num_computed_tokens:])
            request.prompt_token_ids = list(computed_prefix[:request.num_prompt_tokens])
            request.prompt_embeds = None
            request._output_token_ids = (
                list(computed_prefix[request.num_prompt_tokens:]) + suffix
            )
            request.output_token_ids = ConstantList(request._output_token_ids)
            request._all_token_ids = list(computed_prefix) + suffix
            request.all_token_ids = ConstantList(request._all_token_ids)
        request.shuffle_events.append(
            ShuffleEvent(
                num_output_tokens_at_shuffle=request.num_total_generated,
                chunk_index=result.chunk_index,
                chunk_start=result.chunk_start,
                chunk_end=result.chunk_end,
            )
        )
        request.shuffle_control_next_chunk_index = max(
            request.shuffle_control_next_chunk_index,
            result.chunk_index + 1,
        )
        logger.warning(
            "[SHUFFLE] applied request %s chunk=%d range=[%d,%d)",
            request.request_id,
            result.chunk_index,
            result.chunk_start,
            result.chunk_end,
        )
        return 1

    def _apply_noise_control_result(
        self,
        request: Request,
        result: NoiseControlResult,
    ) -> int:
        chunk_len = result.chunk_end - result.chunk_start
        if chunk_len <= 1:
            return 0
        if result.chunk_end > request.num_computed_tokens:
            raise RuntimeError(
                f"noise_control chunk exceeds computed tokens for request "
                f"{request.request_id}: chunk_end={result.chunk_end}, "
                f"num_computed={request.num_computed_tokens}"
            )
        request.noise_events.append(
            NoiseEvent(
                num_output_tokens_at_noise=request.num_total_generated,
                chunk_index=result.chunk_index,
                chunk_start=result.chunk_start,
                chunk_end=result.chunk_end,
                target=result.target,
                std=result.std,
            )
        )
        request.noise_control_next_chunk_index = max(
            request.noise_control_next_chunk_index,
            result.chunk_index + 1,
        )
        logger.warning(
            "[NOISE] applied request %s chunk=%d range=[%d,%d) target=%s std=%.6g",
            request.request_id,
            result.chunk_index,
            result.chunk_start,
            result.chunk_end,
            result.target,
            result.std,
        )
        return 1

    def _validate_attention_matching_request(self, request: Request) -> None:
        if not self._attention_matching_enabled:
            return
        assert request.prompt_embeds is None, (
            "attention_matching does not support prompt_embeds requests"
        )
        assert not request.mm_features, (
            "attention_matching baseline does not support multimodal requests"
        )
        sampling_params = request.sampling_params
        assert sampling_params is not None, (
            "attention_matching only supports generative requests"
        )
        assert sampling_params.prompt_logprobs is None, (
            "attention_matching does not support prompt_logprobs"
        )
        assert sampling_params.presence_penalty == 0.0, (
            "attention_matching does not support presence_penalty"
        )
        assert sampling_params.frequency_penalty == 0.0, (
            "attention_matching does not support frequency_penalty"
        )
        assert sampling_params.repetition_penalty == 1.0, (
            "attention_matching does not support repetition_penalty"
        )
        assert not sampling_params.bad_words, (
            "attention_matching does not support bad_words"
        )
        assert not request.resumable, (
            "attention_matching baseline does not support resumable streaming"
        )

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
        session.logical_prompt_len = session.num_prompt_tokens
        self._refresh_attention_matching_protected_prompt_len(session)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _refresh_attention_matching_protected_prompt_len(
        self, request: Request
    ) -> None:
        """Compute the immutable prompt prefix AM must not overwrite.

        In ``am_full`` prefix-caching mode this also bounds pre-AM prefix-cache
        hits to the same immutable region. Once AM has rewritten the request,
        post-AM blocks get AM-specific prefix-cache hashes.
        """
        if not self._attention_matching_enabled:
            return

        protect_mode = self.cache_config.attention_matching_protect_user_prompts
        if protect_mode == "none":
            protected_len = 0
        elif protect_mode == "all_user" or self.cache_config.compaction_max_turns <= 0:
            protected_len = request.num_prompt_tokens
        else:
            turn_end_token_id = self.cache_config.compaction_turn_end_token_id
            turn_padding_token_id = self.cache_config.compaction_turn_padding_token_id
            token_ids = request.prompt_token_ids or request._all_token_ids[
                : request.num_prompt_tokens
            ]
            protected_len = request.num_prompt_tokens
            if turn_end_token_id is not None:
                turn_ends: list[int] = []
                pos = 0
                source_len = min(request.num_prompt_tokens, len(token_ids))
                while pos < source_len:
                    if token_ids[pos] != turn_end_token_id:
                        pos += 1
                        continue
                    boundary = advance_attention_matching_turn_boundary(
                        token_ids,
                        pos + 1,
                        source_len,
                        turn_padding_token_id,
                    )
                    turn_ends.append(boundary)
                    pos = boundary
                if len(turn_ends) >= 2:
                    # Rendered chat shape is system, user, assistant, ...
                    # Keep system + first user prompt immutable by default.
                    protected_len = turn_ends[1]

        request.attention_matching_protected_prompt_len = min(
            max(protected_len, 0), request.num_prompt_tokens
        )

    def _get_attention_matching_turn_end_token_id(self) -> int:
        token_id = self.cache_config.compaction_turn_end_token_id
        if token_id is not None:
            return token_id
        eos_token_id = getattr(
            self.vllm_config.model_config.hf_config, "eos_token_id", None
        )
        if isinstance(eos_token_id, int):
            return eos_token_id
        if isinstance(eos_token_id, list) and len(eos_token_id) == 1:
            return int(eos_token_id[0])
        raise RuntimeError(
            "AM turn-window compaction requires compaction_turn_end_token_id "
            "or a single integer model eos_token_id"
        )

    def _build_attention_matching_cross_turn_replay(
        self, request: Request
    ) -> AttentionMatchingPrefixCacheReplay | None:
        if (
            not self.cache_config.attention_matching_cross_turn_cache
            or not self._attention_matching_enabled
            or self.cache_config.prefix_caching_mode != "am_full"
            or self.cache_config.compaction_max_turns <= 0
            or request.prompt_token_ids is None
            or request.prompt_embeds is not None
            or request.mm_features
            or request.num_output_tokens > 0
            or request.num_computed_tokens > 0
        ):
            return None
        replay = build_attention_matching_turn_prefix_cache_replay(
            token_ids=request.prompt_token_ids,
            base_seed=int(getattr(self.vllm_config.model_config, "seed", 0) or 0),
            cache_salt=request.cache_salt,
            synthetic_prefix_len=self.cache_config.compaction_stride,
            max_turns=self.cache_config.compaction_max_turns,
            keep_recent_turns=self.cache_config.compaction_eviction_turn_stride,
            turn_end_token_id=self._get_attention_matching_turn_end_token_id(),
            turn_padding_token_id=self.cache_config.compaction_turn_padding_token_id,
            protect_first_user=(
                self.cache_config.attention_matching_protect_user_prompts
                == "first_user"
            ),
            query_source=self.cache_config.attention_matching_query_source,
            max_queries_per_kv_head=(
                self.cache_config.attention_matching_max_queries_per_kv_head
            ),
            zerobeta=self.cache_config.attention_matching_zerobeta,
            forget_gate_enabled=(
                self.cache_config.attention_matching_forget_gate_enabled
            ),
            forget_gate_alpha=self.cache_config.attention_matching_forget_gate_alpha,
            min_protected_prefix_len=0,
        )
        if replay is None or replay.final_step is None:
            return None
        return replay

    def _activate_attention_matching_cross_turn_candidate(
        self,
        request: Request,
        replay: AttentionMatchingPrefixCacheReplay,
        step_index: int,
        original_prompt: list[int],
    ) -> None:
        step = replay.steps[step_index]
        plan = step.plan

        physical_prompt = list(step.physical_token_ids_after)
        exact_kept_tokens = (
            len(physical_prompt)
            - plan.protected_prefix_len
            - plan.synthetic_prefix_len
        )

        event = CompactionEvent(
            num_output_tokens_at_compaction=0,
            tokens_evicted=step.position_offset_after,
            position_offset_after=step.position_offset_after,
            num_prompt_tokens=len(original_prompt),
            compaction_strategy="attention_matching",
            source_len=len(original_prompt),
            target_len=len(physical_prompt),
            protected_prefix_len=plan.protected_prefix_len,
            synthetic_prefix_len=plan.synthetic_prefix_len,
            exact_kept_tokens=exact_kept_tokens,
            attention_matching_query_source=(
                self.cache_config.attention_matching_query_source
            ),
            attention_matching_max_queries_per_kv_head=(
                self.cache_config.attention_matching_max_queries_per_kv_head
            ),
            attention_matching_query_seed=step.query_seed,
            attention_matching_zerobeta=self.cache_config.attention_matching_zerobeta,
            attention_matching_pre_sample=True,
            attention_matching_replay_steps=[
                self._attention_matching_step_payload(
                    source_len=replay_step.plan.source_len,
                    target_len=replay_step.plan.target_len,
                    protected_prefix_len=replay_step.plan.protected_prefix_len,
                    synthetic_prefix_len=replay_step.plan.synthetic_prefix_len,
                    exact_kept_tokens=replay_step.plan.exact_kept_tokens,
                    query_seed=replay_step.query_seed,
                    prefix_cache_key=replay_step.prefix_cache_key,
                    selected_indices=(
                        self.attention_matching_prefix_cache_selected_indices.get(
                            replay_step.prefix_cache_key
                        )
                    ),
                    forget_gate_enabled=(
                        self.cache_config.attention_matching_forget_gate_enabled
                    ),
                    forget_gate_alpha=(
                        self.cache_config.attention_matching_forget_gate_alpha
                    ),
                    forget_gate_applied=(
                        self.cache_config.attention_matching_forget_gate_enabled
                        and replay_step_idx > 0
                    ),
                )
                for replay_step_idx, replay_step in enumerate(
                    replay.steps[: step_index + 1]
                )
            ],
            attention_matching_selected_indices=(
                self.attention_matching_prefix_cache_selected_indices.get(
                    step.prefix_cache_key
                )
            ),
            attention_matching_forget_gate_enabled=(
                self.cache_config.attention_matching_forget_gate_enabled
            ),
            attention_matching_forget_gate_alpha=(
                self.cache_config.attention_matching_forget_gate_alpha
            ),
            attention_matching_forget_gate_applied=(
                self.cache_config.attention_matching_forget_gate_enabled
                and step_index > 0
            ),
        )

        request.attention_matching_original_prompt_token_ids = original_prompt
        request.attention_matching_cross_turn_candidate = True
        request.attention_matching_cross_turn_event = event
        request.attention_matching_cross_turn_replay = replay
        request.attention_matching_cross_turn_replay_index = step_index
        request.prompt_token_ids = physical_prompt
        request.prompt_embeds = None
        request.num_prompt_tokens = len(physical_prompt)
        request.logical_prompt_len = len(original_prompt)
        request._output_token_ids = []
        request.output_token_ids = ConstantList(request._output_token_ids)
        request._all_token_ids = list(physical_prompt)
        request.all_token_ids = ConstantList(request._all_token_ids)
        request.position_offset = step.position_offset_after
        request.attention_matching_active = True
        request.attention_matching_snapshot_version = step_index + 1
        request.attention_matching_target_len = len(physical_prompt)
        request.attention_matching_prefix_cache_key = step.prefix_cache_key
        request.attention_matching_prefix_cache_key_start = plan.protected_prefix_len
        request.attention_matching_prefix_cache_hash_start = 0
        request.attention_matching_synthetic_prefix_len = plan.synthetic_prefix_len
        request.skip_reading_prefix_cache = False
        request.skip_writing_prefix_cache = False
        request.prefix_cache_skip_reason = ""
        request.prefix_cache_write_skip_logged = False
        request.prefix_cache_read_hit_logged = False
        request.num_cached_tokens = -1
        request._prompt_embeds_per_block_hashes.clear()
        request.block_hashes = []
        request.update_block_hashes()
        logger.warning(
            "[PrefixCache][AM][cross-turn] prepared compressed admission "
            "candidate for request %s source_len=%d target_len=%d "
            "protected=%d synthetic=%d exact_tail=%d offset=%d "
            "step=%d/%d key=%s",
            request.request_id,
            len(original_prompt),
            len(physical_prompt),
            plan.protected_prefix_len,
            plan.synthetic_prefix_len,
            exact_kept_tokens,
            step.position_offset_after,
            step_index + 1,
            len(replay.steps),
            step.prefix_cache_key[:16],
        )

    def _maybe_prepare_attention_matching_cross_turn_candidate(
        self, request: Request
    ) -> None:
        replay = self._build_attention_matching_cross_turn_replay(request)
        if replay is None:
            return
        assert request.prompt_token_ids is not None
        self._activate_attention_matching_cross_turn_candidate(
            request,
            replay,
            len(replay.steps) - 1,
            list(request.prompt_token_ids),
        )

    def _restore_attention_matching_cross_turn_candidate(
        self, request: Request
    ) -> None:
        original_prompt = request.attention_matching_original_prompt_token_ids
        if original_prompt is None:
            return
        request.prompt_token_ids = list(original_prompt)
        request.prompt_embeds = None
        request.num_prompt_tokens = len(original_prompt)
        request.logical_prompt_len = request.num_prompt_tokens
        request._output_token_ids = []
        request.output_token_ids = ConstantList(request._output_token_ids)
        request._all_token_ids = list(original_prompt)
        request.all_token_ids = ConstantList(request._all_token_ids)
        request.position_offset = 0
        request.attention_matching_active = False
        request.attention_matching_snapshot_version = None
        request.attention_matching_target_len = None
        request.attention_matching_prefix_cache_key = None
        request.attention_matching_prefix_cache_key_start = 0
        request.attention_matching_prefix_cache_hash_start = 0
        request.attention_matching_synthetic_prefix_len = 0
        request.attention_matching_original_prompt_token_ids = None
        request.attention_matching_cross_turn_candidate = False
        request.attention_matching_cross_turn_event = None
        request.attention_matching_cross_turn_replay = None
        request.attention_matching_cross_turn_replay_index = -1
        request.num_cached_tokens = -1
        request.prefix_cache_read_hit_logged = False
        if (
            self.cache_config.enable_prefix_caching
            and self.cache_config.prefix_caching_mode == "am_full"
        ):
            request.skip_reading_prefix_cache = True
            request.skip_writing_prefix_cache = True
            request.prefix_cache_skip_reason = (
                "attention_matching_compressed_admission_miss"
            )
            request.prefix_cache_write_skip_logged = False
        request._prompt_embeds_per_block_hashes.clear()
        request.block_hashes = []
        request.update_block_hashes()
        self._refresh_attention_matching_protected_prompt_len(request)

    def _maybe_finalize_attention_matching_cross_turn_candidate(
        self,
        request: Request,
        num_local_computed_tokens: int,
    ) -> bool:
        """Return True when the caller must redo prefix lookup after fallback."""
        if not request.attention_matching_cross_turn_candidate:
            return False

        event = request.attention_matching_cross_turn_event
        min_replayable_hit_tokens = 0
        if event is not None:
            # A compressed hit is replayable once it covers the immutable
            # protected prefix plus the synthetic AM prefix. Any missing exact
            # tail remains ordinary physical prompt tokens and is prefilling
            # under the compacted cache. This partial-hit mode is intentionally
            # opt-in because trainer replay needs richer metadata to represent
            # a shallow cached state followed by private deeper AM replay.
            min_replayable_hit_tokens = (
                event.protected_prefix_len + event.synthetic_prefix_len
            )

        max_safe_hit = max(request.num_tokens - 1, 0)
        full_reusable_hit_tokens = (
            max_safe_hit // self.block_size
        ) * self.block_size
        allow_partial_cross_turn_hits = bool(
            getattr(
                getattr(self, "cache_config", None),
                "attention_matching_allow_partial_cross_turn_cache_hits",
                False,
            )
        )
        required_hit_tokens = (
            min_replayable_hit_tokens
            if allow_partial_cross_turn_hits
            else full_reusable_hit_tokens
        )

        def retry_shallower_or_restore(reason: str) -> bool:
            replay = request.attention_matching_cross_turn_replay
            replay_index = request.attention_matching_cross_turn_replay_index
            original_prompt = request.attention_matching_original_prompt_token_ids
            key = (request.attention_matching_prefix_cache_key or "")[:16]

            if (
                replay is not None
                and original_prompt is not None
                and replay_index > 0
            ):
                logger.warning(
                    "[PrefixCache][AM][cross-turn] compressed admission miss "
                    "for request %s hit=%d required=%d min_required=%d "
                    "full_reusable=%d allow_partial=%s step=%d/%d key=%s; "
                    "retrying shallower",
                    request.request_id,
                    num_local_computed_tokens,
                    required_hit_tokens,
                    min_replayable_hit_tokens,
                    full_reusable_hit_tokens,
                    allow_partial_cross_turn_hits,
                    replay_index + 1,
                    len(replay.steps),
                    key,
                )
                logger.warning(
                    "[PrefixCache][AM][cross-turn] retrying request %s with "
                    "shallower compressed AM state step=%d/%d after %s",
                    request.request_id,
                    replay_index,
                    len(replay.steps),
                    reason,
                )
                self._activate_attention_matching_cross_turn_candidate(
                    request,
                    replay,
                    replay_index - 1,
                    list(original_prompt),
                )
                return True

            logger.warning(
                "[PrefixCache][AM][cross-turn] compressed admission miss for "
                "request %s hit=%d required=%d min_required=%d "
                "full_reusable=%d allow_partial=%s step=%d/%d key=%s; "
                "restoring full prompt for private AM replay",
                request.request_id,
                num_local_computed_tokens,
                required_hit_tokens,
                min_replayable_hit_tokens,
                full_reusable_hit_tokens,
                allow_partial_cross_turn_hits,
                max(replay_index + 1, 0),
                len(replay.steps) if replay is not None else 0,
                key,
            )
            if reason != "cache_miss":
                logger.warning(
                    "[PrefixCache][AM][cross-turn] restoring request %s after "
                    "compressed AM admission failed reason=%s",
                    request.request_id,
                    reason,
                )
            self._restore_attention_matching_cross_turn_candidate(request)
            return True

        if (
            required_hit_tokens <= 0
            or num_local_computed_tokens < required_hit_tokens
        ):
            return retry_shallower_or_restore("cache_miss")

        if event is not None and not self._attention_matching_event_has_selected_indices(
            event
        ):
            return retry_shallower_or_restore("missing_selected_indices")
        if event is not None:
            event.attention_matching_cache_hit_tokens = num_local_computed_tokens
            request.compaction_events.append(event)
        request.attention_matching_cross_turn_candidate = False
        request.attention_matching_cross_turn_event = None
        request.attention_matching_original_prompt_token_ids = None
        replay = request.attention_matching_cross_turn_replay
        replay_index = request.attention_matching_cross_turn_replay_index
        request.attention_matching_cross_turn_replay = None
        request.attention_matching_cross_turn_replay_index = -1
        logger.warning(
            "[PrefixCache][AM][cross-turn] compressed admission hit for "
            "request %s hit=%d required=%d min_required=%d full_reusable=%d "
            "allow_partial=%s step=%d/%d key=%s",
            request.request_id,
            num_local_computed_tokens,
            required_hit_tokens,
            min_replayable_hit_tokens,
            full_reusable_hit_tokens,
            allow_partial_cross_turn_hits,
            max(replay_index + 1, 0),
            len(replay.steps) if replay is not None else 0,
            (request.attention_matching_prefix_cache_key or "")[:16],
        )
        return False

    def _maybe_start_am_prefix_cache_tail_finalization(
        self, request: Request
    ) -> bool:
        """Compute hidden filler KV so stopped AM turns end on full blocks."""
        if (
            not self.cache_config.enable_prefix_caching
            or self.cache_config.prefix_caching_mode != "am_full"
            or not self.cache_config.attention_matching_cross_turn_cache
            or not self._attention_matching_enabled
            or request.prefix_cache_tail_finalizing
            or not request.attention_matching_active
            or request.attention_matching_prefix_cache_key is None
        ):
            return False

        padding_token_id = self.cache_config.compaction_turn_padding_token_id
        if padding_token_id is None:
            raise RuntimeError(
                "AM cross-turn prefix caching requires "
                "compaction_turn_padding_token_id so hidden tail "
                "finalization uses the same block-alignment filler as the "
                "orchestrator-rendered next turn."
            )
        if (
            request.prompt_token_ids is None
            or request.prompt_embeds is not None
            or request.mm_features
        ):
            raise RuntimeError(
                "AM prefix-cache tail finalization only supports token-id "
                "text requests; prompt_embeds and multimodal features cannot "
                "be finalized faithfully."
            )

        turn_end_token_id = self._get_attention_matching_turn_end_token_id()
        hidden_token_ids: list[int] = []
        needs_hidden_turn_end = (
            not request.output_token_ids
            or request.output_token_ids[-1] != turn_end_token_id
        )
        if needs_hidden_turn_end:
            hidden_token_ids.append(turn_end_token_id)
        hidden_token_ids.extend(DEFAULT_TURN_SEPARATOR_TOKEN_ID_SEQUENCE)

        remainder = (request.num_tokens + len(hidden_token_ids)) % self.block_size
        if remainder != 0:
            hidden_token_ids.extend(
                [padding_token_id] * (self.block_size - remainder)
            )
        if not hidden_token_ids:
            return False

        # The worker discards the dummy sample from hidden filler prefill, so
        # the hidden tokens only need to fit in the actual model context.
        if request.num_tokens + len(hidden_token_ids) > self.max_model_len:
            logger.warning(
                "[PrefixCache][AM] skipping tail block finalization for "
                "request %s: tokens=%d hidden=%d max_model_len=%d",
                request.request_id,
                request.num_tokens,
                len(hidden_token_ids),
                self.max_model_len,
            )
            return False

        if not request.compaction_events:
            raise RuntimeError(
                "AM prefix-cache tail finalization requires a compaction "
                f"event to carry hidden tail tokens for request {request.request_id}."
            )
        last_event = request.compaction_events[-1]
        if getattr(last_event, "compaction_strategy", "") != "attention_matching":
            raise RuntimeError(
                "AM prefix-cache tail finalization found a non-AM final "
                f"compaction event for request {request.request_id}."
            )
        last_event.attention_matching_hidden_tail_token_ids = list(hidden_token_ids)

        request.prefix_cache_tail_final_status = request.status
        request.prefix_cache_tail_hidden_tokens = 0
        request.append_hidden_output_token_ids(hidden_token_ids)
        request.prefix_cache_tail_finalizing = True
        request.status = RequestStatus.RUNNING
        request.needs_rebuild = True
        self.prev_step_scheduled_req_ids.discard(request.request_id)
        request._prompt_embeds_per_block_hashes.clear()
        request.block_hashes = []
        request.update_block_hashes()
        logger.warning(
            "[PrefixCache][AM] finalizing stopped request %s with %d hidden "
            "tokens to close a cache block (hidden_turn_end=%s tokens=%d key=%s)",
            request.request_id,
            len(hidden_token_ids),
            needs_hidden_turn_end,
            request.num_tokens,
            (request.attention_matching_prefix_cache_key or "")[:16],
        )
        return True

    def _maybe_privatize_attention_matching_blocks(
        self, request: Request, num_tokens: int
    ) -> None:
        """Fork shared AM prefix-cache blocks before the request can mutate KV."""
        if (
            not self.cache_config.enable_prefix_caching
            or self.cache_config.prefix_caching_mode != "am_full"
            or not self.cache_config.attention_matching_cross_turn_cache
            or not self._attention_matching_enabled
            or request.attention_matching_prefix_cache_key is None
            or num_tokens <= 0
        ):
            return
        src, dst = self.kv_cache_manager.privatize_attention_matching_blocks(
            request.request_id, num_tokens
        )
        if not src:
            return
        request.attention_matching_cow_src_block_ids.extend(src)
        request.attention_matching_cow_dst_block_ids.extend(dst)
        logger.warning(
            "[PrefixCache][AM][COW] privatized %d shared cached blocks for "
            "request %s before mutable AM continuation key=%s",
            len(src),
            request.request_id,
            (request.attention_matching_prefix_cache_key or "")[:16],
        )

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
        attention_matching_restore_req_ids: set[str] = set()
        attention_matching_snapshot_versions: dict[str, int] = {}
        attention_matching_prefix_cache_keys: dict[str, str] = {}
        attention_matching_prefix_cache_key_starts: dict[str, int] = {}
        attention_matching_suppress_compaction_req_ids: set[str] = set()
        attention_matching_cow_src_block_ids: dict[str, list[int]] = {}
        attention_matching_cow_dst_block_ids: dict[str, list[int]] = {}

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            if req.attention_matching_prefix_cache_key is not None:
                attention_matching_prefix_cache_keys[req_id] = (
                    req.attention_matching_prefix_cache_key
                )
                attention_matching_prefix_cache_key_starts[req_id] = (
                    req.attention_matching_prefix_cache_key_start
                )
            if req.prefix_cache_tail_finalizing:
                attention_matching_suppress_compaction_req_ids.add(req_id)
            if req.attention_matching_cow_src_block_ids:
                attention_matching_cow_src_block_ids[req_id] = list(
                    req.attention_matching_cow_src_block_ids
                )
                attention_matching_cow_dst_block_ids[req_id] = list(
                    req.attention_matching_cow_dst_block_ids
                )
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
                all_token_ids[req_id] = req.all_token_ids.copy()
                if self._attention_matching_enabled:
                    prompt_lengths[req_id] = req.num_prompt_tokens
                # Send full block_ids (not just new) for rebuild.
                full_block_ids = tuple(
                    [blk.block_id for blk in group]
                    for group in self.kv_cache_manager.coordinator.get_blocks(
                        req_id
                    )
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
                    if req.attention_matching_restore_pending:
                        attention_matching_restore_req_ids.add(req_id)
                        position_offsets[req_id] = req.position_offset
                        prompt_lengths[req_id] = req.num_prompt_tokens
                        snapshot_version = (
                            req.attention_matching_snapshot_version
                        )
                        if snapshot_version is not None:
                            attention_matching_snapshot_versions[req_id] = (
                                snapshot_version
                            )
                        req.attention_matching_restore_pending = False
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
            attention_matching_restore_req_ids=attention_matching_restore_req_ids,
            attention_matching_snapshot_versions=attention_matching_snapshot_versions,
            attention_matching_prefix_cache_keys=attention_matching_prefix_cache_keys,
            attention_matching_prefix_cache_key_starts=(
                attention_matching_prefix_cache_key_starts
            ),
            attention_matching_suppress_compaction_req_ids=(
                attention_matching_suppress_compaction_req_ids
            ),
            attention_matching_cow_src_block_ids=attention_matching_cow_src_block_ids,
            attention_matching_cow_dst_block_ids=attention_matching_cow_dst_block_ids,
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
        cudagraph_stats = model_runner_output.cudagraph_stats
        attention_matching_compactions = (
            model_runner_output.attention_matching_compactions
        )
        shuffle_control_results = model_runner_output.shuffle_control_results
        noise_control_results = model_runner_output.noise_control_results

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
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
            hidden_tail_finalized = False

            # Check for stop and update request status.
            if request.prefix_cache_tail_finalizing:
                # The previous visible step stopped the request but left a
                # partial generated block. Hidden filler may be chunked by
                # token budget, so finish only after all hidden filler KV has
                # actually been computed. Sampled tokens from these filler
                # steps are always discarded.
                new_token_ids = []
                if request.num_computed_tokens < request.num_tokens:
                    logger.warning(
                        "[PrefixCache][AM] continuing hidden tail finalization "
                        "for request %s computed=%d tokens=%d hidden_tokens=%d",
                        request.request_id,
                        request.num_computed_tokens,
                        request.num_tokens,
                        request.prefix_cache_tail_hidden_tokens,
                    )
                else:
                    stopped = True
                    hidden_tail_finalized = True
                    final_status = request.prefix_cache_tail_final_status
                    request.prefix_cache_tail_finalizing = False
                    request.prefix_cache_tail_final_status = None
                    request.status = (
                        final_status
                        if final_status is not None
                        else RequestStatus.FINISHED_STOPPED
                    )
                    logger.warning(
                        "[PrefixCache][AM] completed hidden tail finalization for "
                        "request %s hidden_tokens=%d",
                        request.request_id,
                        request.prefix_cache_tail_hidden_tokens,
                    )
            elif new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            routed_experts = None
            finish_reason = None
            if stopped:
                if (
                    not hidden_tail_finalized
                    and self._maybe_start_am_prefix_cache_tail_finalization(request)
                ):
                    stopped = False
                else:
                    if not hidden_tail_finalized:
                        routed_experts = self._get_routed_experts(request)

                    # Capture finish_reason BEFORE _handle_stopped_request, which may
                    # reset the status to WAITING for streaming requests that continue.
                    finish_reason = request.get_finished_reason()
                    finished = self._handle_stopped_request(request)
                    if finished:
                        kv_transfer_params = self._free_request(request)

                    if status_before_stop == RequestStatus.RUNNING:
                        stopped_running_reqs.add(request)
                    else:
                        stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if (
                not stopped
                and attention_matching_compactions
                and req_id in attention_matching_compactions
            ):
                tokens_evicted = self._apply_attention_matching_compaction(
                    request, attention_matching_compactions[req_id]
                )
                if tokens_evicted > 0:
                    self.prev_step_scheduled_req_ids.discard(request.request_id)
            if (
                not stopped
                and shuffle_control_results
                and req_id in shuffle_control_results
            ):
                applied = 0
                for result in shuffle_control_results[req_id]:
                    applied += self._apply_shuffle_control_result(request, result)
                if applied > 0:
                    self.prev_step_scheduled_req_ids.discard(request.request_id)
            if (
                not stopped
                and noise_control_results
                and req_id in noise_control_results
            ):
                applied = 0
                for result in noise_control_results[req_id]:
                    applied += self._apply_noise_control_result(request, result)
                if applied > 0:
                    self.prev_step_scheduled_req_ids.discard(request.request_id)

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

            # --- KV cache compaction ---
            # After tokens are appended and stop is checked, compact if needed.
            # Must be AFTER stop check (don't compact finished requests).
            if (
                not stopped
                and self._compaction_enabled
                and not request.prefix_cache_tail_finalizing
                and request.num_output_placeholders == 0
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
                shuffle_events = (
                    list(request.shuffle_events)
                    if request.shuffle_events
                    else None
                )
                noise_events = (
                    list(request.noise_events)
                    if request.noise_events
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
                        shuffle_events=shuffle_events,
                        noise_events=noise_events,
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
            self._validate_attention_matching_request(request)
            if request.resumable:
                request.streaming_queue = deque()
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
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
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
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # NOTE(zhuohan): For async scheduling, we need to discard the latest
                # output token on the fly to avoid a redundant repetitive output token.
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

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

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
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
