# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


@dataclass(frozen=True)
class CompactReplaySnapshot:
    """Replay metadata for reconstructing a compacted request.

    token_ids is the full writer timeline. death_indices follows the Flex
    replay contract: key row k is visible to query q iff
    k <= q < death_indices[k].
    """

    token_ids: tuple[int, ...]
    death_indices: tuple[int, ...]
    live_writer_indices: tuple[int, ...]
    evictions: int

    @property
    def final_writer_len(self) -> int:
        return len(self.token_ids)

    def is_valid(self) -> bool:
        if len(self.death_indices) != len(self.token_ids):
            return False
        if any(death_idx <= idx for idx, death_idx in enumerate(self.death_indices)):
            return False
        live_writer_indices = set(self.live_writer_indices)
        if len(live_writer_indices) != len(self.live_writer_indices):
            return False
        if not all(
            0 <= writer_idx < len(self.token_ids)
            for writer_idx in self.live_writer_indices
        ):
            return False
        final_len = len(self.token_ids)
        if any(self.death_indices[idx] != final_len for idx in live_writer_indices):
            return False
        return True


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        reasoning_ended: bool | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        if self.structured_output_request is not None:
            self.structured_output_request.reasoning_ended = reasoning_ended
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # KV cache compaction (block_aligned_finish): once generation has
        # stopped, the scheduler may transition the request into an
        # auto-padding step that runs one more forward over filler tokens
        # so the trailing partial block enters the prefix cache. While
        # `padding_pending` is True, the request is held in `self.running`
        # and the worker should skip sampling for it.
        self.num_padding_tokens: int = 0
        self.padding_pending: bool = False
        self._pending_finish_status: "RequestStatus | None" = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        # Cache per-block prompt-embed hashes to avoid rehashing the same
        # tensor slices when generating extra keys.
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )
        self._kve_compact_replay_token_ids: list[int] | None = None
        self._kve_compact_replay_death_indices: list[int | None] | None = None
        self._kve_compact_replay_live_writer_indices: list[int] | None = None
        self._kve_compact_replay_evictions: int = 0
        self._kve_compact_replay_last_error: str | None = None

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Used in forced preemption (reset_prefix_cache) with async scheduling.
        self.discard_latest_async_tokens = False

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # --- Compaction state ---
        # Cumulative evicted tokens. Used ONLY for RoPE position correction.
        self.position_offset: int = 0
        # Monotonic counter: total output tokens EVER generated (never
        # decremented). Used by check_stop for max_tokens because
        # len(_output_token_ids) shrinks after compaction.
        self.num_total_generated: int = 0
        # Streaming sessions: num_total_generated at the start of the
        # current segment. check_stop budgets each segment independently
        # via num_segment_generated; stays 0 for ordinary requests.
        self.segment_generated_base: int = 0
        # History of compaction events (included in API response metadata).
        self.compaction_events: list = []
        # Cursor: number of compaction_events already streamed to the client.
        # The scheduler sends only events[compaction_events_sent:] each step
        # (delta) instead of the full cumulative list, which previously leaked
        # host RAM in the front-end because each event carries context-length
        # kept_indices/kept_token_ids arrays. The output processor appends the
        # deltas, so the client still ends up with the full cumulative list.
        self.compaction_events_sent: int = 0
        # Flag: request was compacted and needs model runner rebuild.
        self.needs_rebuild: bool = False
        # Managed context: model-selected retries must prefill their visible
        # retry suffix before restored hidden KV is attached for answer decode.
        self.managed_context_defer_restore_until_prefill: bool = False
        # Managed context: per-request recall movement verdict, set by
        # _activate_managed_context_restore and echoed back to the client on the
        # response so the client's [MANAGED-CONTEXT-CLIENT-RESTORE] log can show
        # whether the recall was a CPU->GPU H2D move or GPU-resident (no move).
        # Pure metadata: never touches KV/logits/tokens. dict or None.
        self.managed_context_restore_kind: dict | None = None

        # Turn tracking (only populated when compaction_max_turns > 0).
        # Absolute positions (in the CURRENT post-eviction _all_token_ids)
        # of the first token AFTER each <|im_end|> seen so far. Monotonic.
        # turn_end_positions[0] is the end of the system prompt;
        # turn_end_positions[2*k] (for k >= 1) is the end of turn k.
        self.turn_end_positions: list[int] = []
        # Cursor: tokens in _all_token_ids[:last_turn_scan_pos] have already
        # been scanned for <|im_end|>. Lazy, extended on demand by the
        # scheduler. Reset/adjusted on eviction so positions stay valid.
        self.last_turn_scan_pos: int = 0
        # Count of whole turns (user+assistant pairs) physically evicted by
        # prior compactions on this request. Monotonic.
        self.num_turns_evicted: int = 0
        # Absolute ids for completed turns currently represented by
        # turn_end_positions. Prefix eviction can derive these from
        # num_turns_evicted, but sparse kv-selection needs the explicit list.
        self.live_turn_ids: list[int] = []
        self.next_turn_id: int = 0
        # Streaming-session marker: num_computed_tokens at the moment of the
        # most recent _update_request_as_session call, i.e. the partition
        # between "pre-existing cached prompt" (positions [0, boundary)) and
        # "newly-appended content this call" (positions [boundary, num_prompt)).
        # Mid-call admission eviction fires at this boundary BEFORE the new
        # content is prefilled, so the new content's K vectors are computed
        # under the post-eviction state. See plans/connect_admission_events_
        # to_trainer.md "Operative intent" section. Debug/event metadata only;
        # the kernel uses num_computed_tokens + block_table directly.
        self.session_prefill_boundary: int = 0

        # Multi-modal related
        self.mm_features = mm_features or []

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        self.update_block_hashes()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # Used for streaming
        self.resumable = resumable
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            reasoning_ended=request.reasoning_ended,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
            self.num_total_generated += 1
            self._append_compact_replay_token_ids([token_ids])
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)
            self.num_total_generated += len(token_ids)
            self._append_compact_replay_token_ids(token_ids)

        self.update_block_hashes()

    def append_padding_token_ids(
        self,
        padding_token_id: int,
        count: int,
    ) -> None:
        """Append filler tokens that extend the KV cache to a block
        boundary. Unlike append_output_token_ids these don't enter
        `_output_token_ids` (they aren't user-facing outputs); they
        only grow `_all_token_ids` (which `num_tokens` keys off) so
        the next scheduling iteration's allocate_slots/cache_blocks
        will write K/V for them and cache the resulting full block.

        Used by `compaction_block_aligned_finish` (see CacheConfig).
        """
        if count <= 0:
            return
        for _ in range(count):
            self._all_token_ids.append(padding_token_id)
        self._append_compact_replay_token_ids([padding_token_id] * count)
        self.num_padding_tokens += count
        self.update_block_hashes()

    def _append_compact_replay_token_ids(self, token_ids: list[int]) -> None:
        replay_token_ids = self._kve_compact_replay_token_ids
        death_indices = self._kve_compact_replay_death_indices
        live_writer_indices = self._kve_compact_replay_live_writer_indices
        if (
            replay_token_ids is None
            or death_indices is None
            or live_writer_indices is None
            or not token_ids
        ):
            return
        writer_start = len(replay_token_ids)
        replay_token_ids.extend(token_ids)
        death_indices.extend([None] * len(token_ids))
        live_writer_indices.extend(
            range(writer_start, writer_start + len(token_ids))
        )

    def _ensure_compact_replay_timeline(self) -> bool:
        if self._kve_compact_replay_token_ids is not None:
            return True
        if self.prompt_token_ids is None:
            self._kve_compact_replay_last_error = "prompt token ids unavailable"
            return False
        self._kve_compact_replay_token_ids = self._all_token_ids.copy()
        self._kve_compact_replay_death_indices = [None] * len(
            self._all_token_ids
        )
        self._kve_compact_replay_live_writer_indices = list(
            range(len(self._all_token_ids))
        )
        self._kve_compact_replay_last_error = None
        return True

    def mark_compact_replay_eviction(
        self,
        evict_start: int,
        tokens_evicted: int,
    ) -> bool:
        """Record a KV compaction splice in replay coordinates.

        This deliberately does not delete from the replay token timeline.
        Deleted rows are marked dead at the current writer length and removed
        from the live mapping so future evictions address the post-compaction
        cache coordinates.
        """
        if tokens_evicted <= 0:
            return True
        live_len = (
            len(self._kve_compact_replay_live_writer_indices)
            if self._kve_compact_replay_live_writer_indices is not None
            else len(self._all_token_ids)
        )
        evict_end = evict_start + tokens_evicted
        if evict_start < 0 or evict_end > live_len:
            self._kve_compact_replay_last_error = (
                f"evict range [{evict_start}, {evict_end}) outside live "
                f"writer len {live_len}"
            )
            return False
        if not self._ensure_compact_replay_timeline():
            return False
        replay_token_ids = self._kve_compact_replay_token_ids
        death_indices = self._kve_compact_replay_death_indices
        live_writer_indices = self._kve_compact_replay_live_writer_indices
        assert replay_token_ids is not None
        assert death_indices is not None
        assert live_writer_indices is not None
        death_idx = len(replay_token_ids)
        for writer_idx in live_writer_indices[evict_start:evict_end]:
            death_indices[writer_idx] = death_idx
        del live_writer_indices[evict_start:evict_end]
        self._kve_compact_replay_evictions += 1
        self._kve_compact_replay_last_error = None
        return True

    def compact_replay_snapshot(self) -> CompactReplaySnapshot | None:
        replay_token_ids = self._kve_compact_replay_token_ids
        death_indices = self._kve_compact_replay_death_indices
        live_writer_indices = self._kve_compact_replay_live_writer_indices
        if (
            replay_token_ids is None
            or death_indices is None
            or live_writer_indices is None
            or len(replay_token_ids) != len(death_indices)
        ):
            return None
        final_len = len(replay_token_ids)
        snapshot = CompactReplaySnapshot(
            token_ids=tuple(replay_token_ids),
            death_indices=tuple(
                final_len if death_idx is None else int(death_idx)
                for death_idx in death_indices
            ),
            live_writer_indices=tuple(live_writer_indices),
            evictions=self._kve_compact_replay_evictions,
        )
        return snapshot if snapshot.is_valid() else None

    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_segment_generated(self) -> int:
        # Tokens generated in the current streaming-session segment.
        # Equals num_total_generated for ordinary requests (base 0).
        return self.num_total_generated - self.segment_generated_base

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds()

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}
