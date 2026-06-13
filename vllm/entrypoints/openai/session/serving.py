# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent streaming sessions: one live engine request per episode.

Each session wraps one AsyncLLM.generate() call whose prompt is an async
generator of StreamingInput chunks (the upstream streaming-inputs wire:
updates travel as duplicate ADDs; the scheduler extends the SAME KV stream,
so history is never re-prefilled). Turn cadence is request/response: the
client POSTs a pre-tokenized fragment, we feed it to the generator and
await the segment that ends at the next stop token.
"""

import asyncio
import os
import time
import uuid
from typing import Any

from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.entrypoints.openai.session.protocol import (
    SessionSegmentResponse,
    SessionTurnRequest,
)
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams

logger = init_logger(__name__)

# Parked sessions whose client never returns are aborted after this long.
SESSION_IDLE_TTL_S = float(os.environ.get("KVE_SESSION_IDLE_TTL_S", "600"))


def _include_evicted_token_ids() -> bool:
    replay_mode = (
        os.environ.get("KVE_COMPACT_REPLAY_REFILL_MODE", "").strip().lower()
    )
    truthy = ("1", "true", "yes", "on")
    return (
        os.environ.get("KVE_OPENAI_INCLUDE_EVICTED_TOKEN_IDS", "0") == "1"
        or replay_mode not in ("", "0", "false", "no", "off")
        or os.environ.get(
            "KVE_MANAGED_CONTEXT_REPLAY_ONLY_ARCHIVE", "0"
        ).strip().lower() in truthy
        or os.environ.get(
            "KVE_MANAGED_CONTEXT_SKIP_CPU_ARCHIVE_FOR_REPLAY", "0"
        ).strip().lower() in truthy
    )


def _compaction_event_to_dict(e: Any) -> dict[str, Any]:
    """Mirror of the per-call CompactionEventPayload field mapping."""
    return {
        "num_output_tokens_at_compaction": e.num_output_tokens_at_compaction,
        "tokens_evicted": e.tokens_evicted,
        "position_offset_after": e.position_offset_after,
        "num_prompt_tokens": e.num_prompt_tokens,
        "evict_start": e.evict_start,
        "evicted_token_ids": (
            list(getattr(e, "evicted_token_ids", []) or [])
            if _include_evicted_token_ids()
            else []
        ),
        "last_turn_evicted": int(getattr(e, "last_turn_evicted", -1)),
        "num_turns_evicted_after": int(
            getattr(e, "num_turns_evicted_after", 0)
        ),
        "kept_indices": list(e.kept_indices),
        "kept_token_ids": list(e.kept_token_ids),
        "new_user_fragment_len": int(
            getattr(e, "new_user_fragment_len", 0) or 0
        ),
        "archived_span_ids": [
            str(x) for x in getattr(e, "archived_span_ids", []) or []
        ],
        "writer_len_at_compaction": int(
            getattr(e, "writer_len_at_compaction", 0) or 0
        ),
        "archived_span_bounds": [
            int(x) for x in getattr(e, "archived_span_bounds", []) or []
        ],
        "event_kind": int(getattr(e, "event_kind", 0) or 0),
        "restored_span_ids": [
            str(x) for x in getattr(e, "restored_span_ids", []) or []
        ],
        "visibility_boundary_computed": int(
            getattr(e, "visibility_boundary_computed", -1)
        ),
        "restored_span_token_ids": [
            int(x) for x in getattr(e, "restored_span_token_ids", []) or []
        ],
        "restored_span_pos_start": int(
            getattr(e, "restored_span_pos_start", -1)
        ),
    }


class SessionError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class _Session:
    """State for one live episode (= one engine request)."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.input_queue: asyncio.Queue[StreamingInput | None] = asyncio.Queue()
        self.consumer_task: asyncio.Task | None = None
        self.started = False
        self.dead = False
        self.death_reason: str | None = None
        # Serializes turns: one in-flight turn per session.
        self.turn_lock = asyncio.Lock()
        self.segment_future: asyncio.Future[SessionSegmentResponse] | None = None
        self.last_turn_idx: int | None = None
        self.last_response: SessionSegmentResponse | None = None
        self.last_activity = time.monotonic()
        # Stream-offset accounting (KV tokens live server-side).
        self.stream_len = 0
        self.fed_tokens = 0
        # Per-segment accumulators (reset at segment start).
        self.seg_text = ""
        self.seg_token_ids: list[int] = []
        self.seg_logprobs: list[float] = []
        self.seg_padding: list[int] = []
        self.compaction_events: list[dict[str, Any]] = []
        self.restore_kind: str | None = None

    def reset_segment(self):
        self.seg_text = ""
        self.seg_token_ids = []
        self.seg_logprobs = []
        self.seg_padding = []


class OpenAIServingSession:
    def __init__(self, engine_client: EngineClient, model_name: str):
        self.engine_client = engine_client
        self.model_name = model_name
        self.boot_id = uuid.uuid4().hex
        self.sessions: dict[str, _Session] = {}
        self._reaper_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def create_session(self) -> str:
        session_id = f"sess-{uuid.uuid4().hex}"
        self.sessions[session_id] = _Session(session_id)
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.get_running_loop().create_task(
                self._reap_idle_sessions()
            )
        logger.info("[SESSION] created %s", session_id)
        return session_id

    async def delete_session(self, session_id: str, reason: str = "client"):
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        session.dead = True
        session.death_reason = reason
        # Closing the input generator triggers the finish sentinel; the
        # abort covers a session that is mid-decode.
        session.input_queue.put_nowait(None)
        try:
            await self.engine_client.abort(session_id)
        except Exception:
            logger.exception("[SESSION] abort failed for %s", session_id)
        if session.consumer_task is not None:
            session.consumer_task.cancel()
        if (
            session.segment_future is not None
            and not session.segment_future.done()
        ):
            session.segment_future.set_exception(
                SessionError(410, f"session {session_id} ended: {reason}")
            )
        logger.info("[SESSION] deleted %s (%s)", session_id, reason)

    async def _reap_idle_sessions(self):
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            for sid, session in list(self.sessions.items()):
                if (
                    session.turn_lock.locked()
                    or now - session.last_activity < SESSION_IDLE_TTL_S
                ):
                    continue
                logger.warning("[SESSION] reaping idle session %s", sid)
                await self.delete_session(sid, reason="idle-ttl")

    # ------------------------------------------------------------------
    # Turn handling
    # ------------------------------------------------------------------
    async def turn(
        self, session_id: str, request: SessionTurnRequest
    ) -> SessionSegmentResponse:
        session = self.sessions.get(session_id)
        if session is None or session.dead:
            raise SessionError(410, f"unknown or dead session {session_id}")
        if request.boot_id is not None and request.boot_id != self.boot_id:
            raise SessionError(410, "server restarted (boot_id mismatch)")

        async with session.turn_lock:
            session.last_activity = time.monotonic()
            # Idempotent retry of the last turn returns the cached result.
            if (
                session.last_turn_idx is not None
                and request.turn_idx == session.last_turn_idx
            ):
                assert session.last_response is not None
                return session.last_response
            expected = (
                0 if session.last_turn_idx is None else session.last_turn_idx + 1
            )
            if request.turn_idx != expected:
                raise SessionError(
                    409,
                    f"turn_idx {request.turn_idx} out of order "
                    f"(expected {expected})",
                )

            params = self._build_sampling_params(request)
            chunk = StreamingInput(
                prompt={"prompt_token_ids": request.prompt_token_ids},
                sampling_params=params,
            )
            loop = asyncio.get_running_loop()
            session.segment_future = loop.create_future()
            session.reset_segment()
            session.fed_tokens += len(request.prompt_token_ids)
            session.stream_len += len(request.prompt_token_ids)

            if not session.started:
                session.started = True
                result_gen = self.engine_client.generate(
                    prompt=self._input_gen(session),
                    sampling_params=params,
                    request_id=session_id,
                )
                session.consumer_task = loop.create_task(
                    self._consume(session, result_gen)
                )
            session.input_queue.put_nowait(chunk)

            try:
                response = await session.segment_future
            except asyncio.CancelledError:
                raise SessionError(410, f"session {session_id} aborted")
            session.last_activity = time.monotonic()
            response.turn_idx = request.turn_idx
            session.last_turn_idx = request.turn_idx
            session.last_response = response
            return response

    def _build_sampling_params(
        self, request: SessionTurnRequest
    ) -> SamplingParams:
        return SamplingParams.from_optional(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k is not None else 0,
            min_p=request.min_p if request.min_p is not None else 0.0,
            min_tokens=request.min_tokens,
            presence_penalty=request.presence_penalty,
            repetition_penalty=(
                request.repetition_penalty
                if request.repetition_penalty is not None
                else 1.0
            ),
            seed=request.seed,
            logprobs=0 if request.logprobs else None,
            output_kind=RequestOutputKind.DELTA,
            extra_args=dict(request.vllm_xargs) if request.vllm_xargs else None,
        )

    async def _input_gen(self, session: _Session):
        while True:
            chunk = await session.input_queue.get()
            if chunk is None:
                return
            yield chunk

    # ------------------------------------------------------------------
    # Output consumption
    # ------------------------------------------------------------------
    async def _consume(self, session: _Session, result_gen):
        try:
            async for output in result_gen:
                if output.compaction_events is not None:
                    session.compaction_events = [
                        _compaction_event_to_dict(e)
                        for e in output.compaction_events
                    ]
                if output.managed_context_restore_kind is not None:
                    session.restore_kind = output.managed_context_restore_kind
                if not output.outputs:
                    continue
                completion = output.outputs[0]
                session.seg_text += completion.text
                new_ids = list(completion.token_ids)
                session.seg_token_ids.extend(new_ids)
                if completion.logprobs:
                    # DELTA mode: logprobs list is aligned with the
                    # token_ids delta.
                    for token_id, pos_logprobs in zip(
                        new_ids, completion.logprobs
                    ):
                        entry = pos_logprobs.get(token_id)
                        session.seg_logprobs.append(
                            entry.logprob if entry is not None else 0.0
                        )
                if output.padding_token_ids:
                    session.seg_padding = list(output.padding_token_ids)

                if completion.finish_reason is not None:
                    self._finish_segment(session, completion)
                if output.finished:
                    # Final sentinel or engine-side finish (abort, length on
                    # a non-resumable path, ...). The session is over.
                    break
        except Exception as exc:
            logger.exception("[SESSION] consumer died for %s", session.session_id)
            session.dead = True
            session.death_reason = repr(exc)
            if (
                session.segment_future is not None
                and not session.segment_future.done()
            ):
                session.segment_future.set_exception(
                    SessionError(500, f"engine error: {exc!r}")
                )
        finally:
            session.dead = True

    def _finish_segment(self, session: _Session, completion):
        # Stream accounting: with block-aligned auto-pad the stop token and
        # filler were forwarded into KV before parking (per-call parity);
        # without it the engine discards the final sampled token at the next
        # update, so it is not part of the surviving stream.
        generated = len(session.seg_token_ids)
        padding = len(session.seg_padding)
        if padding:
            session.stream_len += generated + padding
        else:
            session.stream_len += generated - 1
        response = SessionSegmentResponse(
            session_id=session.session_id,
            turn_idx=-1,  # stamped by turn()
            text=session.seg_text,
            token_ids=list(session.seg_token_ids),
            logprobs=list(session.seg_logprobs),
            finish_reason=completion.finish_reason,
            stop_reason=completion.stop_reason,
            compaction_events=list(session.compaction_events),
            padding_token_ids=list(session.seg_padding),
            managed_context_restore_kind=session.restore_kind,
            stream_len=session.stream_len,
            prompt_tokens=session.fed_tokens,
            completion_tokens=len(session.seg_token_ids),
        )
        if (
            session.segment_future is not None
            and not session.segment_future.done()
        ):
            session.segment_future.set_result(response)
