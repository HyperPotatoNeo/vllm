# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire protocol for persistent streaming sessions (one request/episode).

A session keeps ONE live engine request across many conversation turns.
Each turn the client POSTs the new pre-tokenized fragment; the engine
prefills it onto the same KV stream (no re-prefill of history) and decodes
until the stop token; the segment result is returned in the response.
"""

from typing import Any

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    model: str | None = None


class SessionCreateResponse(BaseModel):
    session_id: str
    # Identifies the server process; a turn POST with a stale boot_id gets
    # HTTP 410 so the client can fall back to the per-call path.
    boot_id: str


class SessionTurnRequest(BaseModel):
    # Client-tokenized fragment to append to the stream (the client owns
    # template + filler math, exactly like the per-call Phase 4 path).
    prompt_token_ids: list[int]
    # Monotonic per-session turn counter for idempotency: a retried POST of
    # the last turn returns the cached segment result instead of corrupting
    # the episode with a duplicate update.
    turn_idx: int
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    min_tokens: int = 0
    presence_penalty: float = 0.0
    repetition_penalty: float | None = None
    seed: int | None = None
    logprobs: bool = True
    boot_id: str | None = None
    # Extra per-request args (recall directives etc.) — same namespace as
    # the per-call `vllm_xargs` extra_body field.
    vllm_xargs: dict[str, Any] | None = None


class SessionSegmentResponse(BaseModel):
    session_id: str
    turn_idx: int
    text: str
    token_ids: list[int]
    # Chosen-token logprobs aligned with token_ids (empty if disabled).
    logprobs: list[float] = Field(default_factory=list)
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    # Full cumulative compaction event list as of segment end (client diffs).
    compaction_events: list[dict[str, Any]] = Field(default_factory=list)
    # Filler tokens the engine auto-padded after the stop token (block-
    # aligned finish). Already part of the parked KV stream; the client must
    # fold them into its stream-offset math, per-call-parity style.
    padding_token_ids: list[int] = Field(default_factory=list)
    # Movement verdict dict ({kind, spans, resident, h2d}) — same payload as
    # the per-call ChatCompletion extension field. The engine emits a dict;
    # str is tolerated for forward compatibility.
    managed_context_restore_kind: dict[str, Any] | str | None = None
    # Parked stream length (tokens whose KV is live server-side). The client
    # asserts its own offset math equals this every turn — fail loud.
    stream_len: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
