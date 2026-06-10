# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.session.protocol import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionTurnRequest,
)
from vllm.entrypoints.openai.session.serving import (
    OpenAIServingSession,
    SessionError,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient

router = APIRouter()


def _serving(request: Request) -> OpenAIServingSession:
    serving = request.app.state.openai_serving_session
    if serving is None:
        raise SessionError(501, "session serving not initialized")
    return serving


def _error(exc: SessionError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.message}
    )


@router.post("/v1/session")
async def create_session(body: SessionCreateRequest, request: Request):
    try:
        serving = _serving(request)
        session_id = serving.create_session()
    except SessionError as exc:
        return _error(exc)
    return SessionCreateResponse(session_id=session_id, boot_id=serving.boot_id)


@router.post("/v1/session/{session_id}/turn")
async def session_turn(
    session_id: str, body: SessionTurnRequest, request: Request
):
    try:
        serving = _serving(request)
        return await serving.turn(session_id, body)
    except SessionError as exc:
        return _error(exc)


@router.delete("/v1/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    try:
        serving = _serving(request)
        await serving.delete_session(session_id, reason="client-delete")
    except SessionError as exc:
        return _error(exc)
    return JSONResponse(content={"deleted": session_id})


def attach_router(app: FastAPI):
    app.include_router(router)
    logger.info("Session API router attached")


def init_session_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
):
    model_name = getattr(args, "served_model_name", None) or getattr(
        args, "model", ""
    )
    if isinstance(model_name, list):
        model_name = model_name[0] if model_name else ""
    state.openai_serving_session = OpenAIServingSession(
        engine_client, model_name
    )
