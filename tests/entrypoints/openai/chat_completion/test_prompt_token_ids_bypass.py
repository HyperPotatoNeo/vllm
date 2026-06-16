# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the kv-eviction `prompt_token_ids` bypass on
`ChatCompletionRequest`.

Covers only the `render_chat` branch introduced by the kv-eviction
block-aligned message padding extension:

  - When `request.prompt_token_ids` is None, render_chat falls through
    to the normal chat-template / tokenizer path (unchanged behavior).
  - When `request.prompt_token_ids` is a non-empty list, render_chat
    skips chat-template rendering and returns a TokensInput whose
    prompt_token_ids are the caller-supplied ids verbatim.
  - When `request.prompt_token_ids` is an empty list, render_chat
    returns an ErrorResponse (empty is a caller bug, not "use default").

The bypass exists so clients can pre-tokenize + pad the prompt so that
`<|im_end|>` lands on a PagedAttention block boundary, making
turn-based KV eviction exact rather than inward-snapped. The test is
deliberately narrow: no model runs, no tokenizer — all side effects are
mocked at the OpenAIServingRender boundary.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.serve.render.serving import OpenAIServingRender


def _make_render_server() -> OpenAIServingRender:
    """Construct an OpenAIServingRender with mocked heavy deps.

    The only bits `render_chat`'s bypass branch actually touches are
    `create_error_response` (for the empty-ids case) and — after the
    branch — nothing. We leave the non-bypass path mocked to crash if
    called, so the test double-checks that the bypass doesn't leak into
    it.
    """
    server = OpenAIServingRender.__new__(OpenAIServingRender)
    # Minimal surface used by the bypass.
    server.create_error_response = MagicMock(
        side_effect=lambda msg, **kw: ErrorResponse(
            error=ErrorInfo(message=msg, type="BadRequest", code=400)
        )
    )
    # Trip wire: if bypass falls through to tokenizer-based rendering,
    # AttributeError surfaces immediately.
    server.renderer = MagicMock()
    server.renderer.tokenizer = None
    server.use_harmony = False
    server.tool_parser = None
    server.enable_auto_tools = False
    server.exclude_tools_when_tool_choice_none = False
    server.validate_chat_template = MagicMock()
    # preprocess_chat is the function the *fallthrough* path calls; making
    # it async-crash guarantees bypass tests never accidentally trigger it.
    server.preprocess_chat = AsyncMock(
        side_effect=AssertionError(
            "preprocess_chat must not be called when prompt_token_ids is set"
        )
    )
    return server


async def _test_render_chat_bypass_returns_tokens_input_verbatim():
    server = _make_render_server()
    ids = [1, 2, 3, 4, 5]
    req = ChatCompletionRequest(
        messages=[{"role": "system", "content": "s"}],
        model="dummy",
        prompt_token_ids=ids,
    )

    conversation, engine_inputs = await server.render_chat(req)

    assert len(engine_inputs) == 1
    engine_input = engine_inputs[0]
    assert engine_input["type"] == "token"
    # Exact, not a re-rendering of `messages`.
    assert engine_input["prompt_token_ids"] == ids
    # conversation metadata preserved for downstream tool / reasoning parsers.
    assert conversation == [{"role": "system", "content": "s"}]
    server.preprocess_chat.assert_not_called()


async def _test_render_chat_bypass_empty_list_is_error():
    server = _make_render_server()
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="dummy",
        prompt_token_ids=[],
    )

    result = await server.render_chat(req)
    assert isinstance(result, ErrorResponse)
    assert "empty" in result.error.message.lower()
    server.preprocess_chat.assert_not_called()


async def _test_render_chat_bypass_cache_salt_threaded():
    """cache_salt must flow into the TokensInput so prefix cache salting
    still works when the bypass is active."""
    server = _make_render_server()
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="dummy",
        prompt_token_ids=[10, 20, 30],
        cache_salt="secret",
    )

    _, engine_inputs = await server.render_chat(req)
    assert engine_inputs[0].get("cache_salt") == "secret"


def test_render_chat_bypass_returns_tokens_input_verbatim():
    asyncio.run(_test_render_chat_bypass_returns_tokens_input_verbatim())


def test_render_chat_bypass_empty_list_is_error():
    asyncio.run(_test_render_chat_bypass_empty_list_is_error())


def test_render_chat_bypass_cache_salt_threaded():
    asyncio.run(_test_render_chat_bypass_cache_salt_threaded())
