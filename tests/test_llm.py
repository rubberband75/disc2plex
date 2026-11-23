"""Tests for LLM backends and the response cache. No network: fakes and mock transports."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from disc2plex.llm.base import CachedLLM, LLMError, LLMResponse
from disc2plex.llm.claude import FALLBACK_BETA, ClaudeBackend
from disc2plex.llm.openai_compat import ANSWER_MAX_TOKENS, OpenAICompatBackend

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


# --- cache --------------------------------------------------------------------


class CountingBackend:
    name = "fake/model"

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text='{"x": 1}', model="fake", input_tokens=10, output_tokens=2)


def test_cache_reuses_identical_prompts(tmp_path: Path) -> None:
    backend = CountingBackend()
    llm = CachedLLM(backend, tmp_path)
    first = llm.complete_json("sys", "user", SCHEMA)
    second = CachedLLM(backend, tmp_path).complete_json("sys", "user", SCHEMA)
    assert backend.calls == 1
    assert not first.cached and second.cached
    assert second.text == '{"x": 1}'
    assert second.input_tokens == 0  # cache hits cost nothing


def test_cache_key_includes_prompt_and_schema(tmp_path: Path) -> None:
    backend = CountingBackend()
    llm = CachedLLM(backend, tmp_path)
    llm.complete_json("sys", "user", SCHEMA)
    llm.complete_json("sys", "other user", SCHEMA)
    llm.complete_json("sys", "user", {**SCHEMA, "required": []})
    assert backend.calls == 3


def test_cache_does_not_store_failures(tmp_path: Path) -> None:
    class Failing(CountingBackend):
        def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
            raise LLMError("down")

    with pytest.raises(LLMError):
        CachedLLM(Failing(), tmp_path).complete_json("s", "u", SCHEMA)
    assert list(tmp_path.iterdir()) == []


# --- Claude backend (fake SDK client) -------------------------------------------


class FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def fake_client(messages: FakeMessages) -> anthropic.Anthropic:
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))  # type: ignore[return-value]


def claude_response(text: str = '{"x": 1}', stop: str = "end_turn") -> Any:
    return SimpleNamespace(
        stop_reason=stop,
        model="claude-opus-5-5",
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=1200, output_tokens=80),
    )


def test_claude_request_shape() -> None:
    messages = FakeMessages(claude_response())
    backend = ClaudeBackend(client=fake_client(messages))
    result = backend.complete_json("system prompt", "user prompt", SCHEMA)
    kwargs = messages.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["system"] == "system prompt"
    assert kwargs["messages"] == [{"role": "user", "content": "user prompt"}]
    assert kwargs["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert kwargs["betas"] == [FALLBACK_BETA]
    assert kwargs["fallbacks"] == "default"
    assert "thinking" not in kwargs  # Opus 5.5: adaptive by default; disabling is a 400
    assert (result.text, result.input_tokens, result.output_tokens) == ('{"x": 1}', 1200, 80)
    assert backend.name == "claude/claude-opus-5-5/medium"


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_claude_unusable_stop_reasons(stop: str) -> None:
    backend = ClaudeBackend(client=fake_client(FakeMessages(claude_response(stop=stop))))
    with pytest.raises(LLMError):
        backend.complete_json("s", "u", SCHEMA)


def test_claude_missing_text_block() -> None:
    response = claude_response()
    response.content = [SimpleNamespace(type="thinking")]
    backend = ClaudeBackend(client=fake_client(FakeMessages(response)))
    with pytest.raises(LLMError, match="no usable answer"):
        backend.complete_json("s", "u", SCHEMA)


def test_claude_sdk_errors_become_llm_errors() -> None:
    backend = ClaudeBackend(
        client=fake_client(FakeMessages(error=anthropic.AnthropicError("no credentials")))
    )
    with pytest.raises(LLMError, match="no credentials"):
        backend.complete_json("s", "u", SCHEMA)


# --- OpenAI-compatible backend (mock HTTP) --------------------------------------


def openai_backend(handler: Any, model: str = "llama3.1:8b") -> OpenAICompatBackend:
    return OpenAICompatBackend(
        model,
        base_url="http://localhost:11434/v1",
        api_key="local-key",
        transport=httpx.MockTransport(handler),
    )


def test_openai_compat_request_and_response() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "llama3.1:8b",
                "choices": [{"message": {"content": '{"x": 2}'}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 30},
            },
        )

    result = openai_backend(handler).complete_json("sys", "usr", SCHEMA)
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer local-key"
    assert body["temperature"] == 0
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    # By default the model is asked not to think, and a runaway reply is capped.
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == ANSWER_MAX_TOKENS
    assert (result.text, result.input_tokens, result.output_tokens) == ('{"x": 2}', 900, 30)


def test_openai_compat_thinking_leaves_the_server_in_charge() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    backend = OpenAICompatBackend("qwen", thinking=True, transport=httpx.MockTransport(handler))
    backend.complete_json("s", "u", SCHEMA)
    body = json.loads(seen[0].content)
    assert "chat_template_kwargs" not in body
    assert "max_tokens" not in body
    # A different name keeps thinking and non-thinking answers apart in the cache.
    assert backend.name == "local/qwen/thinking"
    assert openai_backend(handler, "qwen").name == "local/qwen"


def test_openai_compat_truncated_reply_is_an_error() -> None:
    cut_off = httpx.Response(
        200, json={"choices": [{"finish_reason": "length", "message": {"content": '{"x"'}}]}
    )
    with pytest.raises(LLMError, match="ran out of tokens"):
        openai_backend(lambda _r: cut_off).complete_json("s", "u", SCHEMA)


def test_openai_compat_timeout_is_not_reported_as_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(LLMError, match="did not answer within 300 s"):
        openai_backend(handler).complete_json("s", "u", SCHEMA)


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(500, text="boom"), "HTTP 500"),
        (httpx.Response(200, json={"unexpected": True}), "unexpected response shape"),
        (httpx.Response(200, content=b"not json"), "unexpected response shape"),
    ],
)
def test_openai_compat_errors(response: httpx.Response, match: str) -> None:
    with pytest.raises(LLMError, match=match):
        openai_backend(lambda _r: response).complete_json("s", "u", SCHEMA)


def test_openai_compat_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError, match="could not reach local LLM"):
        openai_backend(handler).complete_json("s", "u", SCHEMA)


def test_openai_compat_requires_model() -> None:
    with pytest.raises(LLMError, match="model name is required"):
        OpenAICompatBackend("")
