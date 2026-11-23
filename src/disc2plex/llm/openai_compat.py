"""Local LLM backend for any OpenAI-compatible chat endpoint.

Works with Ollama (``http://localhost:11434/v1``), LM Studio
(``http://localhost:1234/v1``), and llama.cpp's server. Requests a JSON-schema
response format; servers or models that ignore it are still safe because the
caller validates the JSON.

Thinking ("reasoning") models are asked not to think unless ``thinking`` is set:
on this short classification task a small model can think for thousands of
tokens and never reach its answer. The request is the chat-template switch
llama.cpp understands (``chat_template_kwargs.enable_thinking``; verified with
llama-server and Qwen 3.5); servers that don't know the field ignore it, and
the token cap below still stops a runaway reply.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from disc2plex.llm.base import LLMError, LLMResponse

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434/v1"
# Without thinking, a reply is one short JSON object per excerpt (about 100
# tokens each). A model still going after this many tokens isn't answering.
ANSWER_MAX_TOKENS = 2048


class OpenAICompatBackend:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        thinking: bool = False,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not model:
            raise LLMError("a local model name is required (--llm-model or DISC2PLEX_LLM_MODEL)")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )
        self._model = model
        self._base_url = base_url
        self._thinking = thinking
        self._timeout = timeout

    @property
    def name(self) -> str:
        # Part of the reply cache key: answers with and without thinking differ.
        return f"local/{self._model}" + ("/thinking" if self._thinking else "")

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
        }
        if not self._thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
            body["max_tokens"] = ANSWER_MAX_TOKENS
        log.debug("local LLM request to %s (%s)\n%s\n%s", self._base_url, self._model, system, user)
        try:
            response = self._http.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"local LLM at {self._base_url} did not answer within {self._timeout:.0f} s "
                "(model too slow for this machine, or thinking without a limit?)"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach local LLM at {self._base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(f"local LLM returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("local LLM returned an unexpected response shape") from exc
        if choice.get("finish_reason") == "length":
            # Every file would hit the same limit, so stop instead of retrying each one.
            raise LLMError(
                "local LLM ran out of tokens before finishing its answer; if the model is "
                "thinking, turn that off on the server (llama.cpp: --reasoning off)"
            )
        usage = data.get("usage") or {}
        log.debug("local LLM reply:\n%s", text)
        return LLMResponse(
            text=text,
            model=str(data.get("model") or self._model),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )
