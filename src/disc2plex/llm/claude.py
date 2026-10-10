"""Claude backend via the official ``anthropic`` SDK.

Uses structured outputs (``output_config.format``) so the reply is schema-valid
JSON, an explicit effort level (Claude Opus 5.5 defaults to ``medium``; this is
a short classification task), and server-side refusal fallback so a rare
safety decline is retried on another model instead of failing the run.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import anthropic
from anthropic.types.beta import BetaMessageParam, BetaOutputConfigParam

from disc2plex.llm.base import LLMError, LLMResponse

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
Effort = Literal["low", "medium", "high", "xhigh", "max"]
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeBackend:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: Effort = "medium",
        max_tokens: int = 16000,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        # The zero-arg client reads ANTHROPIC_API_KEY (or an `ant auth login` profile).
        try:
            self._client = client or anthropic.Anthropic()
        except anthropic.AnthropicError as exc:
            raise LLMError(f"cannot create the Claude client: {exc}") from exc
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"claude/{self._model}/{self._effort}"

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        log.debug("Claude request (%s)\nSYSTEM:\n%s\nUSER:\n%s", self._model, system, user)
        messages: list[BetaMessageParam] = [{"role": "user", "content": user}]
        output_config: BetaOutputConfigParam = {
            "effort": self._effort,
            "format": {"type": "json_schema", "schema": schema},
        }
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
                output_config=output_config,
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("Claude rejected the API key; check ANTHROPIC_API_KEY") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude rate limit reached; try again later") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the Claude API: {exc}") from exc
        except anthropic.AnthropicError as exc:  # e.g. no credentials configured
            raise LLMError(f"Claude request failed: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Claude declined to answer (refusal)")
        text = next((b.text for b in response.content if b.type == "text"), "")
        if response.stop_reason == "max_tokens" or not text:
            raise LLMError(f"Claude returned no usable answer (stop: {response.stop_reason})")
        usage = response.usage
        log.debug(
            "Claude reply (%s in / %s out tokens):\n%s",
            usage.input_tokens,
            usage.output_tokens,
            text,
        )
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
