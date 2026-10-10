"""The small interface the content matcher needs from an LLM, plus a response cache.

Backends only have to turn (system prompt, user prompt, JSON schema) into the
model's raw JSON text. Parsing and validating that JSON is the caller's job,
so a backend that ignores the schema (some local models) is still safe.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from disc2plex.config import cache_dir

log = logging.getLogger(__name__)

_CACHE_VERSION = 1


class LLMError(Exception):
    """The backend could not produce an answer (network, auth, refusal...)."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class LLMBackend(Protocol):
    @property
    def name(self) -> str:
        """Identifies backend + model; part of the cache key."""
        ...

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        """Return the model's reply, which should be JSON matching ``schema``."""
        ...


def default_llm_cache_dir() -> Path:
    return cache_dir() / "llm"


class CachedLLM:
    """Wraps a backend so identical prompts are answered from disk.

    ``plan`` followed by ``apply`` runs the pipeline twice; without this cache
    every verified file would be paid for twice.
    """

    def __init__(self, backend: LLMBackend, directory: Path) -> None:
        self._backend = backend
        self._directory = directory

    @property
    def name(self) -> str:
        return self._backend.name

    def _file(self, system: str, user: str, schema: dict[str, Any]) -> Path:
        key = json.dumps([_CACHE_VERSION, self.name, system, user, schema], sort_keys=True)
        return self._directory / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        file = self._file(system, user, schema)
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            return LLMResponse(text=data["text"], model=data["model"], cached=True)
        except (OSError, ValueError, KeyError):
            pass
        response = self._backend.complete_json(system, user, schema)
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"text": response.text, "model": response.model}), "utf-8")
        tmp.replace(file)
        return response
