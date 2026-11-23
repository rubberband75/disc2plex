"""Settings and environment loading.

Two sources, kept apart on purpose:

- ``config.toml`` (``~/.config/disc2plex/config.toml`` or ``$DISC2PLEX_CONFIG``):
  preferences such as the library root, default LLM, and thresholds. CLI flags
  override it; it overrides built-in defaults.
- the environment / ``.env``: secrets (API keys, the Plex token). Never in TOML.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import MutableMapping
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


def cache_dir() -> Path:
    """Base cache directory: ``$XDG_CACHE_HOME/disc2plex`` (default ``~/.cache/disc2plex``)."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "disc2plex"


def config_dir() -> Path:
    """Base config directory: ``$XDG_CONFIG_HOME/disc2plex`` (default ``~/.config/disc2plex``)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "disc2plex"


def data_dir() -> Path:
    """Base data directory: ``$XDG_DATA_HOME/disc2plex`` (default ``~/.local/share/disc2plex``)."""
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "disc2plex"


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse simple ``KEY=VALUE`` lines. Supports comments, ``export``, and quoted values."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load_dotenv(
    paths: list[Path] | None = None, environ: MutableMapping[str, str] | None = None
) -> None:
    """Load ``.env`` files into the environment without overriding variables already set.

    Defaults to ``./.env`` then ``~/.config/disc2plex/.env``; earlier files win.
    """
    env = os.environ if environ is None else environ
    for path in paths if paths is not None else [Path(".env"), config_dir() / ".env"]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for key, value in parse_dotenv(text).items():
            env.setdefault(key, value)


# --- config.toml --------------------------------------------------------------------


class ConfigError(Exception):
    """The config file exists but can't be used (bad TOML, unknown key, bad value)."""


class _Section(BaseModel):
    # Unknown keys are errors so a typo doesn't silently do nothing.
    model_config = ConfigDict(extra="forbid", frozen=True)


class LibrarySettings(_Section):
    root: Path | None = None
    action: Literal["move", "hardlink", "copy"] = "move"
    include_titles: bool = True
    include_tmdb_id: bool = True
    extras: Literal["review", "season0", "skip"] = "review"
    review: Literal["move", "leave"] = "move"


class MatchingSettings(_Section):
    order: str = "aired"
    verify: Literal["none", "low", "all"] = "low"
    llm: Literal["claude", "local", "none"] = "none"
    whisper_model: str = "small"
    lang: str = "en"
    confident: float = Field(default=0.85, gt=0, le=1)
    review: float = Field(default=0.5, ge=0, lt=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.review >= self.confident:
            raise ValueError("matching.review must be lower than matching.confident")
        return self


class ClaudeSettings(_Section):
    model: str | None = None  # default: the backend's (claude-opus-5-5)


class LocalLLMSettings(_Section):
    base_url: str | None = None
    model: str | None = None
    # Let a reasoning model think before answering. Off by default: small models
    # can think for minutes on this task without reaching an answer.
    thinking: bool = False


class PlexSettings(_Section):
    url: str | None = None
    # Library section ID; found automatically from the library path when unset.
    section: int | None = None
    # The library root as the Plex server sees it, if it differs (e.g. a share
    # mounted at /mnt/plex here but /data on the server).
    library_path: str | None = None
    scan_after_apply: bool = False


class Settings(_Section):
    library: LibrarySettings = LibrarySettings()
    matching: MatchingSettings = MatchingSettings()
    claude: ClaudeSettings = ClaudeSettings()
    local_llm: LocalLLMSettings = LocalLLMSettings()
    plex: PlexSettings = PlexSettings()


def config_path() -> Path:
    """``$DISC2PLEX_CONFIG`` if set, else ``~/.config/disc2plex/config.toml``."""
    override = os.environ.get("DISC2PLEX_CONFIG")
    return Path(override).expanduser() if override else config_dir() / "config.toml"


def parse_settings(text: str, source: str = "config.toml") -> Settings:
    """Parse and validate config TOML. Raises ``ConfigError`` with a readable message."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{source}: invalid TOML: {exc}") from exc
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '(top level)'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"{source}: {problems}") from exc


def load_settings(path: Path | None = None) -> Settings:
    """Load the config file, or built-in defaults if there isn't one."""
    path = path or config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Settings()
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    settings = parse_settings(text, str(path))
    if settings.library.root is not None:
        library = settings.library.model_copy(update={"root": settings.library.root.expanduser()})
        settings = settings.model_copy(update={"library": library})
    return settings


CONFIG_TEMPLATE = """\
# disc2plex configuration. Command-line flags override these values.
# Secrets (TMDB_API_TOKEN, ANTHROPIC_API_KEY, PLEX_TOKEN) belong in the
# environment or a .env file, not here.

[library]
# Plex TV library root; used when --output isn't given.
# root = "/mnt/plex/TV Shows"
# action = "move"            # move | hardlink | copy
# include_titles = true      # "... - s01e01 - Title.mkv"
# include_tmdb_id = true     # "Show (2005) {tmdb-12345}"
# extras = "review"          # review | season0 | skip
# review = "move"            # move: uncertain files -> _review/<disc>/; leave: in place

[matching]
# order = "aired"            # aired | dvd | <episode-group-id>
# verify = "low"             # none | low | all
# llm = "none"               # none | claude | local (claude sends transcripts to the API)
# whisper_model = "small"    # tiny | base | small | medium | large-v3 ...
# lang = "en"
# confident = 0.85           # at or above: placed in the library
# review = 0.5               # between review and confident: needs review; below: unmatched

[claude]
# Key: ANTHROPIC_API_KEY. --llm-model overrides the model.
# model = "claude-opus-5-5"

[local_llm]
# OpenAI-compatible server (Ollama, llama.cpp, LM Studio). Key: DISC2PLEX_LLM_API_KEY.
# DISC2PLEX_LLM_BASE_URL / DISC2PLEX_LLM_MODEL in the environment override these.
# base_url = "http://localhost:11434/v1"
# model = "llama3.1:8b"
# thinking = false           # true lets a reasoning model think first (much slower)

[plex]
# Optional: ask Plex to scan the show folder after apply. Token: PLEX_TOKEN.
# url = "http://localhost:32400"
# section = 2                # library section ID (see `disc2plex plex sections`)
# library_path = "/data/TV Shows"   # library root as the server sees it, if different
# scan_after_apply = false
"""
