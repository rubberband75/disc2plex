"""Settings and environment loading.

Milestone 5 only needs secrets from the environment / ``.env``; the TOML config
file (CLAUDE.md section 8) arrives in milestone 10.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path


def cache_dir() -> Path:
    """Base cache directory: ``$XDG_CACHE_HOME/disc2plex`` (default ``~/.cache/disc2plex``)."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "disc2plex"


def config_dir() -> Path:
    """Base config directory: ``$XDG_CONFIG_HOME/disc2plex`` (default ``~/.config/disc2plex``)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
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
