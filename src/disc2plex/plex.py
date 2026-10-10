"""Optional Plex Media Server integration: ask Plex to scan a folder after apply.

Only two read-only-ish endpoints are used: list library sections, and
``/library/sections/{id}/refresh?path=...`` (a partial scan of one folder).
Nothing in Plex's database is edited. The token comes from ``PLEX_TOKEN``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

log = logging.getLogger(__name__)


class PlexError(Exception):
    """Plex couldn't be reached or refused the request."""


@dataclass(frozen=True)
class PlexSection:
    key: int
    title: str
    type: str  # "show", "movie", "artist", ...
    locations: tuple[str, ...] = ()


def parse_sections(payload: dict[str, Any]) -> list[PlexSection]:
    """Parse ``GET /library/sections`` JSON."""
    directories = payload.get("MediaContainer", {}).get("Directory", [])
    return [
        PlexSection(
            key=int(d["key"]),
            title=str(d.get("title", "")),
            type=str(d.get("type", "")),
            locations=tuple(str(loc["path"]) for loc in d.get("Location", []) if "path" in loc),
        )
        for d in directories
    ]


def server_path(local: Path, output_root: Path, library_path: str | None) -> str:
    """``local`` (inside ``output_root``) as the Plex server sees it.

    ``library_path`` is the library root on the server, for when it's mounted
    elsewhere there (e.g. ``/mnt/plex/TV`` here, ``/data/TV`` on the server).
    """
    if library_path is None:
        return str(local)
    relative = local.relative_to(output_root)
    return str(PurePosixPath(library_path, *relative.parts))


def _contains(location: str, path: str) -> bool:
    loc = location.rstrip("/")
    return path == loc or path.startswith(loc + "/")


def find_section(sections: Sequence[PlexSection], path: str) -> PlexSection | None:
    """The TV section whose folder contains ``path`` (the most specific one)."""
    best: tuple[int, PlexSection] | None = None
    for section in sections:
        if section.type != "show":
            continue
        for loc in section.locations:
            if _contains(loc, path) and (best is None or len(loc) > best[0]):
                best = (len(loc), section)
    return best[1] if best else None


class PlexClient:
    def __init__(
        self,
        url: str,
        token: str,
        *,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not token:
            raise PlexError("PLEX_TOKEN is not set (add it to your environment or .env)")
        self._client = httpx.Client(
            base_url=url.rstrip("/"),
            headers={"X-Plex-Token": token, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> PlexClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._client.close()

    def _get(self, path: str, params: dict[str, str] | None = None) -> httpx.Response:
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise PlexError(f"could not reach Plex at {self._client.base_url}: {exc}") from exc
        if response.status_code == 401:
            raise PlexError("Plex rejected the token; check PLEX_TOKEN")
        if response.status_code >= 400:
            raise PlexError(f"Plex returned HTTP {response.status_code} for {path}")
        return response

    def sections(self) -> list[PlexSection]:
        try:
            return parse_sections(self._get("/library/sections").json())
        except (ValueError, KeyError, TypeError) as exc:
            raise PlexError(f"unexpected response from Plex: {exc}") from exc

    def refresh(self, section: int, path: str) -> None:
        """Ask Plex to scan one folder of a library section."""
        log.info("Plex: scanning %s in section %s", path, section)
        self._get(f"/library/sections/{section}/refresh", params={"path": path})
