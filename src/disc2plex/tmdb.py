"""Stage 3: TMDB v3 client, on-disk response cache, and show/season resolution.

Network access is isolated in :class:`TMDBClient`; everything that turns raw
JSON into models or picks among candidates is a pure function.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import statistics
import time
import unicodedata
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from disc2plex.config import cache_dir
from disc2plex.models import (
    EpisodeGroupSummary,
    EpisodeInfo,
    SeasonEpisodes,
    SeasonSummary,
    ShowInfo,
    ShowSummary,
)

log = logging.getLogger(__name__)

API_BASE = "https://api.themoviedb.org/3"
DEFAULT_TTL_SECONDS = 7 * 24 * 3600
DVD_GROUP_TYPE = 3
MAX_RATE_LIMIT_RETRIES = 2
MAX_RETRY_WAIT_SECONDS = 10.0

Json = dict[str, Any]

_API_KEY = re.compile(r"^[0-9a-f]{32}$")
_GROUP_ID = re.compile(r"^[0-9a-f]{24}$")


class TMDBError(Exception):
    """A TMDB request failed in a way the user needs to know about."""


class TMDBNotFoundError(TMDBError):
    pass


class AmbiguousShowError(TMDBError):
    """Search found several plausible shows; the user must pick one."""

    def __init__(self, query: str, candidates: Sequence[ShowSummary]) -> None:
        self.query = query
        self.candidates = tuple(candidates)
        lines = [f"Several shows match {query!r}; pass --tmdb-id (or --year) to choose:"]
        lines += [f"  {c.tmdb_id:>8}  {c.name} ({c.year or '?'})" for c in self.candidates]
        super().__init__("\n".join(lines))


# --- cache --------------------------------------------------------------------


def default_tmdb_cache_dir() -> Path:
    return cache_dir() / "tmdb"


class TMDBCache:
    """One JSON file per (path, params) request, expiring after ``ttl_seconds``.

    Keys never include credentials, so cache files are safe to inspect or share.
    """

    def __init__(
        self,
        directory: Path,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.directory = directory
        self.ttl_seconds = ttl_seconds
        self._clock = clock

    def _file(self, path: str, params: dict[str, str]) -> Path:
        key = json.dumps([path, sorted(params.items())])
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def get(self, path: str, params: dict[str, str]) -> Json | None:
        file = self._file(path, params)
        try:
            entry = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if self._clock() - float(entry.get("fetched_at", 0)) > self.ttl_seconds:
            return None
        data: Json = entry["data"]
        return data

    def put(self, path: str, params: dict[str, str], data: Json) -> None:
        file = self._file(path, params)
        file.parent.mkdir(parents=True, exist_ok=True)
        entry = {"fetched_at": self._clock(), "path": path, "params": params, "data": data}
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry), encoding="utf-8")
        tmp.replace(file)


# --- pure parsing -------------------------------------------------------------


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _year(value: Any) -> int | None:
    parsed = _parse_date(value)
    return parsed.year if parsed else None


def _runtime(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and value > 0 else None


def parse_search_results(data: Json) -> list[ShowSummary]:
    return [
        ShowSummary(
            tmdb_id=int(r["id"]),
            name=str(r.get("name") or r.get("original_name") or "?"),
            year=_year(r.get("first_air_date")),
            original_name=r.get("original_name"),
            overview=str(r.get("overview") or ""),
            popularity=float(r.get("popularity") or 0.0),
        )
        for r in data.get("results", [])
        if "id" in r
    ]


def parse_episode(data: Json) -> EpisodeInfo:
    return EpisodeInfo(
        season=int(data["season_number"]),
        episode=int(data["episode_number"]),
        title=str(data.get("name") or ""),
        overview=str(data.get("overview") or ""),
        runtime_minutes=_runtime(data.get("runtime")),
        air_date=_parse_date(data.get("air_date")),
        tmdb_id=data.get("id"),
    )


def parse_show(data: Json, groups: Json | None = None) -> ShowInfo:
    runtimes = [r for r in data.get("episode_run_time") or [] if _runtime(r)]
    return ShowInfo(
        tmdb_id=int(data["id"]),
        name=str(data.get("name") or data.get("original_name") or "?"),
        year=_year(data.get("first_air_date")),
        seasons=tuple(
            SeasonSummary(
                number=int(s["season_number"]),
                name=str(s.get("name") or f"Season {s['season_number']}"),
                episode_count=int(s.get("episode_count") or 0),
            )
            for s in data.get("seasons", [])
        ),
        episode_groups=tuple(
            EpisodeGroupSummary(
                id=str(g["id"]),
                name=str(g.get("name") or ""),
                type=int(g.get("type") or 0),
                group_count=int(g.get("group_count") or 0),
                episode_count=int(g.get("episode_count") or 0),
                description=str(g.get("description") or ""),
            )
            for g in (groups or {}).get("results", [])
        ),
        default_runtime_minutes=statistics.median(runtimes) if runtimes else None,
    )


def parse_season(data: Json) -> list[EpisodeInfo]:
    episodes = [parse_episode(e) for e in data.get("episodes", [])]
    return sorted(episodes, key=lambda e: e.episode)


# --- pure selection -----------------------------------------------------------


def _normalize_title(text: str) -> str:
    """Lower-case, strip accents and punctuation: "Marvel's Agents of S.H.I.E.L.D." ->
    "marvels agents of shield"."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", ascii_text.lower())).strip()


def choose_show(query: str, results: Sequence[ShowSummary], year: int | None) -> ShowSummary:
    """Pick the search result the user meant, or raise if it is genuinely ambiguous.

    Accepts the only result, or the only result whose normalized name equals the
    query (and whose year matches, if a year was given). Anything else raises
    :class:`AmbiguousShowError` listing the top candidates.
    """
    if year is not None:
        results = [r for r in results if r.year is None or r.year == year] or list(results)
    if not results:
        raise TMDBNotFoundError(f"No TMDB show found for {query!r}")
    if len(results) == 1:
        return results[0]
    wanted = _normalize_title(query)
    exact = [
        r
        for r in results
        if wanted in (_normalize_title(r.name), _normalize_title(r.original_name or ""))
    ]
    if len(exact) == 1:
        return exact[0]
    raise AmbiguousShowError(query, (exact or list(results))[:8])


def _is_specials(name: str) -> bool:
    return "special" in name.lower()


def select_group_season(groups: Sequence[Json], season: int) -> Json | None:
    """Find the sub-group of an episode group that represents ``season``.

    Episode groups name their sub-groups freely, so try in turn: a name like
    "Season 2" / "Series 2" / "Volume 2"; "Specials" for season 0; otherwise the
    Nth non-special sub-group by ``order``.
    """
    pattern = re.compile(rf"\b(season|series|volume|vol\.?|disc set)\s*0*{season}\b", re.I)
    named = [g for g in groups if pattern.search(str(g.get("name", "")))]
    if len(named) == 1:
        return named[0]
    if season == 0:
        specials = [g for g in groups if _is_specials(str(g.get("name", "")))]
        return specials[0] if specials else None
    regular = sorted(
        (g for g in groups if not _is_specials(str(g.get("name", "")))),
        key=lambda g: int(g.get("order") or 0),
    )
    return regular[season - 1] if 0 < season <= len(regular) else None


def choose_dvd_group(show: ShowInfo) -> EpisodeGroupSummary:
    """Return the show's DVD episode group, raising if there is none or several."""
    dvd = [g for g in show.episode_groups if g.type == DVD_GROUP_TYPE]
    if len(dvd) == 1:
        return dvd[0]
    if not dvd:
        raise TMDBError(
            f"{show.name} has no DVD episode group on TMDB; use --order aired "
            "or pass an episode-group ID (see `disc2plex tmdb show`)"
        )
    listing = "\n".join(f"  {g.id}  {g.name} ({g.episode_count} episodes)" for g in dvd)
    raise TMDBError(
        f"{show.name} has several DVD episode groups; pass one with --order:\n{listing}"
    )


def expected_runtime_seconds(season: SeasonEpisodes) -> float | None:
    """Median per-episode runtime for the season, falling back to the show default."""
    runtimes = [e.runtime_minutes for e in season.episodes if e.runtime_minutes]
    if runtimes:
        return statistics.median(runtimes) * 60
    if season.show.default_runtime_minutes:
        return season.show.default_runtime_minutes * 60
    return None


# --- client -------------------------------------------------------------------


class TMDBClient:
    """Minimal TMDB v3 client with caching and friendly errors.

    ``token`` may be a v4 read-access token (sent as a Bearer header, preferred)
    or a 32-character v3 API key (sent as ``api_key``). Pass ``transport`` in
    tests to avoid real network access.
    """

    def __init__(
        self,
        token: str,
        *,
        cache: TMDBCache | None = None,
        language: str = "en-US",
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        base_url: str = API_BASE,
    ) -> None:
        if not token:
            raise TMDBError(
                "TMDB_API_TOKEN is not set. Add it to .env (see .env.example); get one at "
                "https://www.themoviedb.org/settings/api"
            )
        self._auth_params: dict[str, str] = {}
        headers = {"Accept": "application/json"}
        if _API_KEY.match(token):
            self._auth_params["api_key"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(
            base_url=base_url, headers=headers, timeout=20.0, transport=transport
        )
        self._cache = cache
        self._language = language
        self._sleep = sleep

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> TMDBClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def get(self, path: str, params: dict[str, str] | None = None) -> Json:
        """GET ``path`` (relative to the API base), using the cache when possible."""
        query = {"language": self._language, **(params or {})}
        if self._cache and (cached := self._cache.get(path, query)) is not None:
            log.debug("TMDB cache hit %s %s", path, query)
            return cached
        data = self._request(path, query)
        if self._cache:
            self._cache.put(path, query, data)
        return data

    def _request(self, path: str, query: dict[str, str]) -> Json:
        log.debug("TMDB GET %s %s", path, query)  # auth params are added below, never logged
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = self._http.get(path, params={**query, **self._auth_params})
            except httpx.HTTPError as exc:
                raise TMDBError(f"Could not reach TMDB: {exc}") from exc
            if response.status_code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                wait = min(float(response.headers.get("Retry-After", 1)), MAX_RETRY_WAIT_SECONDS)
                log.info("TMDB rate limit hit; retrying in %.1fs", wait)
                self._sleep(wait)
                continue
            break
        if response.status_code == 401:
            raise TMDBError("TMDB rejected the API token (401). Check TMDB_API_TOKEN.")
        if response.status_code == 404:
            raise TMDBNotFoundError(f"TMDB has no resource at {path}")
        if response.status_code >= 400:
            raise TMDBError(f"TMDB request {path} failed: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise TMDBError(f"TMDB returned invalid JSON for {path}") from exc
        if not isinstance(data, dict):
            raise TMDBError(f"Unexpected TMDB response for {path}")
        return data

    # Endpoints -------------------------------------------------------------

    def search_tv(self, query: str, year: int | None = None) -> list[ShowSummary]:
        params = {"query": query, "include_adult": "false", "page": "1"}
        if year is not None:
            params["first_air_date_year"] = str(year)
        return parse_search_results(self.get("/search/tv", params))

    def get_show(self, tmdb_id: int) -> ShowInfo:
        return parse_show(self.get(f"/tv/{tmdb_id}"), self.get(f"/tv/{tmdb_id}/episode_groups"))

    def get_season(self, tmdb_id: int, season: int) -> list[EpisodeInfo]:
        return parse_season(self.get(f"/tv/{tmdb_id}/season/{season}"))

    def get_episode_group(self, group_id: str) -> Json:
        return self.get(f"/tv/episode_group/{group_id}")


# --- orchestration ------------------------------------------------------------


def resolve_show(
    client: TMDBClient,
    *,
    name: str | None = None,
    year: int | None = None,
    tmdb_id: int | None = None,
) -> ShowInfo:
    """Find the show by ID, or by name (+ year) via search. Raises if ambiguous."""
    if tmdb_id is not None:
        return client.get_show(tmdb_id)
    if not name:
        raise TMDBError("Either a show name or a TMDB ID is required")
    chosen = choose_show(name, client.search_tv(name, year), year)
    log.info("resolved %r to %s (%s) tmdb-%d", name, chosen.name, chosen.year, chosen.tmdb_id)
    return client.get_show(chosen.tmdb_id)


def season_episodes(
    client: TMDBClient, show: ShowInfo, season: int, order: str = "aired"
) -> SeasonEpisodes:
    """Episodes for ``season`` under ``order`` ("aired", "dvd", or an episode-group ID).

    For group orders, episodes come back in the group's order but keep their
    aired season/episode numbers (see :class:`EpisodeInfo`).
    """
    if order == "aired":
        episodes = client.get_season(show.tmdb_id, season)
        return SeasonEpisodes(
            show=show,
            season=season,
            order=order,
            label=f"Season {season} (aired order)",
            episodes=tuple(episodes),
        )

    if order == "dvd":
        group = choose_dvd_group(show)
    elif _GROUP_ID.match(order):
        group = next(
            (g for g in show.episode_groups if g.id == order),
            EpisodeGroupSummary(id=order, name=order, type=0),
        )
    else:
        raise TMDBError(
            f"--order must be 'aired', 'dvd', or a 24-character group ID, not {order!r}"
        )

    data = client.get_episode_group(group.id)
    sub = select_group_season(data.get("groups", []), season)
    if sub is None:
        names = ", ".join(str(g.get("name")) for g in data.get("groups", []))
        raise TMDBError(f"Episode group {group.name!r} has no season {season} (groups: {names})")
    ordered = sorted(sub.get("episodes", []), key=lambda e: int(e.get("order") or 0))
    return SeasonEpisodes(
        show=show,
        season=season,
        order=order,
        label=f"Season {season} ({group.type_name or 'group'}: {group.name} / {sub.get('name')})",
        episodes=tuple(parse_episode(e) for e in ordered),
    )
