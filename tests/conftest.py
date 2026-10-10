"""Shared test helpers: fixture loading and fake rip folders (no real video)."""

from __future__ import annotations

import json
import os
import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from disc2plex.models import AudioTrack, EpisodeInfo, MediaFile, SeasonEpisodes, ShowInfo
from disc2plex.scan import ProbeData, ProbeError, ProbeFn, parse_title_index
from disc2plex.tmdb import TMDBCache, TMDBClient

FIXTURES = Path(__file__).parent / "fixtures"

_USER_SETTINGS_VARS = (
    "DISC2PLEX_CONFIG",
    "DISC2PLEX_LLM_MODEL",
    "DISC2PLEX_LLM_BASE_URL",
    "DISC2PLEX_LLM_API_KEY",
    "PLEX_URL",
    "PLEX_TOKEN",
)


@pytest.fixture(autouse=True)
def _isolate_user_dirs(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep every test away from the user's real config, caches, and manifests."""
    # Whisper models stay in the real Hugging Face cache (the opt-in real test needs them).
    monkeypatch.setenv(
        "HF_HOME", os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    )
    base = tmp_path_factory.mktemp("xdg")
    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(base / var.lower()))
    for var in _USER_SETTINGS_VARS:
        monkeypatch.delenv(var, raising=False)


def load_probe_fixture(name: str) -> ProbeData:
    """Load ``tests/fixtures/ffprobe/<name>.json``."""
    data: ProbeData = json.loads((FIXTURES / "ffprobe" / f"{name}.json").read_text())
    return data


def synthetic_probe(
    duration: float,
    *,
    chapters: list[float] | None = None,
    audio_languages: tuple[str, ...] = ("eng",),
) -> ProbeData:
    """Build minimal ffprobe-shaped JSON for a file of the given duration."""
    streams: list[dict[str, Any]] = [
        {"index": 0, "codec_type": "video", "codec_name": "mpeg2video"}
    ]
    for i, lang in enumerate(audio_languages, start=1):
        streams.append(
            {
                "index": i,
                "codec_type": "audio",
                "codec_name": "ac3",
                "channels": 2,
                "tags": {"language": lang},
            }
        )
    return {
        "streams": streams,
        "chapters": [{"start_time": str(start)} for start in (chapters or [])],
        "format": {"duration": str(duration)},
    }


def _bitrate_factor(name: str) -> float:
    """Deterministic per-name bitrate wobble (0.85-1.15) so sizes look like real discs."""
    return 0.85 + (zlib.crc32(name.encode()) % 300) / 1000


def make_media(
    name: str,
    minutes: float,
    *,
    size: int | None = None,
    chapters: Sequence[float] = (),
    audio: bool = True,
) -> MediaFile:
    """Build a MediaFile directly (no probing). ``chapters`` are start times in minutes.

    Default size is ~8 Mbit/s with a per-name bitrate wobble (+-15%), like real
    discs, so equal-length files aren't byte-identical "duplicates" and a long
    title isn't accidentally the exact size of two others.
    """
    seconds = minutes * 60
    default_size = int(seconds * 1_000_000 * _bitrate_factor(name))
    return MediaFile(
        path=Path("/rip") / name,
        size_bytes=default_size if size is None else size,
        duration_seconds=seconds,
        audio_tracks=(AudioTrack(index=1, codec="ac3", language="eng", channels=2),)
        if audio
        else (),
        chapter_starts=tuple(c * 60 for c in chapters),
        makemkv_title_index=parse_title_index(Path(name)),
    )


def make_show(name: str = "Test Show", year: int | None = 2001, tmdb_id: int = 4242) -> ShowInfo:
    return ShowInfo(tmdb_id=tmdb_id, name=name, year=year)


def make_season(
    runtimes: Sequence[float | None],
    *,
    season: int = 1,
    show: ShowInfo | None = None,
    numbers: Sequence[int] | None = None,
) -> SeasonEpisodes:
    """Season whose episode at position k has ``runtimes[k]`` minutes.

    ``numbers`` overrides the aired episode numbers (to mimic DVD order).
    """
    nums = list(numbers) if numbers is not None else list(range(1, len(runtimes) + 1))
    return SeasonEpisodes(
        show=show or make_show(),
        season=season,
        order="aired" if numbers is None else "dvd",
        label=f"Season {season}",
        episodes=tuple(
            EpisodeInfo(season=season, episode=n, title=f"Episode {n}", runtime_minutes=r)
            for n, r in zip(nums, runtimes, strict=True)
        ),
    )


# A spec is a fixture name, a ready-made probe dict, or an exception to raise,
# optionally paired with an explicit file size in bytes.
ProbeSource = str | ProbeData | ProbeError
ProbeSpec = ProbeSource | tuple[ProbeSource, int]
FakeRipBuilder = Callable[[Mapping[str, ProbeSpec]], tuple[Path, ProbeFn]]


def _probe_data(source: ProbeSource) -> ProbeData | None:
    if isinstance(source, ProbeError):
        return None
    return load_probe_fixture(source) if isinstance(source, str) else source


@pytest.fixture
def fake_rip(tmp_path: Path) -> FakeRipBuilder:
    """Return a builder that creates sparse .mkv files and a matching fake probe.

    Files are sized like real video (duration x ~8 Mbit/s, wobbled per name)
    using sparse files, so they take no disk space. Usage::

        rip_dir, probe = fake_rip(
            {"title_t00.mkv": "episode", "title_t01.mkv": (synthetic_probe(60), 12345)}
        )
        result = scan_directory(rip_dir, probe=probe)
    """

    def build(specs: Mapping[str, ProbeSpec]) -> tuple[Path, ProbeFn]:
        rip_dir = tmp_path / "rip"
        rip_dir.mkdir(exist_ok=True)
        sources: dict[str, ProbeSource] = {}
        for name, spec in specs.items():
            source, size = spec if isinstance(spec, tuple) else (spec, None)
            sources[name] = source
            if size is None:
                data = _probe_data(source)
                duration = float(data.get("format", {}).get("duration", 0) or 0) if data else 0
                size = int(duration * 1_000_000 * _bitrate_factor(name)) or 1
            path = rip_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                handle.truncate(size)

        def probe(path: Path) -> ProbeData:
            # Files moved elsewhere (e.g. into a review folder) are found by name.
            inside = path.is_relative_to(rip_dir)
            key = path.relative_to(rip_dir).as_posix() if inside else path.name
            source = sources[key]
            if isinstance(source, ProbeError):
                raise source
            data = _probe_data(source)
            assert data is not None
            return data

        return rip_dir, probe

    return build


# --- fake TMDB ------------------------------------------------------------------

DVD_GROUP = "5b0c3a5c0e0a267b7a00a1f1"
PROD_GROUP = "5c1d2e3f4a5b6c7d8e9f0a1b"
FAKE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.fake-test-token"


def tmdb_fixture(name: str) -> dict[str, object]:
    data: dict[str, object] = json.loads((FIXTURES / "tmdb" / f"{name}.json").read_text())
    return data


ROUTES = {
    "/3/search/tv": "search_firefly",
    "/3/tv/1437": "tv_1437",
    "/3/tv/1437/episode_groups": "tv_1437_episode_groups",
    "/3/tv/1437/season/1": "tv_1437_season_1",
    f"/3/tv/episode_group/{DVD_GROUP}": "episode_group_dvd",
}


class FakeTMDB:
    """Records requests and serves fixtures by URL path."""

    def __init__(self, routes: dict[str, str] | None = None) -> None:
        self.routes = dict(ROUTES if routes is None else routes)
        self.requests: list[httpx.Request] = []
        self.responses: dict[str, list[httpx.Response]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if queued := self.responses.get(path):
            return queued.pop(0)
        if path == "/3/search/tv" and request.url.params.get("query") == "Doctor Who":
            return httpx.Response(200, json=tmdb_fixture("search_doctor_who"))
        if path in self.routes:
            return httpx.Response(200, json=tmdb_fixture(self.routes[path]))
        return httpx.Response(404, json={"status_message": "not found"})

    def client(self, cache: TMDBCache | None = None, token: str = FAKE_TOKEN) -> TMDBClient:
        return TMDBClient(
            token, cache=cache, transport=httpx.MockTransport(self.handler), sleep=lambda _s: None
        )

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


@pytest.fixture
def fake() -> FakeTMDB:
    return FakeTMDB()
