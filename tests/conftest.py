"""Shared test helpers: fixture loading and fake rip folders (no real video)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from disc2plex.models import AudioTrack, MediaFile
from disc2plex.scan import ProbeData, ProbeError, ProbeFn, parse_title_index

FIXTURES = Path(__file__).parent / "fixtures"


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


def make_media(
    name: str,
    minutes: float,
    *,
    size: int | None = None,
    chapters: Sequence[float] = (),
    audio: bool = True,
) -> MediaFile:
    """Build a MediaFile directly (no probing). ``chapters`` are start times in minutes.

    Default size is ~8 Mbit/s DVD bitrate plus a per-name offset so two files of
    equal length are not accidentally byte-identical "duplicates".
    """
    seconds = minutes * 60
    default_size = int(seconds * 1_000_000) + sum(name.encode()) * 1_000
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


# A spec is a fixture name, a ready-made probe dict, or an exception to raise.
ProbeSpec = str | ProbeData | ProbeError
FakeRipBuilder = Callable[[Mapping[str, ProbeSpec]], tuple[Path, ProbeFn]]


@pytest.fixture
def fake_rip(tmp_path: Path) -> FakeRipBuilder:
    """Return a builder that creates zero-byte .mkv files and a matching fake probe.

    Usage::

        rip_dir, probe = fake_rip(
            {"title_t00.mkv": "episode", "title_t01.mkv": synthetic_probe(60)}
        )
        result = scan_directory(rip_dir, probe=probe)
    """

    def build(specs: Mapping[str, ProbeSpec]) -> tuple[Path, ProbeFn]:
        rip_dir = tmp_path / "rip"
        rip_dir.mkdir(exist_ok=True)
        for name in specs:
            (rip_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (rip_dir / name).touch()

        def probe(path: Path) -> ProbeData:
            spec = specs[path.relative_to(rip_dir).as_posix()]
            if isinstance(spec, ProbeError):
                raise spec
            return load_probe_fixture(spec) if isinstance(spec, str) else spec

        return rip_dir, probe

    return build
