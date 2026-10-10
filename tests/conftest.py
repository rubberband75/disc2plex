"""Shared test helpers: fixture loading and fake rip folders (no real video)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from disc2plex.scan import ProbeData, ProbeError, ProbeFn

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
