"""Tests for ffprobe parsing, file discovery, caching, and the scan command."""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import FakeRipBuilder, load_probe_fixture, synthetic_probe
from disc2plex import cli
from disc2plex.scan import (
    ProbeCache,
    ProbeData,
    ProbeError,
    find_mkv_files,
    parse_probe,
    parse_title_index,
    run_ffprobe,
    scan_directory,
)

# --- title index --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("title_t00.mkv", 0),
        ("title_t03.mkv", 3),
        ("FIREFLY_D1_t12.mkv", 12),
        ("B1_T07.MKV", 7),
        ("t04.mkv", 4),
        ("episode.mkv", None),
        ("Batman_t00_extra.mkv", None),
        ("matt04.mkv", None),
    ],
)
def test_parse_title_index(name: str, expected: int | None) -> None:
    assert parse_title_index(Path(name)) == expected


# --- parse_probe --------------------------------------------------------------


def test_parse_episode_fixture() -> None:
    media = parse_probe(Path("title_t00.mkv"), 1503238553, load_probe_fixture("episode"))
    assert media.duration_seconds == pytest.approx(2580.123)
    assert media.size_bytes == 1503238553
    assert media.makemkv_title_index == 0
    assert media.chapter_count == 5
    assert media.chapter_starts[1] == pytest.approx(312.512)

    assert [(a.language, a.codec, a.channels) for a in media.audio_tracks] == [
        ("eng", "ac3", 6),
        ("fre", "ac3", 2),
    ]
    assert media.audio_tracks[0].default
    assert media.audio_tracks[0].title == "Surround 5.1"
    assert not media.audio_tracks[1].default

    assert [s.language for s in media.subtitle_tracks] == ["eng", "spa"]
    assert all(s.is_bitmap for s in media.subtitle_tracks)
    assert media.subtitle_tracks[1].forced


def test_parse_file_without_audio() -> None:
    media = parse_probe(Path("title_t09.mkv"), 10, load_probe_fixture("menu_no_audio"))
    assert media.audio_tracks == ()
    assert media.subtitle_tracks == ()
    assert media.chapter_count == 0
    assert media.duration_seconds == pytest.approx(30.03)


def test_parse_falls_back_to_stream_duration_tags() -> None:
    media = parse_probe(Path("x.mkv"), 10, load_probe_fixture("tag_duration_only"))
    assert media.duration_seconds == pytest.approx(2650.5)
    assert media.audio_tracks[0].language == "und"


def test_parse_tolerates_minimal_and_junk_values() -> None:
    data: ProbeData = {
        "streams": [{"codec_type": "audio"}, {"codec_type": "data"}],
        "chapters": [{"start_time": "bogus"}, {"start_time": "5.0"}, {}],
        "format": {"duration": "N/A"},
    }
    media = parse_probe(Path("x.mkv"), 0, data)
    assert media.duration_seconds == 0.0
    assert media.audio_tracks[0].codec == "unknown"
    assert media.chapter_starts == (5.0,)


def test_parse_sorts_chapters() -> None:
    media = parse_probe(Path("x.mkv"), 0, synthetic_probe(100, chapters=[50.0, 0.0, 25.0]))
    assert media.chapter_starts == (0.0, 25.0, 50.0)


# --- discovery ----------------------------------------------------------------


def test_find_mkv_files_sorts_by_title_index(tmp_path: Path) -> None:
    for name in ["title_t10.mkv", "title_t02.mkv", "notes.txt", "bonus.mkv", "title_t00.MKV"]:
        (tmp_path / name).touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "title_t01.mkv").touch()

    names = [p.name for p in find_mkv_files(tmp_path)]
    assert names == ["title_t00.MKV", "title_t02.mkv", "title_t10.mkv", "bonus.mkv"]

    recursive = [p.name for p in find_mkv_files(tmp_path, recursive=True)]
    assert "title_t01.mkv" in recursive


def test_find_mkv_files_ignores_directories_named_mkv(tmp_path: Path) -> None:
    (tmp_path / "weird.mkv").mkdir()
    assert find_mkv_files(tmp_path) == []


# --- scan_directory -----------------------------------------------------------


def test_scan_directory_with_fake_rip(fake_rip: FakeRipBuilder) -> None:
    rip_dir, probe = fake_rip(
        {
            "title_t01.mkv": "extra_short",
            "title_t00.mkv": "episode",
            "title_t02.mkv": "menu_no_audio",
        }
    )
    result = scan_directory(rip_dir, probe=probe)
    assert [f.makemkv_title_index for f in result.files] == [0, 1, 2]
    assert result.failures == ()


def test_scan_directory_reports_probe_failures(fake_rip: FakeRipBuilder) -> None:
    rip_dir, probe = fake_rip(
        {"title_t00.mkv": "episode", "title_t01.mkv": ProbeError("Invalid data found")}
    )
    result = scan_directory(rip_dir, probe=probe)
    assert len(result.files) == 1
    assert result.failures[0].path.name == "title_t01.mkv"
    assert "Invalid data" in result.failures[0].error


def test_scan_directory_empty(tmp_path: Path) -> None:
    result = scan_directory(tmp_path, probe=lambda _p: pytest.fail("should not probe"))
    assert result.files == ()


# --- cache --------------------------------------------------------------------


def test_cache_avoids_reprobing(fake_rip: FakeRipBuilder, tmp_path: Path) -> None:
    rip_dir, probe = fake_rip({"title_t00.mkv": "episode"})
    calls: list[Path] = []

    def counting_probe(path: Path) -> ProbeData:
        calls.append(path)
        return probe(path)

    cache_path = tmp_path / "cache" / "probe.json"
    scan_directory(rip_dir, cache=ProbeCache(cache_path), probe=counting_probe)
    second = scan_directory(rip_dir, cache=ProbeCache(cache_path), probe=counting_probe)
    assert len(calls) == 1
    assert second.files[0].duration_seconds == pytest.approx(2580.123)


def test_cache_invalidated_when_file_changes(fake_rip: FakeRipBuilder, tmp_path: Path) -> None:
    rip_dir, probe = fake_rip({"title_t00.mkv": "episode"})
    calls: list[Path] = []

    def counting_probe(path: Path) -> ProbeData:
        calls.append(path)
        return probe(path)

    cache_path = tmp_path / "probe.json"
    scan_directory(rip_dir, cache=ProbeCache(cache_path), probe=counting_probe)
    (rip_dir / "title_t00.mkv").write_bytes(b"changed")
    scan_directory(rip_dir, cache=ProbeCache(cache_path), probe=counting_probe)
    assert len(calls) == 2


def test_cache_does_not_store_failures(fake_rip: FakeRipBuilder, tmp_path: Path) -> None:
    rip_dir, probe = fake_rip({"title_t00.mkv": ProbeError("boom")})
    cache = ProbeCache(tmp_path / "probe.json")
    scan_directory(rip_dir, cache=cache, probe=probe)
    assert not (tmp_path / "probe.json").exists()


@pytest.mark.parametrize("content", ["not json", json.dumps({"version": 999, "entries": {}})])
def test_cache_ignores_corrupt_or_old_file(tmp_path: Path, content: str) -> None:
    cache_path = tmp_path / "probe.json"
    cache_path.write_text(content)
    cache = ProbeCache(cache_path)
    assert cache.get(tmp_path / "x.mkv", 0, 0) is None


# --- run_ffprobe (subprocess, mocked) ----------------------------------------


def _completed(
    stdout: str = "", stderr: str = "", code: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=stdout, stderr=stderr)


def test_run_ffprobe_error_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stderr="moov atom", code=1))
    with pytest.raises(ProbeError, match="moov atom"):
        run_ffprobe(Path("x.mkv"))


def test_run_ffprobe_error_uses_last_stderr_line(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = "Truncating packet\nEBML header parsing failed\nx.mkv: Invalid data found\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stderr=stderr, code=1))
    with pytest.raises(ProbeError) as excinfo:
        run_ffprobe(Path("x.mkv"))
    assert str(excinfo.value) == "Invalid data found"


def test_run_ffprobe_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout="{nope"))
    with pytest.raises(ProbeError, match="invalid JSON"):
        run_ffprobe(Path("x.mkv"))


def test_run_ffprobe_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=1)

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(ProbeError, match="timed out"):
        run_ffprobe(Path("x.mkv"))


# --- real ffprobe on a tiny generated file ------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_run_ffprobe_on_generated_mkv(tmp_path: Path) -> None:
    """End-to-end check against real ffprobe using a 3-second synthetic file."""
    meta = tmp_path / "chapters.txt"
    meta.write_text(
        ";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1500\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=1500\nEND=3000\n"
    )
    out = tmp_path / "title_t05.mkv"
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=64x48:rate=5",
            "-f", "lavfi", "-i", "sine=duration=3",
            "-i", str(meta), "-map", "0", "-map", "1", "-map_metadata", "2",
            "-c:v", "mpeg2video", "-c:a", "ac3",
            "-metadata:s:a:0", "language=eng",
            str(out),
        ],
        check=True,
    )  # fmt: skip
    media = parse_probe(out, out.stat().st_size, run_ffprobe(out))
    assert media.duration_seconds == pytest.approx(3.0, abs=0.3)
    assert media.makemkv_title_index == 5
    assert media.chapter_count == 2
    assert media.audio_tracks[0].language == "eng"


def test_run_ffprobe_rejects_non_video(tmp_path: Path) -> None:
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not installed")
    bogus = tmp_path / "title_t00.mkv"
    bogus.write_bytes(b"definitely not matroska")
    with pytest.raises(ProbeError):
        run_ffprobe(bogus)


# --- CLI ----------------------------------------------------------------------

runner = CliRunner(env={"COLUMNS": "200"})


@pytest.fixture
def cli_with_fake_probe(
    fake_rip: FakeRipBuilder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    rip_dir, probe = fake_rip(
        {"title_t00.mkv": "episode", "title_t01.mkv": ProbeError("corrupt file")}
    )
    monkeypatch.setattr(cli, "scan_directory", functools.partial(scan_directory, probe=probe))
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    return rip_dir


def test_scan_command_table(cli_with_fake_probe: Path) -> None:
    result = runner.invoke(cli.app, ["scan", str(cli_with_fake_probe)])
    assert result.exit_code == 0, result.output
    assert "title_t00.mkv" in result.output
    assert "43:00" in result.output
    assert "eng ac3 6ch" in result.output
    assert "Could not probe title_t01.mkv" in result.output


def test_scan_command_json(cli_with_fake_probe: Path) -> None:
    result = runner.invoke(cli.app, ["scan", str(cli_with_fake_probe), "--json", "--no-cache"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["items"][0]["file"]["makemkv_title_index"] == 0
    assert data["items"][0]["file"]["duration_seconds"] == pytest.approx(2580.123)
    assert data["items"][0]["file_class"] == "unknown"  # lone title, no expected length
    assert data["failures"][0]["error"] == "corrupt file"


def test_scan_command_uses_episode_length(cli_with_fake_probe: Path) -> None:
    result = runner.invoke(
        cli.app, ["scan", str(cli_with_fake_probe), "--json", "--episode-length", "44"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["reference_source"] == "expected"
    assert data["items"][0]["file_class"] == "episode"


def test_scan_command_writes_cache_under_xdg(cli_with_fake_probe: Path, tmp_path: Path) -> None:
    runner.invoke(cli.app, ["scan", str(cli_with_fake_probe)])
    assert (tmp_path / "xdg-cache" / "disc2plex" / "probe.json").exists()
