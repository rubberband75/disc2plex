"""Stage 1: find .mkv files in a rip folder and probe them with ffprobe.

Side effects (subprocess, cache file I/O) live here. Parsing ffprobe output is
kept in pure functions so it can be tested against saved JSON fixtures.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from disc2plex.config import cache_dir
from disc2plex.models import AudioTrack, MediaFile, ScanFailure, ScanResult, SubtitleTrack

log = logging.getLogger(__name__)

ProbeData = dict[str, Any]
ProbeFn = Callable[[Path], ProbeData]

# MakeMKV names titles "title_t00.mkv" or "<Disc Name>_t03.mkv".
_TITLE_INDEX = re.compile(r"(?:^|_)t(\d+)$", re.IGNORECASE)
# Matroska per-stream duration tag, e.g. "00:42:13.123000000".
_HMS = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$")

FFPROBE_TIMEOUT_SECONDS = 60
# Bump when the cache format changes so stale entries are ignored.
_CACHE_VERSION = 1


class ProbeError(Exception):
    """ffprobe failed or returned unusable output for a file."""


# --- pure parsing -------------------------------------------------------------


def parse_title_index(path: Path) -> int | None:
    """Extract the MakeMKV title number from a filename, if it follows the convention."""
    match = _TITLE_INDEX.search(path.stem)
    return int(match.group(1)) if match else None


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _parse_hms(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    match = _HMS.match(value.strip())
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _duration(data: ProbeData) -> float:
    """Best available duration: container first, then the longest stream."""
    container = _to_float(data.get("format", {}).get("duration"))
    if container:
        return container
    stream_durations = []
    for stream in data.get("streams", []):
        value = _to_float(stream.get("duration"))
        if value is None:
            tags = stream.get("tags", {})
            value = _parse_hms(tags.get("DURATION") or tags.get("duration"))
        if value is not None:
            stream_durations.append(value)
    return max(stream_durations, default=0.0)


def _tags(stream: ProbeData) -> dict[str, Any]:
    tags = stream.get("tags", {})
    return tags if isinstance(tags, dict) else {}


def _language(stream: ProbeData) -> str:
    return str(_tags(stream).get("language") or "und")


def _disposition(stream: ProbeData, key: str) -> bool:
    return bool(stream.get("disposition", {}).get(key, 0))


def parse_probe(path: Path, size_bytes: int, data: ProbeData) -> MediaFile:
    """Convert ffprobe JSON (``-show_format -show_streams -show_chapters``) to a MediaFile."""
    audio: list[AudioTrack] = []
    subtitles: list[SubtitleTrack] = []
    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "audio":
            audio.append(
                AudioTrack(
                    index=int(stream.get("index", len(audio))),
                    codec=str(stream.get("codec_name", "unknown")),
                    language=_language(stream),
                    channels=int(stream.get("channels") or 0),
                    title=_tags(stream).get("title"),
                    default=_disposition(stream, "default"),
                )
            )
        elif kind == "subtitle":
            subtitles.append(
                SubtitleTrack(
                    index=int(stream.get("index", len(subtitles))),
                    codec=str(stream.get("codec_name", "unknown")),
                    language=_language(stream),
                    title=_tags(stream).get("title"),
                    forced=_disposition(stream, "forced"),
                )
            )

    chapter_starts = sorted(
        start
        for chapter in data.get("chapters", [])
        if (start := _to_float(chapter.get("start_time"))) is not None
    )

    return MediaFile(
        path=path,
        size_bytes=size_bytes,
        duration_seconds=_duration(data),
        audio_tracks=tuple(audio),
        subtitle_tracks=tuple(subtitles),
        chapter_starts=tuple(chapter_starts),
        makemkv_title_index=parse_title_index(path),
    )


def title_sort_key(path: Path) -> tuple[int, int, str]:
    """Order by MakeMKV title index when present, then by name."""
    index = parse_title_index(path)
    return (0 if index is not None else 1, index or 0, path.name.lower())


# --- side effects -------------------------------------------------------------


def find_mkv_files(rip_dir: Path, *, recursive: bool = False) -> list[Path]:
    """List ``.mkv`` files in ``rip_dir`` (case-insensitive), sorted by title index."""
    pattern = "**/*" if recursive else "*"
    files = [p for p in rip_dir.glob(pattern) if p.is_file() and p.suffix.lower() == ".mkv"]
    return sorted(files, key=title_sort_key)


def _summarize_stderr(stderr: str, path: Path) -> str:
    """Reduce ffprobe's stderr to its last line, minus the redundant path prefix."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1].removeprefix(f"{path}: ")


def run_ffprobe(path: Path) -> ProbeData:
    """Run ffprobe on ``path`` and return its parsed JSON output."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        str(path),
    ]
    log.debug("running %s", cmd)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_SECONDS, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"ffprobe timed out after {FFPROBE_TIMEOUT_SECONDS}s") from exc
    if proc.returncode != 0:
        log.debug("ffprobe stderr for %s:\n%s", path, proc.stderr)
        raise ProbeError(_summarize_stderr(proc.stderr, path) or f"exit code {proc.returncode}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict) or "format" not in data:
        raise ProbeError("ffprobe output has no format section")
    return data


def default_cache_path() -> Path:
    """Probe cache location: ``$XDG_CACHE_HOME/disc2plex/probe.json``."""
    return cache_dir() / "probe.json"


class ProbeCache:
    """Raw ffprobe output keyed by (absolute path, size, mtime).

    Raw output is stored (rather than parsed MediaFiles) so parser changes take
    effect without invalidating the cache.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("ignoring unreadable probe cache %s: %s", self.path, exc)
            return
        if isinstance(raw, dict) and raw.get("version") == _CACHE_VERSION:
            self._entries = raw.get("entries", {})

    @staticmethod
    def _key(path: Path) -> str:
        return str(path.resolve())

    def get(self, path: Path, size: int, mtime_ns: int) -> ProbeData | None:
        entry = self._entries.get(self._key(path))
        if entry and entry.get("size") == size and entry.get("mtime_ns") == mtime_ns:
            probe: ProbeData = entry["probe"]
            return probe
        return None

    def put(self, path: Path, size: int, mtime_ns: int, probe: ProbeData) -> None:
        self._entries[self._key(path)] = {"size": size, "mtime_ns": mtime_ns, "probe": probe}
        self._dirty = True

    def save(self) -> None:
        """Write the cache atomically (temp file + rename) if anything changed."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        payload = {"version": _CACHE_VERSION, "entries": self._entries}
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False


def scan_directory(
    rip_dir: Path,
    *,
    recursive: bool = False,
    cache: ProbeCache | None = None,
    probe: ProbeFn = run_ffprobe,
) -> ScanResult:
    """Probe every .mkv in ``rip_dir``. Files that fail to probe are reported, not raised."""
    files: list[MediaFile] = []
    failures: list[ScanFailure] = []
    for path in find_mkv_files(rip_dir, recursive=recursive):
        stat = path.stat()
        data = cache.get(path, stat.st_size, stat.st_mtime_ns) if cache else None
        if data is None:
            try:
                data = probe(path)
            except ProbeError as exc:
                # Reported to the user via ScanResult.failures; log for -v runs.
                log.info("could not probe %s: %s", path, exc)
                failures.append(ScanFailure(path=path, error=str(exc)))
                continue
            if cache:
                cache.put(path, stat.st_size, stat.st_mtime_ns, data)
        else:
            log.debug("probe cache hit for %s", path)
        files.append(parse_probe(path, stat.st_size, data))
    if cache:
        cache.save()
    return ScanResult(files=tuple(files), failures=tuple(failures))
