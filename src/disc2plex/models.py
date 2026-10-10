"""Core data models shared across pipeline stages.

Models are added in the milestone that first needs them (see CLAUDE.md section 6).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

# DVD/Blu-ray subtitles are images; they need OCR before they are usable as text.
BITMAP_SUBTITLE_CODECS = frozenset({"dvd_subtitle", "hdmv_pgs_subtitle", "dvb_subtitle"})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class FileClass(StrEnum):
    """What a ripped title appears to be."""

    EPISODE = "episode"
    PLAY_ALL = "play_all"
    EXTRA = "extra"
    JUNK = "junk"
    UNKNOWN = "unknown"


class AudioTrack(_Frozen):
    index: int
    codec: str
    language: str = "und"
    channels: int = 0
    title: str | None = None
    default: bool = False


class SubtitleTrack(_Frozen):
    index: int
    codec: str
    language: str = "und"
    title: str | None = None
    forced: bool = False

    @property
    def is_bitmap(self) -> bool:
        """True for image-based subtitles (typical on DVDs)."""
        return self.codec in BITMAP_SUBTITLE_CODECS


class MediaFile(_Frozen):
    """One probed video file from a rip folder."""

    path: Path
    size_bytes: int
    duration_seconds: float
    audio_tracks: tuple[AudioTrack, ...] = ()
    subtitle_tracks: tuple[SubtitleTrack, ...] = ()
    # Chapter start times in seconds; used to spot Play-All tracks whose chapters
    # line up with episode boundaries.
    chapter_starts: tuple[float, ...] = ()
    makemkv_title_index: int | None = None

    @property
    def chapter_count(self) -> int:
        return len(self.chapter_starts)


class ScanFailure(_Frozen):
    """A file that could not be probed (corrupt, truncated, not really video...)."""

    path: Path
    error: str


class ScanResult(_Frozen):
    files: tuple[MediaFile, ...] = ()
    failures: tuple[ScanFailure, ...] = ()
