"""Core data models shared across pipeline stages.

Models are added in the milestone that first needs them (see CLAUDE.md section 6).
"""

from __future__ import annotations

from datetime import date
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


class Classification(_Frozen):
    """Stage 2 verdict for one file, with the evidence behind it."""

    file: MediaFile
    file_class: FileClass
    reasons: tuple[str, ...] = ()
    # Duration relative to the reference episode length, when one is known.
    length_ratio: float | None = None
    # Probable episode count for EPISODE files: 2 for a double-length title.
    episode_count_hint: int = 1
    # For PLAY_ALL (or a suspected one): the titles it appears to concatenate.
    play_all_of: tuple[Path, ...] = ()
    duplicate_of: Path | None = None


class ClassifyResult(_Frozen):
    items: tuple[Classification, ...] = ()
    # Typical single-episode length used for classification, if determinable.
    reference_seconds: float | None = None
    reference_source: str | None = None  # "cluster", "expected", or None
    notes: tuple[str, ...] = ()

    def by_class(self, file_class: FileClass) -> tuple[Classification, ...]:
        return tuple(item for item in self.items if item.file_class is file_class)


class EpisodeInfo(_Frozen):
    """One episode from TMDB. ``season``/``episode`` are always TMDB's aired numbering,
    which is what Plex's TMDB agent matches on, even when DVD order was used to
    decide which episodes are on a disc."""

    season: int
    episode: int
    title: str
    overview: str = ""
    runtime_minutes: float | None = None
    air_date: date | None = None
    tmdb_id: int | None = None


class ShowSummary(_Frozen):
    """A TMDB search hit."""

    tmdb_id: int
    name: str
    year: int | None = None
    original_name: str | None = None
    overview: str = ""
    popularity: float = 0.0


class SeasonSummary(_Frozen):
    number: int
    name: str
    episode_count: int


# TMDB episode-group type codes.
EPISODE_GROUP_TYPES = {
    1: "Original air date",
    2: "Absolute",
    3: "DVD",
    4: "Digital",
    5: "Story arc",
    6: "Production",
    7: "TV",
}


class EpisodeGroupSummary(_Frozen):
    id: str
    name: str
    type: int
    group_count: int = 0
    episode_count: int = 0
    description: str = ""

    @property
    def type_name(self) -> str:
        return EPISODE_GROUP_TYPES.get(self.type, f"type {self.type}")


class ShowInfo(_Frozen):
    """Show-level TMDB metadata. Episode lists are fetched per season on demand."""

    tmdb_id: int
    name: str
    year: int | None = None
    seasons: tuple[SeasonSummary, ...] = ()
    episode_groups: tuple[EpisodeGroupSummary, ...] = ()
    # Show-wide typical runtime (TMDB's episode_run_time); often empty nowadays.
    default_runtime_minutes: float | None = None


class SeasonEpisodes(_Frozen):
    """The episodes making up one season under a chosen ordering, in that order."""

    show: ShowInfo
    season: int
    order: str  # "aired", "dvd", or an episode-group ID
    label: str  # human-readable, e.g. "Season 1 (aired)" or "Season 1 (DVD: DVD Order)"
    episodes: tuple[EpisodeInfo, ...] = ()


class AssignmentStatus(StrEnum):
    CONFIDENT = "confident"
    NEEDS_REVIEW = "needs_review"
    UNMATCHED = "unmatched"


class Assignment(_Frozen):
    """The decision for one file: which episode(s) it is and how sure we are."""

    file: MediaFile
    file_class: FileClass
    episodes: tuple[EpisodeInfo, ...] = ()
    confidence: float = 0.0
    status: AssignmentStatus = AssignmentStatus.UNMATCHED
    # Named numeric signals that fed the confidence (e.g. "order", "runtime_fit").
    signals: dict[str, float] = {}
    reasons: tuple[str, ...] = ()

    @property
    def episode_label(self) -> str:
        """``"s01e01"`` / ``"s01e01-e02"`` style label, or ``""`` if unassigned."""
        if not self.episodes:
            return ""
        first, last = self.episodes[0], self.episodes[-1]
        label = f"s{first.season:02d}e{first.episode:02d}"
        if len(self.episodes) > 1:
            label += (
                f"-e{last.episode:02d}"
                if last.season == first.season
                else f"+s{last.season:02d}e{last.episode:02d}"
            )
        return label


class MatchResult(_Frozen):
    assignments: tuple[Assignment, ...] = ()
    # 1-based position (in the chosen order) of the first episode on the disc.
    start_position: int | None = None
    start_source: str | None = None  # "range", "disc1", "inferred"
    notes: tuple[str, ...] = ()


class PlanAction(StrEnum):
    MOVE = "move"
    HARDLINK = "hardlink"
    COPY = "copy"
    SKIP = "skip"


class PlanItem(_Frozen):
    source: Path
    destination: Path | None
    action: PlanAction
    reason: str
    file_class: FileClass
    status: AssignmentStatus
    confidence: float = 0.0
    episode_label: str = ""


class Plan(_Frozen):
    output_root: Path
    items: tuple[PlanItem, ...] = ()
    # Errors block apply (e.g. two files mapping to one destination).
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class EntryStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"
    UNDONE = "undone"


class ManifestEntry(BaseModel):
    """One planned file operation and what actually happened to it.

    Mutable on purpose: the manifest is rewritten after every operation so an
    interrupted run can still be undone.
    """

    source: Path
    destination: Path
    action: PlanAction
    status: EntryStatus = EntryStatus.PENDING
    message: str = ""
    # Recorded after placing the file, so undo can tell if it changed since.
    size_bytes: int | None = None
    inode: int | None = None


class Manifest(BaseModel):
    run_id: str
    created_at: str
    tool_version: str
    rip_dir: Path
    output_root: Path
    args: dict[str, str | int | float | bool | None] = {}
    entries: list[ManifestEntry] = []
    # Directories and marker files this run created (removed on undo if untouched).
    created_dirs: list[Path] = []
    created_files: list[Path] = []
    finished: bool = False
    undone_at: str | None = None


class ScanFailure(_Frozen):
    """A file that could not be probed (corrupt, truncated, not really video...)."""

    path: Path
    error: str


class ScanResult(_Frozen):
    files: tuple[MediaFile, ...] = ()
    failures: tuple[ScanFailure, ...] = ()
