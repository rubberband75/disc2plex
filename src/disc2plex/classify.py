"""Stage 2: sort probed files into episodes, Play-All tracks, extras, and junk.

Pure functions only. The guiding rule is to be conservative: when evidence is
weak or conflicting, the file is classed ``UNKNOWN`` with reasons attached so a
later stage (or the user) can decide.

Passes, in order:

1. Junk: no duration, very short, no audio, or an exact duplicate of another title.
2. Play-All: duration matches the sum of two or more other titles, confirmed by
   matching total size (same underlying video) or by chapter marks at the
   component boundaries. An unconfirmed match is only "suspected".
3. Reference length: the dominant cluster of similar durations, optionally
   steered by an expected runtime (from TMDB).
4. Ratio bands: each remaining file is judged by duration / reference.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from disc2plex.models import Classification, ClassifyResult, FileClass, MediaFile
from disc2plex.report import format_duration


@dataclass(frozen=True)
class ClassifyConfig:
    """Tunable thresholds. Ratios are relative to the reference episode length."""

    junk_max_seconds: float = 60.0
    duplicate_duration_seconds: float = 1.0
    duplicate_size_tolerance: float = 0.005
    duplicate_chapter_seconds: float = 0.5
    play_all_duration_tolerance: float = 0.025
    play_all_size_tolerance: float = 0.03
    chapter_match_seconds: float = 3.0
    # Fraction of component boundaries that must have a nearby chapter mark.
    chapter_match_fraction: float = 0.75
    cluster_tolerance: float = 0.15
    # Without an expected runtime, ignore short files when finding the episode
    # cluster so a pile of similar-length featurettes can't win.
    min_cluster_seconds: float = 600.0
    episode_tolerance: float = 0.20
    double_tolerance: float = 0.20
    extra_max_ratio: float = 0.5
    # Without any reference length, files shorter than this are called extras.
    no_reference_extra_seconds: float = 300.0


# 6! = 720 orderings is cheap; beyond that only title order is checked.
_MAX_PERMUTED_COMPONENTS = 6


@dataclass
class _Verdict:
    """Mutable working state for one file while passes run."""

    file: MediaFile
    file_class: FileClass | None = None
    reasons: list[str] | None = None
    play_all_of: tuple[Path, ...] = ()
    suspected_play_all_of: tuple[Path, ...] = ()
    duplicate_of: Path | None = None

    def decide(self, file_class: FileClass, reason: str) -> None:
        self.file_class = file_class
        self.note(reason)

    def note(self, reason: str) -> None:
        if self.reasons is None:
            self.reasons = []
        self.reasons.append(reason)


def _within(a: float, b: float, tolerance: float) -> bool:
    """True if ``a`` is within ``tolerance`` (a fraction) of ``b``."""
    return abs(a - b) <= tolerance * b


def _names(files: Sequence[MediaFile]) -> str:
    return ", ".join(f.path.name for f in files)


# --- pass 1: junk -------------------------------------------------------------


def _is_duplicate(a: MediaFile, b: MediaFile, cfg: ClassifyConfig) -> bool:
    """Same duration, same size, and (if present) the same chapter layout.

    Requiring matching chapters keeps two different episodes that happen to
    have near-identical length and size from being flagged.
    """
    if abs(a.duration_seconds - b.duration_seconds) > cfg.duplicate_duration_seconds:
        return False
    if not _within(a.size_bytes, b.size_bytes, cfg.duplicate_size_tolerance):
        return False
    if a.chapter_count != b.chapter_count:
        return False
    return all(
        abs(x - y) <= cfg.duplicate_chapter_seconds
        for x, y in zip(a.chapter_starts, b.chapter_starts, strict=True)
    )


def _mark_junk(verdicts: list[_Verdict], cfg: ClassifyConfig) -> None:
    kept: list[MediaFile] = []
    for v in verdicts:
        f = v.file
        if f.duration_seconds <= 0:
            v.decide(FileClass.JUNK, "no usable duration")
        elif f.duration_seconds < cfg.junk_max_seconds:
            v.decide(
                FileClass.JUNK,
                f"very short ({format_duration(f.duration_seconds)} "
                f"< {format_duration(cfg.junk_max_seconds)})",
            )
        elif not f.audio_tracks:
            v.decide(FileClass.JUNK, "no audio tracks (likely a menu or still loop)")
        elif original := next((k for k in kept if _is_duplicate(k, f, cfg)), None):
            v.duplicate_of = original.path
            v.decide(FileClass.JUNK, f"duplicate of {original.path.name} (same length and size)")
        else:
            kept.append(f)


# --- pass 2: Play-All ---------------------------------------------------------


def _find_component_subset(
    target: float, items: Sequence[MediaFile], tolerance: float
) -> tuple[MediaFile, ...] | None:
    """Find two or more ``items`` whose durations sum to within ``tolerance`` of ``target``.

    Subset-sum over whole seconds. Each reachable sum keeps the first subset
    found, and items are visited in title order, so results favour earlier titles.
    Of the matching subsets, the one closest to ``target`` wins.
    """
    high = target * (1 + tolerance)
    low = target * (1 - tolerance)
    sums: dict[int, tuple[int, ...]] = {0: ()}
    for index, item in enumerate(items):
        seconds = round(item.duration_seconds)
        for total, subset in list(sums.items()):
            new_total = total + seconds
            if new_total <= high and new_total not in sums:
                sums[new_total] = (*subset, index)
    matches = [
        (abs(total - target), subset)
        for total, subset in sums.items()
        if low <= total <= high and len(subset) >= 2
    ]
    if not matches:
        return None
    _, best = min(matches)
    return tuple(items[i] for i in best)


def _boundaries_matched(
    chapter_starts: Sequence[float], components: Sequence[MediaFile], cfg: ClassifyConfig
) -> bool:
    """True if chapter marks sit at most boundaries between ``components`` played in order.

    Tolerance grows slightly with offset because small per-episode differences
    accumulate over a long Play-All.
    """
    boundaries: list[float] = []
    offset = 0.0
    for component in components[:-1]:
        offset += component.duration_seconds
        boundaries.append(offset)
    matched = sum(
        1
        for boundary in boundaries
        if any(
            abs(start - boundary) <= cfg.chapter_match_seconds + 0.005 * boundary
            for start in chapter_starts
        )
    )
    return matched >= math.ceil(cfg.chapter_match_fraction * len(boundaries))


def _chapters_align(
    candidate: MediaFile, components: Sequence[MediaFile], cfg: ClassifyConfig
) -> bool:
    """True if the candidate's chapters line up with component boundaries in some order.

    Titles can be out of order relative to the Play-All, so small sets try every
    ordering; larger sets only try title order.
    """
    if len(components) < 2 or not candidate.chapter_starts:
        return False
    orders = (
        itertools.permutations(components)
        if len(components) <= _MAX_PERMUTED_COMPONENTS
        else iter([tuple(components)])
    )
    return any(_boundaries_matched(candidate.chapter_starts, order, cfg) for order in orders)


def _mark_play_all(verdicts: list[_Verdict], cfg: ClassifyConfig) -> None:
    live = [v for v in verdicts if v.file_class is None]
    confirmed: set[Path] = set()
    for v in sorted(live, key=lambda v: v.file.duration_seconds, reverse=True):
        candidate = v.file
        items = [
            other.file
            for other in live
            if other is not v
            and other.file.path not in confirmed
            and other.file.duration_seconds < candidate.duration_seconds
        ]
        components = _find_component_subset(
            candidate.duration_seconds, items, cfg.play_all_duration_tolerance
        )
        if components is None:
            continue
        total = sum(c.duration_seconds for c in components)
        summary = (
            f"length {format_duration(candidate.duration_seconds)} matches "
            f"{len(components)} titles totalling {format_duration(total)} ({_names(components)})"
        )
        paths = tuple(c.path for c in components)
        size_match = _within(
            candidate.size_bytes,
            sum(c.size_bytes for c in components),
            cfg.play_all_size_tolerance,
        )
        if size_match or _chapters_align(candidate, components, cfg):
            v.play_all_of = paths
            v.decide(FileClass.PLAY_ALL, summary)
            v.note(
                "total file size also matches"
                if size_match
                else "chapter marks line up with title boundaries"
            )
            confirmed.add(candidate.path)
        elif candidate.chapter_count >= 2:
            # Real Play-Alls have chapters at episode boundaries; chapters that
            # don't line up are evidence this is one long title instead.
            v.note(f"{summary}, but chapters don't line up, so treating it as one title")
        else:
            v.suspected_play_all_of = paths
            v.note(f"possible Play-All: {summary}, but size and chapters don't confirm it")


# --- pass 3: reference length -------------------------------------------------


def _find_reference(
    files: Sequence[MediaFile], expected_seconds: float | None, cfg: ClassifyConfig
) -> tuple[float | None, str | None, list[str]]:
    """Pick the typical single-episode length.

    Candidate clusters are files within ``cluster_tolerance`` of some centre file.
    With an expected runtime, the multi-file cluster nearest to it wins (TMDB
    runtimes are rounded, so the observed median is preferred when close).
    Without one, the largest cluster of files >= ``min_cluster_seconds`` wins,
    ties broken by total duration.
    """
    notes: list[str] = []
    pool = (
        list(files)
        if expected_seconds
        else [f for f in files if f.duration_seconds >= cfg.min_cluster_seconds]
    )
    clusters: list[list[float]] = []
    for centre in pool:
        group = sorted(
            f.duration_seconds
            for f in pool
            if _within(f.duration_seconds, centre.duration_seconds, cfg.cluster_tolerance)
        )
        if len(group) >= 2:
            clusters.append(group)

    if expected_seconds:
        near = [
            c
            for c in clusters
            if _within(statistics.median(c), expected_seconds, 2 * cfg.episode_tolerance)
        ]
        if near:
            best = min(near, key=lambda c: (abs(statistics.median(c) - expected_seconds), -len(c)))
            return statistics.median(best), "cluster", notes
        if clusters:
            notes.append(
                f"no cluster of titles is near the expected runtime "
                f"{format_duration(expected_seconds)}; check the show/season"
            )
        return expected_seconds, "expected", notes

    if not clusters:
        notes.append(
            "could not infer episode length (fewer than two similar-length titles); "
            "pass the expected runtime to classify more files"
        )
        return None, None, notes
    best = max(clusters, key=lambda c: (len(c), sum(c)))
    return statistics.median(best), "cluster", notes


# --- pass 4: ratio bands ------------------------------------------------------


def _classify_by_ratio(v: _Verdict, reference: float | None, cfg: ClassifyConfig) -> None:
    duration = v.file.duration_seconds
    if reference is None:
        if duration < cfg.no_reference_extra_seconds:
            v.decide(
                FileClass.EXTRA,
                f"short ({format_duration(duration)}) and no episode length to compare against",
            )
        else:
            v.decide(FileClass.UNKNOWN, "no episode length to compare against")
        return

    ratio = duration / reference
    ref_text = format_duration(reference)
    if v.suspected_play_all_of:
        # Includes the classic ambiguity: a 2x title equal to two episodes
        # could be a two-episode Play-All or a genuine double-length episode.
        v.decide(FileClass.UNKNOWN, f"{ratio:.2f}x episode length; Play-All or long episode?")
    elif _within(ratio, 1.0, cfg.episode_tolerance):
        v.decide(FileClass.EPISODE, f"{ratio:.2f}x typical episode length ({ref_text})")
    elif _within(ratio, 2.0, cfg.double_tolerance):
        v.decide(
            FileClass.EPISODE,
            f"{ratio:.2f}x episode length: likely two-part episode or two episodes in one file",
        )
    elif ratio < cfg.extra_max_ratio:
        v.decide(FileClass.EXTRA, f"only {ratio:.2f}x episode length ({ref_text})")
    elif ratio < 1:
        v.decide(FileClass.UNKNOWN, f"shorter than a typical episode ({ratio:.2f}x {ref_text})")
    else:
        v.decide(FileClass.UNKNOWN, f"longer than a typical episode ({ratio:.2f}x {ref_text})")


def _episode_count_hint(ratio: float | None, file_class: FileClass, cfg: ClassifyConfig) -> int:
    is_double = ratio is not None and _within(ratio, 2.0, cfg.double_tolerance)
    return 2 if file_class is FileClass.EPISODE and is_double else 1


# --- entry point --------------------------------------------------------------


def classify(
    files: Sequence[MediaFile],
    *,
    expected_runtime_seconds: float | None = None,
    config: ClassifyConfig | None = None,
) -> ClassifyResult:
    """Classify every file from one disc.

    ``expected_runtime_seconds`` is the typical episode runtime (e.g. the median
    TMDB runtime for the season); it helps on discs with few episodes and on
    shows with unusually short or long episodes.
    """
    cfg = config or ClassifyConfig()
    verdicts = [_Verdict(file=f) for f in files]

    _mark_junk(verdicts, cfg)
    _mark_play_all(verdicts, cfg)

    remaining = [v for v in verdicts if v.file_class is None]
    reference, source, notes = _find_reference(
        [v.file for v in remaining if not v.suspected_play_all_of],
        expected_runtime_seconds,
        cfg,
    )
    for v in remaining:
        _classify_by_ratio(v, reference, cfg)

    items = []
    for v in verdicts:
        assert v.file_class is not None
        ratio = v.file.duration_seconds / reference if reference else None
        items.append(
            Classification(
                file=v.file,
                file_class=v.file_class,
                reasons=tuple(v.reasons or ()),
                length_ratio=ratio,
                episode_count_hint=_episode_count_hint(ratio, v.file_class, cfg),
                play_all_of=v.play_all_of or v.suspected_play_all_of,
                duplicate_of=v.duplicate_of,
            )
        )
    return ClassifyResult(
        items=tuple(items),
        reference_seconds=reference,
        reference_source=source,
        notes=tuple(notes),
    )
