"""Stage 4: match episode files to TMDB episodes by title order and runtime.

Model: MakeMKV titles (in title-index order) usually correspond to a contiguous
run of episodes in the chosen ordering. A dynamic program aligns files to
episode positions, scoring each pairing by runtime similarity, and allows:

- a double-length file to take two consecutive episodes (if its runtime fits),
- an episode to be skipped mid-run (penalised; signals a missing episode),
- a file to be left unmatched (heavily penalised).

When the start of the run isn't given, every start is tried. How clearly the
best start beats the runner-up becomes the "order" confidence, so a disc whose
episodes all run 44 minutes honestly comes out as needing review.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from disc2plex.models import (
    Assignment,
    Classification,
    EpisodeInfo,
    FileClass,
    MatchResult,
    MediaFile,
    SeasonEpisodes,
)
from disc2plex.report import format_duration
from disc2plex.scan import title_sort_key


@dataclass(frozen=True)
class HeuristicConfig:
    # Relative runtime error at which fit falls to ~0.37 (TMDB runtimes are coarse).
    runtime_sigma: float = 0.15
    # Fit used when TMDB has no runtime: neutral, so it doesn't steer alignment.
    unknown_runtime_fit: float = 0.55
    # Log-score penalties for skipping an episode mid-run / leaving a file unmatched.
    skip_penalty: float = math.log(0.15)
    drop_penalty: float = math.log(0.02)
    # Mild prior that a file holds one episode: breaks ties between "one long
    # episode" and "two normal episodes" when runtimes fit both equally.
    double_penalty: float = math.log(0.7)
    # Order confidence when the start is pinned by the user or by "disc 1".
    range_confidence: float = 0.95
    disc1_confidence: float = 0.90
    # Inferred start: confidence = max - spread * exp(-margin / scale).
    inferred_max_confidence: float = 0.90
    inferred_spread: float = 0.40
    inferred_margin_scale: float = 1.5
    # Confidence ceiling when the run has gaps or doesn't fill the given range.
    gap_confidence_cap: float = 0.80
    # Multiplier applied to TMDB runtimes before comparing. Set per disc by
    # calibration (real discs often run 5-20% longer than TMDB's broadcast
    # runtimes), clamped to this range so a wrong season can't be "fixed" by it.
    runtime_scale: float = 1.0
    min_runtime_scale: float = 0.80
    max_runtime_scale: float = 1.35
    # Weight of transcript evidence in pair scores (< 1: LLM confidence is not
    # calibrated, so one confident-but-wrong answer can't outvote order + runtime).
    content_weight: float = 0.6


# Transcript evidence per file: ((season, episode), confidence) per answered excerpt.
ContentVotes = Mapping[Path, Sequence[tuple[tuple[int, int], float]]]

# Runtime fit below this (about 17% off with the default sigma) is called a mismatch.
MISMATCH_FIT = 0.3


@dataclass
class _Path:
    """One alignment: per-file episode position ranges (None = unmatched)."""

    score: float
    spans: list[tuple[int, int] | None] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)


def runtime_fit(
    duration_seconds: float, episodes: Sequence[EpisodeInfo], cfg: HeuristicConfig
) -> float | None:
    """Similarity in (0, 1] between a file's duration and the episodes' total runtime.

    Returns None if any episode lacks a TMDB runtime.
    """
    runtimes = [e.runtime_minutes for e in episodes]
    if any(r is None for r in runtimes):
        return None
    expected = sum(r for r in runtimes if r is not None) * 60 * cfg.runtime_scale
    relative_error = abs(duration_seconds - expected) / expected
    return math.exp(-((relative_error / cfg.runtime_sigma) ** 2))


def content_score(
    votes: Sequence[tuple[tuple[int, int], float]],
    episodes: Sequence[EpisodeInfo],
    candidate_count: int,
    cfg: HeuristicConfig,
) -> float:
    """Log-likelihood ratio of the transcript answers if the file holds ``episodes``.

    Each answer "excerpt is episode X with confidence c" is treated as: true
    episode answered with probability c, otherwise one of the other N-1 at
    random; scored against a uniform 1/N baseline and averaged over excerpts.
    A matching answer adds ~log(cN), a contradicting one subtracts.
    """
    if not votes or candidate_count < 2:
        return 0.0
    keys = {(e.season, e.episode) for e in episodes}
    n = candidate_count
    total = 0.0
    for key, confidence in votes:
        c = min(max(confidence, 0.05), 0.95)
        total += math.log(c * n) if key in keys else math.log((1 - c) * n / (n - 1))
    return cfg.content_weight * total / len(votes)


def _pair_score(
    item: Classification,
    episodes: Sequence[EpisodeInfo],
    cfg: HeuristicConfig,
    votes: Sequence[tuple[tuple[int, int], float]] = (),
    candidate_count: int = 0,
) -> float:
    fit = runtime_fit(item.file.duration_seconds, episodes, cfg)
    if fit is None:
        # No runtimes: trust classification's guess of how many episodes it holds.
        fit = cfg.unknown_runtime_fit if len(episodes) == item.episode_count_hint else 0.25
    return math.log(max(fit, 1e-6)) + content_score(votes, episodes, candidate_count, cfg)


def _align(
    items: Sequence[Classification],
    episodes: Sequence[EpisodeInfo],
    start: int,
    end: int,
    cfg: HeuristicConfig,
    votes: ContentVotes | None = None,
) -> _Path:
    """Best alignment of ``items`` onto episode positions ``start..end-1`` (0-based)."""
    m = len(items)
    neg = -math.inf
    # best[i][j]: best score after placing i files with next free episode position j.
    best = [[neg] * (end + 1) for _ in range(m + 1)]
    back: list[list[tuple[str, int, int] | None]] = [[None] * (end + 1) for _ in range(m + 1)]
    best[0][start] = 0.0

    def evidence(item: Classification) -> tuple[Sequence[tuple[tuple[int, int], float]], int]:
        return (votes or {}).get(item.file.path, ()), len(episodes)

    def relax(i: int, j: int, score: float, move: str, pi: int, pj: int) -> None:
        if score > best[i][j]:
            best[i][j] = score
            back[i][j] = (move, pi, pj)

    for i in range(m + 1):
        for j in range(start, end + 1):
            score = best[i][j]
            if score == neg:
                continue
            if i > 0 and j < end:
                relax(i, j + 1, score + cfg.skip_penalty, "skip", i, j)
            if i == m:
                continue
            item = items[i]
            relax(i + 1, j, score + cfg.drop_penalty, "drop", i, j)
            if j < end:
                one = _pair_score(item, episodes[j : j + 1], cfg, *evidence(item))
                relax(i + 1, j + 1, score + one, "one", i, j)
            if j + 1 < end and item.episode_count_hint >= 2:
                two = (
                    _pair_score(item, episodes[j : j + 2], cfg, *evidence(item))
                    + cfg.double_penalty
                )
                relax(i + 1, j + 2, score + two, "two", i, j)

    # Trailing unused episodes are free (the disc may stop before the season ends).
    final_j = max(range(start, end + 1), key=lambda j: best[m][j])
    path = _Path(score=best[m][final_j], spans=[None] * m)
    i, j = m, final_j
    while (i, j) != (0, start):
        step = back[i][j]
        assert step is not None
        move, pi, pj = step
        if move == "one":
            path.spans[pi] = (pj, pj + 1)
        elif move == "two":
            path.spans[pi] = (pj, pj + 2)
        elif move == "skip":
            path.skipped.append(pj)
        i, j = pi, pj
    # Skips after the last matched file are just the end of the disc, not gaps.
    last_used = max((s[1] for s in path.spans if s), default=start)
    path.skipped = sorted(p for p in path.skipped if p < last_used)
    return path


def _order_confidence(margin: float, cfg: HeuristicConfig) -> float:
    return cfg.inferred_max_confidence - cfg.inferred_spread * math.exp(
        -margin / cfg.inferred_margin_scale
    )


def _runtime_factor(fit: float | None) -> float:
    """Scale order confidence by runtime agreement (unknown runtime: slight discount)."""
    return min(1.0, 0.4 + (0.55 if fit is None else fit))


def calibrate_runtime_scale(
    items: Sequence[Classification], episodes: Sequence[EpisodeInfo], cfg: HeuristicConfig
) -> float:
    """How much longer (or shorter) this disc's episodes run than TMDB says.

    Each single-episode file is compared with the nearest TMDB runtime in the
    season (so a disc of atypical-length episodes still calibrates correctly);
    the median ratio is the scale. Needs two files and some runtimes, else 1.0.
    """
    runtimes = sorted({e.runtime_minutes * 60 for e in episodes if e.runtime_minutes})
    ratios = (
        [
            i.file.duration_seconds / min(runtimes, key=lambda r: abs(r - i.file.duration_seconds))
            for i in items
            if i.episode_count_hint == 1
        ]
        if runtimes
        else []
    )
    if len(ratios) < 2:
        return 1.0
    return min(max(statistics.median(ratios), cfg.min_runtime_scale), cfg.max_runtime_scale)


def _describe_runtime(
    file: MediaFile, eps: Sequence[EpisodeInfo], fit: float | None, cfg: HeuristicConfig
) -> str:
    actual = format_duration(file.duration_seconds)
    if fit is None:
        return f"runtime {actual}; TMDB lists no runtime to compare"
    expected = sum(e.runtime_minutes or 0 for e in eps) * 60 * cfg.runtime_scale
    off = abs(file.duration_seconds - expected) / expected
    parts = " + ".join(f"{e.runtime_minutes:g}" for e in eps)
    scaled = f" x{cfg.runtime_scale:.2f}" if abs(cfg.runtime_scale - 1) >= 0.01 else ""
    verdict = (
        "matches" if fit >= 0.6 else "roughly matches" if fit >= MISMATCH_FIT else "differs from"
    )
    return f"runtime {actual} {verdict} TMDB {parts} min{scaled} ({off:.0%} off)"


def match_heuristic(
    classified: Sequence[Classification],
    season: SeasonEpisodes,
    *,
    disc_range: tuple[int, int] | None = None,
    disc_number: int | None = None,
    config: HeuristicConfig | None = None,
    content_votes: ContentVotes | None = None,
) -> MatchResult:
    """Assign EPISODE-class files to episodes of ``season``.

    ``disc_range`` is an inclusive 1-based range of positions in the season's
    chosen order (for aired order these are the episode numbers). Non-episode
    files are passed through unassigned; statuses are set later by ``combine``.
    ``content_votes`` (from transcript verification) adds evidence to each
    pairing's score, which can also pin down an otherwise ambiguous start.
    """
    cfg = config or HeuristicConfig()
    episodes = list(season.episodes)
    n = len(episodes)
    notes: list[str] = []
    items = sorted(
        (c for c in classified if c.file_class is FileClass.EPISODE),
        key=lambda c: title_sort_key(c.file.path),
    )
    passthrough = [
        Assignment(file=c.file, file_class=c.file_class, reasons=c.reasons)
        for c in classified
        if c.file_class is not FileClass.EPISODE
    ]
    if not items or n == 0:
        if items:
            notes.append(f"{season.label} has no episodes on TMDB")
        unmatched = [
            Assignment(file=c.file, file_class=c.file_class, reasons=(*c.reasons, "not matched"))
            for c in items
        ]
        return MatchResult(assignments=tuple(unmatched + passthrough), notes=tuple(notes))

    # Decide which start positions to try.
    if disc_range is not None:
        lo, hi = disc_range
        if not (1 <= lo <= hi <= n):
            raise ValueError(f"--disc-episodes {lo}-{hi} is outside {season.label} (1-{n})")
        starts, end, source = [lo - 1], hi, "range"
        pinned_conf = cfg.range_confidence
    elif disc_number == 1:
        starts, end, source = [0], n, "disc1"
        pinned_conf = cfg.disc1_confidence
    else:
        starts, end, source = list(range(n)), n, "inferred"
        pinned_conf = None

    scale = calibrate_runtime_scale(items, episodes, cfg)
    cfg = replace(cfg, runtime_scale=scale)
    if abs(scale - 1) >= 0.05:
        notes.append(
            f"this disc's episodes run {scale:.2f}x TMDB's listed runtimes; "
            "runtimes were scaled before comparing"
        )
    paths = {s: _align(items, episodes, s, end, cfg, content_votes) for s in starts}
    best_start = max(paths, key=lambda s: paths[s].score)
    best = paths[best_start]

    if pinned_conf is not None:
        order_conf = pinned_conf
        start_text = (
            f"disc range {disc_range[0]}-{disc_range[1]}"
            if disc_range
            else "disc 1 starts at the first episode"
        )
    else:
        others = [p.score for s, p in paths.items() if s != best_start]
        margin = best.score - max(others) if others else math.inf
        order_conf = _order_confidence(margin, cfg)
        start_text = (
            f"start inferred from {'runtimes and transcripts' if content_votes else 'runtimes'} "
            f"(margin {margin:.1f} over next-best start)"
            if math.isfinite(margin)
            else "only one possible start"
        )
        if order_conf < 0.7:
            notes.append(
                "the starting episode is ambiguous from runtimes alone; pass "
                "--disc-episodes (e.g. 5-8) or --disc to pin it"
            )

    gap_reasons: list[str] = []
    if best.skipped:
        missing = ", ".join(str(p + 1) for p in best.skipped)
        gap_reasons.append(f"episode position(s) {missing} appear to be missing from the disc")
    used_end = max((s[1] for s in best.spans if s), default=best_start)
    if disc_range is not None and used_end < end:
        gap_reasons.append(
            f"range {disc_range[0]}-{disc_range[1]} has more episodes than matched files"
        )
    notes.extend(gap_reasons)

    assignments: list[Assignment] = []
    for item, span in zip(items, best.spans, strict=True):
        if span is None:
            assignments.append(
                Assignment(
                    file=item.file,
                    file_class=item.file_class,
                    reasons=(
                        *item.reasons,
                        "no episode fits this file in title order (titles may be out of order)",
                    ),
                )
            )
            continue
        eps = episodes[span[0] : span[1]]
        fit = runtime_fit(item.file.duration_seconds, eps, cfg)
        confidence = order_conf * _runtime_factor(fit)
        reasons = [
            f"title order -> position {span[0] + 1}" + (f"-{span[1]}" if len(eps) > 1 else ""),
            start_text,
            _describe_runtime(item.file, eps, fit, cfg),
        ]
        if fit is not None and fit < MISMATCH_FIT:
            reasons.append("runtime mismatch: titles may be out of order or misclassified")
        if gap_reasons:
            confidence = min(confidence, cfg.gap_confidence_cap)
            reasons.extend(gap_reasons)
        signals = {"order": round(order_conf, 3)}
        if fit is not None:
            signals["runtime_fit"] = round(fit, 3)
        assignments.append(
            Assignment(
                file=item.file,
                file_class=item.file_class,
                episodes=tuple(eps),
                confidence=round(confidence, 3),
                signals=signals,
                reasons=tuple(reasons),
            )
        )

    return MatchResult(
        assignments=tuple(assignments + passthrough),
        start_position=best_start + 1,
        start_source=source,
        runtime_scale=round(scale, 4),
        notes=tuple(notes),
    )
