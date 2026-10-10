"""Glue for the full scan -> classify -> TMDB -> match -> plan pipeline.

Shared by the ``plan`` and ``apply`` commands. Side effects come only
from the injected scanner, TMDB client, and ``exists`` check.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from disc2plex.classify import classify
from disc2plex.match.combine import Thresholds, apply_overrides, decide
from disc2plex.match.content import ContentVerifier, VerifyMode, apply_evidence
from disc2plex.match.heuristic import ContentVotes, match_heuristic
from disc2plex.models import (
    ClassifyResult,
    EpisodeInfo,
    MatchResult,
    MediaFile,
    Plan,
    ScanResult,
    SeasonEpisodes,
    ShowInfo,
)
from disc2plex.plan import PlanOptions, build_plan
from disc2plex.tmdb import (
    TMDBClient,
    TMDBError,
    expected_runtime_seconds,
    resolve_show,
    season_episodes,
    season_runtimes_seconds,
)

_RANGE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


def parse_disc_range(text: str) -> tuple[int, int]:
    """Parse ``"1-4"`` or ``"5"`` into an inclusive (start, end) pair."""
    match = _RANGE.match(text)
    if not match:
        raise ValueError(f"expected a range like 1-4 or a single number, got {text!r}")
    start = int(match.group(1))
    end = int(match.group(2) or start)
    if start < 1 or end < start:
        raise ValueError(f"invalid episode range {text!r}")
    return start, end


def parse_assign(text: str) -> tuple[str, tuple[int, ...]]:
    """Parse ``"t04=6"`` or ``"title_t00.mkv=1-2"`` into (file spec, episode numbers)."""
    spec, sep, episodes = text.rpartition("=")
    if not sep or not spec.strip():
        raise ValueError(f"expected FILE=EPISODES like t04=6 or t00=1-2, got {text!r}")
    episodes = episodes.strip().lower().removeprefix("e").replace("-e", "-")
    start, end = parse_disc_range(episodes)
    return spec.strip(), tuple(range(start, end + 1))


def _file_matches(spec: str, media: MediaFile) -> bool:
    if spec in (media.path.name, media.path.stem):
        return True
    index = re.fullmatch(r"t?(\d+)", spec.lower())
    return index is not None and media.makemkv_title_index == int(index.group(1))


def resolve_overrides(
    specs: Sequence[tuple[str, tuple[int, ...]]],
    files: Sequence[MediaFile],
    season: SeasonEpisodes,
) -> dict[Path, tuple[EpisodeInfo, ...]]:
    """Map ``--assign`` specs to files and episodes (aired numbers in ``season``)."""
    by_number = {e.episode: e for e in season.episodes if e.season == season.season}
    overrides: dict[Path, tuple[EpisodeInfo, ...]] = {}
    claimed: dict[int, str] = {}
    for spec, numbers in specs:
        matches = [f for f in files if _file_matches(spec, f)]
        if len(matches) != 1:
            found = "no file" if not matches else f"{len(matches)} files"
            raise ValueError(f"--assign {spec}: {found} on this disc match {spec!r}")
        path = matches[0].path
        if path in overrides:
            raise ValueError(f"--assign: {path.name} is assigned twice")
        missing = [n for n in numbers if n not in by_number]
        if missing:
            raise ValueError(
                f"--assign {spec}: season {season.season} has no episode "
                f"{', '.join(str(n) for n in missing)}"
            )
        for n in numbers:
            if n in claimed:
                raise ValueError(f"--assign: episode {n} is given to both {claimed[n]} and {spec}")
            claimed[n] = spec
        overrides[path] = tuple(by_number[n] for n in numbers)
    return overrides


def infer_season(show: ShowInfo) -> int:
    """Return the only regular season, or raise asking for --season."""
    regular = [s.number for s in show.seasons if s.number > 0]
    if len(regular) == 1:
        return regular[0]
    raise TMDBError(
        f"{show.name} has {len(regular)} seasons; pass --season "
        f"(available: {', '.join(str(s.number) for s in show.seasons)})"
    )


@dataclass(frozen=True)
class PlanRequest:
    rip_dir: Path
    output_root: Path
    show_name: str | None = None
    year: int | None = None
    tmdb_id: int | None = None
    season: int | None = None
    disc: int | None = None
    disc_range: tuple[int, int] | None = None
    order: str = "aired"
    verify: VerifyMode = VerifyMode.NONE
    options: PlanOptions = field(default_factory=PlanOptions)
    thresholds: Thresholds = field(default_factory=Thresholds)
    # User-identified files from --assign: (file spec, aired episode numbers).
    assign: tuple[tuple[str, tuple[int, ...]], ...] = ()


@dataclass(frozen=True)
class PlanRun:
    """Every intermediate result, so reports can show the evidence."""

    scan: ScanResult
    show: ShowInfo
    season: SeasonEpisodes
    classified: ClassifyResult
    match: MatchResult
    plan: Plan


def run_plan(
    request: PlanRequest,
    *,
    scanner: Callable[[Path], ScanResult],
    client: TMDBClient,
    exists: Callable[[Path], bool] = lambda _p: False,
    verifier: ContentVerifier | None = None,
) -> PlanRun:
    """Run every stage up to (not including) applying the plan.

    Content verification runs only when a ``verifier`` is given and
    ``request.verify`` isn't ``none``.
    """
    show = resolve_show(client, name=request.show_name, year=request.year, tmdb_id=request.tmdb_id)
    season_number = request.season if request.season is not None else infer_season(show)
    season = season_episodes(client, show, season_number, request.order)

    scan = scanner(request.rip_dir)
    classified = classify(
        scan.files,
        expected_runtime_seconds=expected_runtime_seconds(season),
        season_runtimes_seconds=season_runtimes_seconds(season),
    )

    overrides = resolve_overrides(request.assign, scan.files, season)

    def align(votes: ContentVotes | None = None) -> MatchResult:
        result = match_heuristic(
            classified.items,
            season,
            disc_range=request.disc_range,
            disc_number=request.disc,
            content_votes=votes,
        )
        # Before verification too, so assigned files aren't sent for checking.
        return apply_overrides(result, overrides)

    heuristic = align()
    if verifier is not None and request.verify is not VerifyMode.NONE:
        evidence = verifier.gather(heuristic, season, request.verify)
        if evidence.by_file:
            heuristic = align(evidence.votes(verifier.config.min_answer_confidence))
            if verifier.listen_again(heuristic, season, evidence):
                heuristic = align(evidence.votes(verifier.config.min_answer_confidence))
        heuristic = apply_overrides(apply_evidence(heuristic, evidence, verifier.config), overrides)
    matched = decide(heuristic, request.thresholds)
    plan = build_plan(
        matched,
        show,
        request.output_root,
        request.rip_dir.name,
        request.options,
        exists=exists,
    )
    if scan.failures:
        names = ", ".join(f.path.name for f in scan.failures)
        plan = plan.model_copy(
            update={"warnings": (*plan.warnings, f"could not probe (left in place): {names}")}
        )
    return PlanRun(
        scan=scan, show=show, season=season, classified=classified, match=matched, plan=plan
    )
