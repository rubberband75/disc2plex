"""Glue for the full scan -> classify -> TMDB -> match -> plan pipeline.

Shared by the ``plan`` and (later) ``apply`` commands. Side effects come only
from the injected scanner, TMDB client, and ``exists`` check.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from disc2plex.classify import classify
from disc2plex.match.combine import Thresholds, decide
from disc2plex.match.heuristic import match_heuristic
from disc2plex.models import ClassifyResult, MatchResult, Plan, ScanResult, SeasonEpisodes, ShowInfo
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
    options: PlanOptions = field(default_factory=PlanOptions)
    thresholds: Thresholds = field(default_factory=Thresholds)


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
) -> PlanRun:
    """Run every stage up to (not including) applying the plan."""
    show = resolve_show(client, name=request.show_name, year=request.year, tmdb_id=request.tmdb_id)
    season_number = request.season if request.season is not None else infer_season(show)
    season = season_episodes(client, show, season_number, request.order)

    scan = scanner(request.rip_dir)
    classified = classify(
        scan.files,
        expected_runtime_seconds=expected_runtime_seconds(season),
        season_runtimes_seconds=season_runtimes_seconds(season),
    )
    matched = decide(
        match_heuristic(
            classified.items,
            season,
            disc_range=request.disc_range,
            disc_number=request.disc,
        ),
        request.thresholds,
    )
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
