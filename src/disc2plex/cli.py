"""Typer entry point for the ``disc2plex`` command.

Commands not yet implemented are declared so ``--help`` documents the intended
interface; they exit with a "not implemented" message. Pipeline logic lives in
``pipeline.py``; this module only parses options, prompts, and prints.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from disc2plex import __version__
from disc2plex.classify import classify
from disc2plex.config import load_dotenv
from disc2plex.models import PlanAction, ScanResult, ShowSummary
from disc2plex.pipeline import PlanRequest, PlanRun, parse_disc_range, run_plan
from disc2plex.plan import ExtrasMode, PlanOptions
from disc2plex.report import (
    plan_to_json,
    render_plan,
    render_scan,
    render_search_results,
    render_season,
    render_show,
    scan_to_json,
)
from disc2plex.scan import ProbeCache, default_cache_path, scan_directory
from disc2plex.tmdb import (
    AmbiguousShowError,
    TMDBCache,
    TMDBClient,
    TMDBError,
    default_tmdb_cache_dir,
    season_episodes,
)

app = typer.Typer(
    help="Identify, rename, and organize MakeMKV TV-series rips for Plex.",
    no_args_is_help=True,
)
tmdb_app = typer.Typer(help="TMDB helper commands.", no_args_is_help=True)
app.add_typer(tmdb_app, name="tmdb")


class VerifyMode(StrEnum):
    NONE = "none"
    LOW = "low"
    ALL = "all"


class LLMChoice(StrEnum):
    CLAUDE = "claude"
    LOCAL = "local"
    NONE = "none"


class FileAction(StrEnum):
    MOVE = "move"
    HARDLINK = "hardlink"
    COPY = "copy"


# Shared option types so `plan` and `apply` stay in sync.
RipDir = Annotated[
    Path,
    typer.Argument(
        exists=True, file_okay=False, dir_okay=True, help="Folder of .mkv files from one disc."
    ),
]
ShowOpt = Annotated[str | None, typer.Option("--show", help="Show name to look up on TMDB.")]
YearOpt = Annotated[int | None, typer.Option("--year", help="Show's first-air year.")]
TmdbIdOpt = Annotated[int | None, typer.Option("--tmdb-id", help="TMDB show ID (skips search).")]
SeasonOpt = Annotated[int | None, typer.Option("--season", help="Season number on this disc.")]
DiscOpt = Annotated[int | None, typer.Option("--disc", help="Disc number within the season.")]
DiscEpisodesOpt = Annotated[
    str | None,
    typer.Option(
        "--disc-episodes",
        help="Episode positions on this disc in the chosen order, e.g. 1-4.",
    ),
]
OrderOpt = Annotated[
    str, typer.Option("--order", help="Episode order: aired, dvd, or a TMDB episode-group ID.")
]
OutputOpt = Annotated[Path | None, typer.Option("--output", help="Plex TV library root.")]
VerifyOpt = Annotated[
    VerifyMode, typer.Option("--verify", help="When to run content verification.")
]
LLMOpt = Annotated[LLMChoice, typer.Option("--llm", help="LLM backend for content matching.")]
WhisperOpt = Annotated[str, typer.Option("--whisper-model", help="faster-whisper model size.")]
LangOpt = Annotated[str, typer.Option("--lang", help="Preferred audio language.")]
ActionOpt = Annotated[FileAction, typer.Option("--action", help="How to place files.")]
TitlesOpt = Annotated[
    bool,
    typer.Option("--include-titles/--no-titles", help="Append episode titles to filenames."),
]
ExtrasOpt = Annotated[ExtrasMode, typer.Option("--extras", help="Where extras go.")]
YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="Skip interactive prompts.")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable output.")]

log = logging.getLogger(__name__)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"disc2plex {__version__}")
        raise typer.Exit()


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def require_ffprobe() -> None:
    """Exit with a clear error if ``ffprobe`` is not on PATH."""
    if shutil.which("ffprobe") is None:
        typer.secho(
            "Error: ffprobe not found. Install FFmpeg (e.g. `sudo apt install ffmpeg`).",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


def _not_implemented(milestone: int) -> None:
    typer.secho(f"Not implemented yet (milestone {milestone}).", err=True, fg=typer.colors.YELLOW)
    raise typer.Exit(code=1)


@app.callback()
def main(
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="-v for info, -vv for debug.")
    ] = 0,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """Identify, rename, and organize MakeMKV TV-series rips for Plex."""
    _configure_logging(verbose)


@app.command()
def scan(
    rip_dir: RipDir,
    recursive: Annotated[
        bool, typer.Option("--recursive", help="Descend into subfolders.")
    ] = False,
    json_out: JsonOpt = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Re-probe every file, ignoring the cache.")
    ] = False,
    episode_length: Annotated[
        float | None,
        typer.Option(
            "--episode-length",
            min=1,
            help="Expected episode runtime in minutes (helps on discs with few episodes).",
        ),
    ] = None,
) -> None:
    """Show probed files and their classification."""
    require_ffprobe()
    cache = None if no_cache else ProbeCache(default_cache_path())
    result = scan_directory(rip_dir, recursive=recursive, cache=cache)
    expected = episode_length * 60 if episode_length else None
    classified = classify(result.files, expected_runtime_seconds=expected)
    if json_out:
        typer.echo(scan_to_json(result, classified))
    else:
        render_scan(result, classified, Console())


def _fail(message: str) -> typer.Exit:
    typer.secho(f"Error: {message}", err=True, fg=typer.colors.RED)
    return typer.Exit(code=1)


def _plan_request(
    rip_dir: Path,
    output: Path | None,
    show: str | None,
    year: int | None,
    tmdb_id: int | None,
    season: int | None,
    disc: int | None,
    disc_episodes: str | None,
    order: str,
    action: FileAction,
    include_titles: bool,
    extras: ExtrasMode,
) -> PlanRequest:
    if output is None:
        raise _fail("--output is required (your Plex TV library root)")
    if show is None and tmdb_id is None:
        raise _fail("pass --show NAME or --tmdb-id ID")
    try:
        disc_range = parse_disc_range(disc_episodes) if disc_episodes else None
    except ValueError as exc:
        raise _fail(f"--disc-episodes: {exc}") from exc
    return PlanRequest(
        rip_dir=rip_dir.resolve(),
        output_root=output.expanduser().resolve(),
        show_name=show,
        year=year,
        tmdb_id=tmdb_id,
        season=season,
        disc=disc,
        disc_range=disc_range,
        order=order,
        options=PlanOptions(
            action=PlanAction(action.value), include_titles=include_titles, extras=extras
        ),
    )


def _stdin_is_terminal() -> bool:
    return sys.stdin.isatty()


def _prompt_for_show(candidates: Sequence[ShowSummary]) -> int:
    console = Console()
    console.print("Several shows match. Which one?")
    for number, c in enumerate(candidates, start=1):
        console.print(f"  {number}. {c.name} ({c.year or '?'})  tmdb-{c.tmdb_id}")
    while True:
        choice: int = typer.prompt("Number", type=int)
        if 1 <= choice <= len(candidates):
            return candidates[choice - 1].tmdb_id
        console.print(f"Enter a number from 1 to {len(candidates)}.")


def _run_pipeline(request: PlanRequest, *, interactive: bool) -> PlanRun:
    cache = ProbeCache(default_cache_path())

    def scanner(rip_dir: Path) -> ScanResult:
        return scan_directory(rip_dir, cache=cache)

    with tmdb_session() as client:
        try:
            return run_plan(request, scanner=scanner, client=client, exists=Path.exists)
        except AmbiguousShowError as exc:
            if not interactive:
                raise
            chosen = replace(request, tmdb_id=_prompt_for_show(exc.candidates))
            return run_plan(chosen, scanner=scanner, client=client, exists=Path.exists)
        except ValueError as exc:  # e.g. --disc-episodes outside the season
            raise _fail(str(exc)) from exc


@app.command()
def plan(
    rip_dir: RipDir,
    show: ShowOpt = None,
    year: YearOpt = None,
    tmdb_id: TmdbIdOpt = None,
    season: SeasonOpt = None,
    disc: DiscOpt = None,
    disc_episodes: DiscEpisodesOpt = None,
    order: OrderOpt = "aired",
    output: OutputOpt = None,
    verify: VerifyOpt = VerifyMode.LOW,
    llm: LLMOpt = LLMChoice.NONE,
    whisper_model: WhisperOpt = "small",
    lang: LangOpt = "en",
    action: ActionOpt = FileAction.MOVE,
    include_titles: TitlesOpt = True,
    extras: ExtrasOpt = ExtrasMode.REVIEW,
    yes: YesOpt = False,
    json_out: JsonOpt = False,
) -> None:
    """Run the full pipeline and print the rename plan (always a dry run)."""
    require_ffprobe()
    request = _plan_request(
        rip_dir, output, show, year, tmdb_id, season, disc, disc_episodes, order,
        action, include_titles, extras,
    )  # fmt: skip
    if verify is not VerifyMode.NONE and not json_out:
        typer.secho(
            "Note: content verification isn't available yet; matching uses title order "
            "and runtimes only.",
            err=True,
            dim=True,
        )
    interactive = not yes and not json_out and _stdin_is_terminal()
    run = _run_pipeline(request, interactive=interactive)
    if json_out:
        typer.echo(plan_to_json(run))
    else:
        render_plan(run, Console())
    if run.plan.errors:
        raise typer.Exit(code=1)


@app.command()
def apply(
    rip_dir: RipDir,
    show: ShowOpt = None,
    year: YearOpt = None,
    tmdb_id: TmdbIdOpt = None,
    season: SeasonOpt = None,
    disc: DiscOpt = None,
    disc_episodes: DiscEpisodesOpt = None,
    order: OrderOpt = "aired",
    output: OutputOpt = None,
    verify: VerifyOpt = VerifyMode.LOW,
    llm: LLMOpt = LLMChoice.NONE,
    whisper_model: WhisperOpt = "small",
    lang: LangOpt = "en",
    action: ActionOpt = FileAction.MOVE,
    include_titles: TitlesOpt = True,
    extras: ExtrasOpt = ExtrasMode.REVIEW,
    do_apply: Annotated[
        bool, typer.Option("--apply", help="Actually move files. Without it, this is a dry run.")
    ] = False,
    yes: YesOpt = False,
    json_out: JsonOpt = False,
) -> None:
    """Build the plan, then execute it if --apply is given."""
    require_ffprobe()
    _not_implemented(7)


@app.command()
def undo(
    manifest: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Manifest JSON from an apply run.")
    ],
    yes: YesOpt = False,
) -> None:
    """Reverse a previous apply using its manifest."""
    _not_implemented(7)


@contextmanager
def tmdb_session() -> Iterator[TMDBClient]:
    """Yield a cached TMDB client; turn TMDB errors into a clean CLI exit."""
    load_dotenv()
    try:
        with TMDBClient(
            os.environ.get("TMDB_API_TOKEN", ""), cache=TMDBCache(default_tmdb_cache_dir())
        ) as client:
            yield client
    except TMDBError as exc:
        typer.secho(f"Error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@tmdb_app.command("search")
def tmdb_search(
    query: Annotated[str, typer.Argument(help="Show name to search for.")],
    year: YearOpt = None,
) -> None:
    """Search TMDB for a show and print matching IDs."""
    with tmdb_session() as client:
        render_search_results(client.search_tv(query, year), Console())


@tmdb_app.command("show")
def tmdb_show(tmdb_id: Annotated[int, typer.Argument(help="TMDB show ID.")]) -> None:
    """List a show's seasons and episode groups (IDs usable with --order)."""
    with tmdb_session() as client:
        render_show(client.get_show(tmdb_id), Console())


@tmdb_app.command("season")
def tmdb_season(
    tmdb_id: Annotated[int, typer.Argument(help="TMDB show ID.")],
    season: Annotated[int, typer.Argument(help="Season number.")],
    order: OrderOpt = "aired",
) -> None:
    """List a season's episodes in aired, DVD, or episode-group order."""
    with tmdb_session() as client:
        show = client.get_show(tmdb_id)
        render_season(season_episodes(client, show, season, order), Console())
