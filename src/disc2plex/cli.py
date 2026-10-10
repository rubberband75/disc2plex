"""Typer entry point for the ``disc2plex`` command.

Commands not yet implemented are declared so ``--help`` documents the intended
interface; they exit with a "not implemented" message.
"""

from __future__ import annotations

import logging
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from disc2plex import __version__
from disc2plex.classify import classify
from disc2plex.report import render_scan, scan_to_json
from disc2plex.scan import ProbeCache, default_cache_path, scan_directory

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


class ExtrasMode(StrEnum):
    REVIEW = "review"
    SEASON0 = "season0"
    SKIP = "skip"


# Shared option types so `plan` and `apply` stay in sync.
RipDir = Annotated[
    Path,
    typer.Argument(
        exists=True, file_okay=False, dir_okay=True, help="Folder of .mkv files from one disc."
    ),
]
ShowOpt = Annotated[str, typer.Option("--show", help="Show name to look up on TMDB.")]
YearOpt = Annotated[int | None, typer.Option("--year", help="Show's first-air year.")]
TmdbIdOpt = Annotated[int | None, typer.Option("--tmdb-id", help="TMDB show ID (skips search).")]
SeasonOpt = Annotated[int | None, typer.Option("--season", help="Season number on this disc.")]
DiscOpt = Annotated[int | None, typer.Option("--disc", help="Disc number within the season.")]
DiscEpisodesOpt = Annotated[
    str | None,
    typer.Option("--disc-episodes", help="Episode range on this disc, e.g. 1-4."),
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


@app.command()
def plan(
    rip_dir: RipDir,
    show: ShowOpt,
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
    _not_implemented(6)


@app.command()
def apply(
    rip_dir: RipDir,
    show: ShowOpt,
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


@tmdb_app.command("search")
def tmdb_search(
    query: Annotated[str, typer.Argument(help="Show name to search for.")],
    year: YearOpt = None,
) -> None:
    """Search TMDB for a show and print matching IDs."""
    _not_implemented(5)
