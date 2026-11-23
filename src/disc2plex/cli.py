"""Typer entry point for the ``disc2plex`` command.

Pipeline logic lives in ``pipeline.py`` and file operations in ``apply.py``;
this module only parses options, prompts, and prints.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from rich.console import Console

from disc2plex import __version__
from disc2plex.apply import (
    ApplyError,
    apply_plan,
    default_manifest_dir,
    read_manifest,
    undo_manifest,
)
from disc2plex.classify import classify
from disc2plex.config import (
    CONFIG_TEMPLATE,
    ConfigError,
    Settings,
    config_path,
    load_dotenv,
    load_settings,
)
from disc2plex.llm.base import CachedLLM, LLMBackend, LLMError, default_llm_cache_dir
from disc2plex.llm.claude import ClaudeBackend
from disc2plex.llm.openai_compat import DEFAULT_BASE_URL, OpenAICompatBackend
from disc2plex.match.combine import Thresholds
from disc2plex.match.content import ContentVerifier, VerifyMode
from disc2plex.models import (
    EntryStatus,
    FileClass,
    FileTranscript,
    Manifest,
    PlanAction,
    ScanResult,
    ShowSummary,
)
from disc2plex.pipeline import PlanRequest, PlanRun, parse_assign, parse_disc_range, run_plan
from disc2plex.plan import EXTRAS_REVIEW_DIR, REVIEW_DIR, ExtrasMode, PlanOptions, ReviewMode
from disc2plex.plex import PlexClient, PlexError, find_section, server_path
from disc2plex.report import (
    plan_payload,
    plan_to_json,
    render_manifest,
    render_plan,
    render_plex_sections,
    render_scan,
    render_search_results,
    render_season,
    render_show,
    render_transcripts,
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
from disc2plex.transcribe import (
    WHISPER_MODELS,
    FasterWhisperTranscriber,
    TranscriptCache,
    TranscriptionError,
    choose_slices,
    default_transcript_cache_dir,
    transcribe_file,
)

app = typer.Typer(
    help="Identify, rename, and organize MakeMKV TV-series rips for Plex.",
    no_args_is_help=True,
)
tmdb_app = typer.Typer(help="TMDB helper commands.", no_args_is_help=True)
app.add_typer(tmdb_app, name="tmdb")
config_app = typer.Typer(help="Show or create the config file.", no_args_is_help=True)
app.add_typer(config_app, name="config")
plex_app = typer.Typer(help="Plex Media Server helpers.", no_args_is_help=True)
app.add_typer(plex_app, name="plex")

T = TypeVar("T")


class LLMChoice(StrEnum):
    CLAUDE = "claude"
    LOCAL = "local"
    NONE = "none"


class WhisperDevice(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


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
# Options backed by config.toml default to None, meaning "not given": the config
# value applies, then the built-in default. Help text names the built-in default.
OrderOpt = Annotated[
    str | None,
    typer.Option(
        "--order", help="Episode order: aired, dvd, or a TMDB episode-group ID. \\[default: aired]"
    ),
]
TmdbOrderOpt = Annotated[
    str, typer.Option("--order", help="Episode order: aired, dvd, or a TMDB episode-group ID.")
]
OutputOpt = Annotated[
    Path | None,
    typer.Option("--output", help="Plex TV library root. \\[default: library.root from config]"),
]
VerifyOpt = Annotated[
    VerifyMode | None,
    typer.Option("--verify", help="When to run content verification. \\[default: low]"),
]
LLMOpt = Annotated[
    LLMChoice | None,
    typer.Option(
        "--llm",
        help="LLM for content verification; claude sends transcripts to the API. \\[default: none]",
    ),
]
LLMModelOpt = Annotated[
    str | None,
    typer.Option(
        "--llm-model",
        help="Model name (default: claude-opus-5-5 for claude; DISC2PLEX_LLM_MODEL for local).",
    ),
]
WhisperOpt = Annotated[
    str | None,
    typer.Option("--whisper-model", help="faster-whisper model size. \\[default: small]"),
]
WhisperDeviceOpt = Annotated[
    WhisperDevice | None,
    typer.Option(
        "--whisper-device",
        help="Where Whisper runs; auto uses an NVIDIA GPU when it can, else the CPU. "
        "\\[default: auto]",
    ),
]
LangOpt = Annotated[
    str | None, typer.Option("--lang", help="Preferred audio language. \\[default: en]")
]
ActionOpt = Annotated[
    FileAction | None, typer.Option("--action", help="How to place files. \\[default: move]")
]
TitlesOpt = Annotated[
    bool | None,
    typer.Option(
        "--include-titles/--no-titles",
        help="Append episode titles to filenames. \\[default: include]",
    ),
]
ExtrasOpt = Annotated[
    ExtrasMode | None, typer.Option("--extras", help="Where extras go. \\[default: review]")
]
ReviewOpt = Annotated[
    ReviewMode | None,
    typer.Option(
        "--review",
        help="Uncertain files: move to <output>/_review/<disc>/, or leave in the rip folder. "
        "\\[default: move]",
    ),
]
AssignOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--assign",
        help="Identify a file yourself: FILE=EPISODES (aired numbers), e.g. t04=6 or t00=1-2. "
        "Repeatable.",
    ),
]
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


def _settings() -> Settings:
    try:
        return load_settings()
    except ConfigError as exc:
        raise _fail(str(exc)) from exc


def _pick(flag: T | None, configured: T) -> T:
    """CLI flag if given, else the config value (which carries the built-in default)."""
    return configured if flag is None else flag


def _plan_request(
    settings: Settings,
    rip_dir: Path,
    output: Path | None,
    show: str | None,
    year: int | None,
    tmdb_id: int | None,
    season: int | None,
    disc: int | None,
    disc_episodes: str | None,
    order: str | None,
    action: FileAction | None,
    include_titles: bool | None,
    extras: ExtrasMode | None,
    review: ReviewMode | None,
    verify: VerifyMode | None,
    assign: list[str] | None,
) -> PlanRequest:
    lib, matching = settings.library, settings.matching
    root = _pick(output, lib.root)
    if root is None:
        raise _fail(
            "--output is required (your Plex TV library root), "
            f"or set library.root in {config_path()}"
        )
    if show is None and tmdb_id is None:
        raise _fail("pass --show NAME or --tmdb-id ID")
    try:
        disc_range = parse_disc_range(disc_episodes) if disc_episodes else None
    except ValueError as exc:
        raise _fail(f"--disc-episodes: {exc}") from exc
    try:
        overrides = tuple(parse_assign(text) for text in assign or ())
    except ValueError as exc:
        raise _fail(f"--assign: {exc}") from exc
    return PlanRequest(
        rip_dir=rip_dir.resolve(),
        output_root=root.expanduser().resolve(),
        show_name=show,
        year=year,
        tmdb_id=tmdb_id,
        season=season,
        disc=disc,
        disc_range=disc_range,
        order=_pick(order, matching.order),
        verify=_pick(verify, VerifyMode(matching.verify)),
        options=PlanOptions(
            action=PlanAction(_pick(action, FileAction(lib.action)).value),
            include_titles=_pick(include_titles, lib.include_titles),
            include_tmdb_id=lib.include_tmdb_id,
            extras=_pick(extras, ExtrasMode(lib.extras)),
            review=_pick(review, ReviewMode(lib.review)),
        ),
        thresholds=Thresholds(confident=matching.confident, review=matching.review),
        assign=overrides,
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


def _make_llm(llm: LLMChoice, model: str | None, settings: Settings) -> LLMBackend:
    """Model: --llm-model, then (local only) DISC2PLEX_LLM_MODEL, then config."""
    if llm is LLMChoice.CLAUDE:
        model = model or settings.claude.model
        return ClaudeBackend(model) if model else ClaudeBackend()
    local = settings.local_llm
    return OpenAICompatBackend(
        model or os.environ.get("DISC2PLEX_LLM_MODEL") or local.model or "",
        base_url=os.environ.get("DISC2PLEX_LLM_BASE_URL") or local.base_url or DEFAULT_BASE_URL,
        api_key=os.environ.get("DISC2PLEX_LLM_API_KEY"),
        thinking=local.thinking,
    )


def _build_verifier(
    settings: Settings,
    verify: VerifyMode,
    llm_flag: LLMChoice | None,
    llm_model: str | None,
    whisper_flag: str | None,
    device_flag: WhisperDevice | None,
    lang_flag: str | None,
    *,
    quiet: bool,
) -> ContentVerifier | None:
    """Assemble transcription + LLM for content verification, or None if not requested."""
    llm = _pick(llm_flag, LLMChoice(settings.matching.llm))
    whisper_model = _pick(whisper_flag, settings.matching.whisper_model)
    device = _pick(device_flag, WhisperDevice(settings.matching.whisper_device))
    lang = _pick(lang_flag, settings.matching.lang)
    if verify is VerifyMode.NONE:
        return None
    if llm is LLMChoice.NONE:
        if not quiet:
            typer.secho(
                "Note: content verification is off (pass --llm claude or --llm local); "
                "matching uses title order and runtimes only.",
                err=True,
                dim=True,
            )
        return None
    if whisper_model not in WHISPER_MODELS:
        raise _fail(f"--whisper-model must be one of {', '.join(WHISPER_MODELS)}")
    load_dotenv()  # ANTHROPIC_API_KEY / HF_TOKEN / DISC2PLEX_LLM_* may live in .env
    try:
        backend = _make_llm(llm, llm_model, settings)
    except LLMError as exc:
        raise _fail(str(exc)) from exc
    return ContentVerifier(
        transcriber=FasterWhisperTranscriber(whisper_model, device=device.value),
        llm=CachedLLM(backend, default_llm_cache_dir()),
        lang=lang,
        transcript_cache=TranscriptCache(default_transcript_cache_dir()),
    )


def _run_pipeline(
    request: PlanRequest, *, interactive: bool, verifier: ContentVerifier | None = None
) -> PlanRun:
    cache = ProbeCache(default_cache_path())

    def scanner(rip_dir: Path) -> ScanResult:
        return scan_directory(rip_dir, cache=cache)

    with tmdb_session() as client:
        try:
            return run_plan(
                request, scanner=scanner, client=client, exists=Path.exists, verifier=verifier
            )
        except AmbiguousShowError as exc:
            if not interactive:
                raise
            chosen = replace(request, tmdb_id=_prompt_for_show(exc.candidates))
            return run_plan(
                chosen, scanner=scanner, client=client, exists=Path.exists, verifier=verifier
            )
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
    order: OrderOpt = None,
    output: OutputOpt = None,
    verify: VerifyOpt = None,
    llm: LLMOpt = None,
    llm_model: LLMModelOpt = None,
    whisper_model: WhisperOpt = None,
    whisper_device: WhisperDeviceOpt = None,
    lang: LangOpt = None,
    action: ActionOpt = None,
    include_titles: TitlesOpt = None,
    extras: ExtrasOpt = None,
    review: ReviewOpt = None,
    assign: AssignOpt = None,
    yes: YesOpt = False,
    json_out: JsonOpt = False,
) -> None:
    """Run the full pipeline and print the rename plan (always a dry run)."""
    require_ffprobe()
    settings = _settings()
    request = _plan_request(
        settings, rip_dir, output, show, year, tmdb_id, season, disc, disc_episodes, order,
        action, include_titles, extras, review, verify, assign,
    )  # fmt: skip
    verifier = _build_verifier(
        settings,
        request.verify,
        llm,
        llm_model,
        whisper_model,
        whisper_device,
        lang,
        quiet=json_out,
    )
    interactive = not yes and not json_out and _stdin_is_terminal()
    run = _run_pipeline(request, interactive=interactive, verifier=verifier)
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
    order: OrderOpt = None,
    output: OutputOpt = None,
    verify: VerifyOpt = None,
    llm: LLMOpt = None,
    llm_model: LLMModelOpt = None,
    whisper_model: WhisperOpt = None,
    whisper_device: WhisperDeviceOpt = None,
    lang: LangOpt = None,
    action: ActionOpt = None,
    include_titles: TitlesOpt = None,
    extras: ExtrasOpt = None,
    review: ReviewOpt = None,
    assign: AssignOpt = None,
    do_apply: Annotated[
        bool, typer.Option("--apply", help="Actually move files. Without it, this is a dry run.")
    ] = False,
    plex_scan: Annotated[
        bool | None,
        typer.Option(
            "--plex-scan/--no-plex-scan",
            help="Ask Plex to scan the show folder afterwards. \\[default: plex.scan_after_apply]",
        ),
    ] = None,
    yes: YesOpt = False,
    json_out: JsonOpt = False,
) -> None:
    """Build the plan, then execute it if --apply is given."""
    require_ffprobe()
    settings = _settings()
    request = _plan_request(
        settings, rip_dir, output, show, year, tmdb_id, season, disc, disc_episodes, order,
        action, include_titles, extras, review, verify, assign,
    )  # fmt: skip
    if do_apply and json_out and not yes:
        raise _fail("--json with --apply needs --yes (there's no prompt in JSON mode)")
    verifier = _build_verifier(
        settings,
        request.verify,
        llm,
        llm_model,
        whisper_model,
        whisper_device,
        lang,
        quiet=json_out,
    )
    terminal = _stdin_is_terminal()
    run = _run_pipeline(
        request, interactive=not yes and not json_out and terminal, verifier=verifier
    )
    console = Console()
    if not json_out:
        render_plan(run, console, show_evidence=not do_apply, dry_run=not do_apply)
    if run.plan.errors or not do_apply:
        if json_out:
            typer.echo(plan_to_json(run))
        elif not run.plan.errors:
            console.print("Dry run only. Re-run with [bold]--apply[/bold] to execute this plan.")
        raise typer.Exit(code=1 if run.plan.errors else 0)

    count = sum(1 for i in run.plan.items if i.action is not PlanAction.SKIP)
    if not yes:
        if not terminal:
            raise _fail("refusing to apply without confirmation; pass --yes")
        if not typer.confirm(f"Apply {count} file operation(s) under {request.output_root}?"):
            raise typer.Exit(code=1)
    opts = request.options
    args: dict[str, str | int | float | bool | None] = {
        "show": show, "tmdb_id": run.show.tmdb_id, "season": run.season.season,
        "order": request.order, "disc": disc, "disc_episodes": disc_episodes,
        "action": opts.action.value, "extras": opts.extras.value,
        "review": opts.review.value, "include_titles": opts.include_titles,
        "assign": ", ".join(assign) if assign else None,
    }  # fmt: skip
    try:
        manifest, manifest_path = apply_plan(
            run.plan, request.rip_dir, default_manifest_dir(), args=args
        )
    except ApplyError as exc:
        raise _fail(str(exc)) from exc
    failed = any(e.status is EntryStatus.FAILED for e in manifest.entries)
    plex_note = None
    if _pick(plex_scan, settings.plex.scan_after_apply) and not failed:
        plex_note = _plex_scan(manifest, settings)
    if json_out:
        payload = {
            **plan_payload(run),
            "manifest_path": str(manifest_path),
            "manifest": manifest.model_dump(mode="json"),
            "plex": plex_note,
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        render_manifest(manifest, console, title="Applied")
        # Plain echo so long paths aren't wrapped mid-word (they get copy-pasted).
        typer.echo(f"Manifest: {manifest_path}")
        typer.echo(f"Undo with: disc2plex undo '{manifest_path}'")
        if plex_note:
            typer.echo(f"Plex: {plex_note}")
    if failed:
        raise typer.Exit(code=1)


def _library_folders(manifest: Manifest) -> list[Path]:
    """Show folders that received files in this run (review folders excluded)."""
    folders: set[Path] = set()
    for e in manifest.entries:
        if e.status is not EntryStatus.DONE:
            continue
        top = e.destination.relative_to(manifest.output_root).parts[0]
        if top not in (REVIEW_DIR, EXTRAS_REVIEW_DIR):
            folders.add(manifest.output_root / top)
    return sorted(folders)


def _plex_client(settings: Settings) -> PlexClient:
    load_dotenv()
    url = settings.plex.url or os.environ.get("PLEX_URL")
    if not url:
        raise PlexError(f"set plex.url in {config_path()} (or PLEX_URL)")
    return PlexClient(url, os.environ.get("PLEX_TOKEN", ""))


def _plex_scan(manifest: Manifest, settings: Settings) -> str:
    """Ask Plex to scan each show folder that got files. Returns a one-line outcome.

    Never fails the command: the files are already in place, and a scan can be
    started from Plex itself.
    """
    folders = _library_folders(manifest)
    if not folders:
        return "nothing new in the library; no scan requested"
    library_path = settings.plex.library_path
    root = manifest.output_root
    try:
        with _plex_client(settings) as plex:
            section = settings.plex.section
            if section is None:
                found = find_section(plex.sections(), server_path(root, root, library_path))
                if found is None:
                    raise PlexError(
                        f"no Plex TV library contains {server_path(root, root, library_path)}; "
                        "set plex.section (see `disc2plex plex sections`) or plex.library_path"
                    )
                section = found.key
            for folder in folders:
                plex.refresh(section, server_path(folder, root, library_path))
    except PlexError as exc:
        typer.secho(f"Warning: Plex scan not started: {exc}", err=True, fg=typer.colors.YELLOW)
        return f"scan not started ({exc})"
    return f"asked library {section} to scan {', '.join(f.name for f in folders)}"


@app.command()
def undo(
    manifest: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Manifest JSON from an apply run.")
    ],
    yes: YesOpt = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be reversed without doing it.")
    ] = False,
    json_out: JsonOpt = False,
) -> None:
    """Reverse a previous apply using its manifest."""
    try:
        loaded = read_manifest(manifest)
    except ApplyError as exc:
        raise _fail(str(exc)) from exc
    console = Console()
    todo = [e for e in loaded.entries if e.status is EntryStatus.DONE]

    def emit_json(m: Manifest, *, undone: bool) -> None:
        payload = {"undone": undone, "pending": len(todo), "manifest": m.model_dump(mode="json")}
        typer.echo(json.dumps(payload, indent=2))

    if not json_out:
        render_manifest(loaded, console, title=f"Run {loaded.run_id} ({loaded.created_at})")
    if not todo or dry_run:
        if json_out:
            emit_json(loaded, undone=False)
        elif not todo:
            console.print("Nothing to undo.")
        else:
            console.print(f"Dry run: {len(todo)} operation(s) would be reversed.")
        return
    if not yes:
        if json_out or not _stdin_is_terminal():
            raise _fail("refusing to undo without confirmation; pass --yes")
        if not typer.confirm(f"Reverse {len(todo)} operation(s)?"):
            raise typer.Exit(code=1)
    result = undo_manifest(loaded, manifest)
    incomplete = any(e.status is EntryStatus.DONE for e in result.entries)
    if json_out:
        emit_json(result, undone=True)
    else:
        render_manifest(result, console, title="Undo result")
        if incomplete:
            console.print(
                "[yellow]Some operations could not be reversed; see notes above.[/yellow]"
            )
    if incomplete:
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    rip_dir: RipDir,
    file: Annotated[
        list[str] | None,
        typer.Option("--file", "-f", help="Only these file names (repeatable)."),
    ] = None,
    all_files: Annotated[
        bool, typer.Option("--all", help="Include Play-All, extras, and junk titles.")
    ] = False,
    whisper_model: WhisperOpt = None,
    whisper_device: WhisperDeviceOpt = None,
    lang: LangOpt = None,
    opening: Annotated[
        bool, typer.Option("--opening", help="Also transcribe a slice near the start.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Re-transcribe even if cached.")
    ] = False,
    json_out: JsonOpt = False,
) -> None:
    """Transcribe a slice of each episode's audio with local Whisper (debugging aid)."""
    require_ffprobe()
    settings = _settings()
    whisper_model = _pick(whisper_model, settings.matching.whisper_model)
    device = _pick(whisper_device, WhisperDevice(settings.matching.whisper_device))
    lang = _pick(lang, settings.matching.lang)
    if whisper_model not in WHISPER_MODELS:
        raise _fail(f"--whisper-model must be one of {', '.join(WHISPER_MODELS)}")
    load_dotenv()  # HF_TOKEN avoids anonymous rate limits on the first model download
    scan_result = scan_directory(rip_dir, cache=ProbeCache(default_cache_path()))
    classified = classify(scan_result.files)
    wanted = set(file or ())
    targets = [
        item.file
        for item in classified.items
        if (item.file.path.name in wanted if wanted else True)
        and (all_files or wanted or item.file_class in (FileClass.EPISODE, FileClass.UNKNOWN))
    ]
    if wanted - {t.path.name for t in targets}:
        raise _fail(
            f"not found in {rip_dir}: {', '.join(sorted(wanted - {t.path.name for t in targets}))}"
        )
    transcriber = FasterWhisperTranscriber(whisper_model, device=device.value)
    cache = None if no_cache else TranscriptCache(default_transcript_cache_dir())
    results: list[FileTranscript] = []
    for media in targets:
        if not json_out:
            typer.secho(f"Transcribing {media.path.name}...", err=True, dim=True)
        try:
            results.append(
                transcribe_file(
                    media,
                    transcriber,
                    lang=lang,
                    slices=choose_slices(media.duration_seconds, include_opening=opening),
                    cache=cache,
                )
            )
        except TranscriptionError as exc:
            typer.secho(f"Skipped {media.path.name}: {exc}", err=True, fg=typer.colors.YELLOW)
    if json_out:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
    else:
        render_transcripts(results, Console())


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
    order: TmdbOrderOpt = "aired",
) -> None:
    """List a season's episodes in aired, DVD, or episode-group order."""
    with tmdb_session() as client:
        show = client.get_show(tmdb_id)
        render_season(season_episodes(client, show, season, order), Console())


@config_app.command("path")
def config_path_cmd() -> None:
    """Print where the config file is read from."""
    path = config_path()
    typer.echo(f"{path}{'' if path.exists() else '  (not created yet)'}")


@config_app.command("show")
def config_show() -> None:
    """Print the effective settings (config file over built-in defaults)."""
    settings = _settings()
    path = config_path()
    typer.echo(f"# from {path}" if path.exists() else f"# {path} not found: built-in defaults")
    for section, values in settings.model_dump(mode="json").items():
        typer.echo(f"[{section}]")
        for key, value in values.items():
            typer.echo(f"{key} = {json.dumps(value)}")


@config_app.command("init")
def config_init() -> None:
    """Write a commented config template (never overwrites an existing file)."""
    path = config_path()
    if path.exists():
        raise _fail(f"{path} already exists; edit it directly")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    typer.echo(f"Wrote {path}")


@plex_app.command("sections")
def plex_sections() -> None:
    """List Plex libraries and their IDs (for plex.section in the config)."""
    settings = _settings()
    try:
        with _plex_client(settings) as plex:
            render_plex_sections(plex.sections(), Console())
    except PlexError as exc:
        raise _fail(str(exc)) from exc
