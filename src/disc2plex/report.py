"""Human-readable and JSON output for CLI commands."""

from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from disc2plex.models import (
    ClassifyResult,
    FileClass,
    MediaFile,
    ScanResult,
    SeasonEpisodes,
    ShowInfo,
    ShowSummary,
)


def format_duration(seconds: float) -> str:
    """Format seconds as ``H:MM:SS`` (or ``M:SS`` under an hour)."""
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_size(size_bytes: int) -> str:
    """Format a byte count with binary units, e.g. ``1.4 GiB``."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(size_bytes)
    unit = 0
    while size >= 1024 and unit < len(units) - 1:
        size /= 1024
        unit += 1
    return f"{size_bytes} B" if unit == 0 else f"{size:.1f} {units[unit]}"


def _audio_summary(media: MediaFile) -> str:
    if not media.audio_tracks:
        return "none"
    return ", ".join(f"{t.language} {t.codec} {t.channels}ch" for t in media.audio_tracks)


def _subtitle_summary(media: MediaFile) -> str:
    if not media.subtitle_tracks:
        return "-"
    return ", ".join(sorted({t.language for t in media.subtitle_tracks}))


_CLASS_STYLES = {
    FileClass.EPISODE: "green",
    FileClass.PLAY_ALL: "cyan",
    FileClass.EXTRA: "blue",
    FileClass.JUNK: "dim",
    FileClass.UNKNOWN: "yellow",
}


def render_scan(scan: ScanResult, classified: ClassifyResult, console: Console) -> None:
    """Print a table of probed, classified files plus any probe failures."""
    if classified.reference_seconds is not None:
        source = "from similar titles" if classified.reference_source == "cluster" else "expected"
        console.print(
            f"Typical episode length: {format_duration(classified.reference_seconds)} ({source})"
        )
    for note in classified.notes:
        console.print(f"[yellow]Note:[/yellow] {note}")

    table = Table(title="Scanned files")
    table.add_column("Title", justify="right")
    table.add_column("File")
    table.add_column("Duration", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Ch", justify="right")
    table.add_column("Audio")
    table.add_column("Subs")
    table.add_column("Class")
    table.add_column("Why")
    for item in classified.items:
        media = item.file
        index = media.makemkv_title_index
        style = _CLASS_STYLES[item.file_class]
        label = item.file_class.value
        if item.episode_count_hint > 1:
            label += f" x{item.episode_count_hint}"
        table.add_row(
            "-" if index is None else str(index),
            media.path.name,
            format_duration(media.duration_seconds),
            format_size(media.size_bytes),
            str(media.chapter_count),
            _audio_summary(media),
            _subtitle_summary(media),
            f"[{style}]{label}[/{style}]",
            "; ".join(item.reasons),
        )
    console.print(table)
    if not classified.items:
        console.print("[yellow]No .mkv files found.[/yellow]")
    for failure in scan.failures:
        console.print(f"[red]Could not probe {failure.path.name}:[/red] {failure.error}")


def scan_to_json(scan: ScanResult, classified: ClassifyResult) -> str:
    """Serialize scan + classification for ``--json`` output."""
    payload = classified.model_dump(mode="json")
    payload["failures"] = [f.model_dump(mode="json") for f in scan.failures]
    return json.dumps(payload, indent=2)


def _truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def render_search_results(results: list[ShowSummary], console: Console) -> None:
    """Print TMDB search hits."""
    if not results:
        console.print("[yellow]No shows found.[/yellow]")
        return
    table = Table(title="TMDB search results")
    table.add_column("TMDB ID", justify="right")
    table.add_column("Name")
    table.add_column("Year", justify="right")
    table.add_column("Overview")
    for r in results:
        table.add_row(str(r.tmdb_id), r.name, str(r.year or "?"), _truncate(r.overview, 70))
    console.print(table)


def render_show(show: ShowInfo, console: Console) -> None:
    """Print a show's seasons and episode groups (for choosing --season / --order)."""
    console.print(f"[bold]{show.name}[/bold] ({show.year or '?'})  tmdb-{show.tmdb_id}")
    seasons = Table(title="Seasons")
    seasons.add_column("Season", justify="right")
    seasons.add_column("Name")
    seasons.add_column("Episodes", justify="right")
    for s in show.seasons:
        seasons.add_row(str(s.number), s.name, str(s.episode_count))
    console.print(seasons)
    if not show.episode_groups:
        console.print("No episode groups (only aired order is available).")
        return
    groups = Table(title="Episode groups (use the ID with --order)")
    groups.add_column("ID")
    groups.add_column("Type")
    groups.add_column("Name")
    groups.add_column("Groups", justify="right")
    groups.add_column("Episodes", justify="right")
    for g in show.episode_groups:
        groups.add_row(g.id, g.type_name, g.name, str(g.group_count), str(g.episode_count))
    console.print(groups)


def render_season(season: SeasonEpisodes, console: Console) -> None:
    """Print a season's episodes in the chosen order."""
    table = Table(title=f"{season.show.name}: {season.label}")
    table.add_column("#", justify="right")
    table.add_column("Code")
    table.add_column("Title")
    table.add_column("Runtime", justify="right")
    table.add_column("Aired")
    for position, e in enumerate(season.episodes, start=1):
        table.add_row(
            str(position),
            f"s{e.season:02d}e{e.episode:02d}",
            e.title,
            f"{e.runtime_minutes:g} min" if e.runtime_minutes else "-",
            e.air_date.isoformat() if e.air_date else "-",
        )
    console.print(table)
