"""Human-readable and JSON output for CLI commands."""

from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from disc2plex.models import MediaFile, ScanResult


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


def render_scan(result: ScanResult, console: Console) -> None:
    """Print a table of probed files plus any probe failures."""
    table = Table(title="Scanned files")
    table.add_column("Title", justify="right")
    table.add_column("File")
    table.add_column("Duration", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Ch", justify="right")
    table.add_column("Audio")
    table.add_column("Subs")
    for media in result.files:
        index = media.makemkv_title_index
        table.add_row(
            "-" if index is None else str(index),
            media.path.name,
            format_duration(media.duration_seconds),
            format_size(media.size_bytes),
            str(media.chapter_count),
            _audio_summary(media),
            _subtitle_summary(media),
        )
    console.print(table)
    if not result.files:
        console.print("[yellow]No .mkv files found.[/yellow]")
    for failure in result.failures:
        console.print(f"[red]Could not probe {failure.path.name}:[/red] {failure.error}")


def scan_to_json(result: ScanResult) -> str:
    """Serialize a scan result for ``--json`` output."""
    return json.dumps(result.model_dump(mode="json"), indent=2)
