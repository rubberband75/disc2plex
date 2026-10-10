"""Plex-compliant folder and file names.

Everything here is pure: functions build names and paths but never touch the
filesystem. Target layout (see CLAUDE.md section 3)::

    <root>/Show Name (2005) {tmdb-12345}/Season 01/Show Name (2005) - s01e01 - Title.mkv
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

# Most filesystems (ext4, NTFS, APFS) cap a single name at 255 bytes/units.
# Measuring UTF-8 bytes is the strictest common interpretation.
MAX_COMPONENT_BYTES = 255

# Names Windows refuses regardless of extension (matters for SMB shares).
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
# "Star Trek: Voyager" reads best as "Star Trek - Voyager"; a bare colon
# ("Re:Zero", "10:30") becomes a hyphen.
_COLON_SPACED = re.compile(r"\s*:\s+")
_WHITESPACE = re.compile(r"\s+")
_LAYOUT_WHITESPACE = re.compile(r"[\t\n\r\v\f]")
_YEAR_SUFFIX = re.compile(r"\(\d{4}\)$")

# Characters illegal on Windows/macOS/Linux, mapped to safe stand-ins.
_REPLACEMENTS = str.maketrans(
    {
        ":": "-",
        "/": "-",
        "\\": "-",
        "|": "-",
        '"': "'",
        "<": "",
        ">": "",
        "?": "",
        "*": "",
    }
)


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Cut ``text`` to at most ``max_bytes`` UTF-8 bytes without splitting a character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _clean_edges(text: str) -> str:
    """Trim whitespace on both ends and dots on the right (illegal on Windows)."""
    return text.strip().rstrip(". ").strip()


def sanitize_component(
    text: str, *, fallback: str = "_", max_bytes: int = MAX_COMPONENT_BYTES
) -> str:
    """Make ``text`` safe to use as a single path component on any common OS.

    Replaces characters illegal on Windows/macOS/Linux, removes control
    characters, collapses whitespace, trims trailing dots/spaces, avoids
    Windows reserved device names, and enforces a byte-length limit.
    Returns ``fallback`` if nothing usable is left (e.g. ``".."``), which also
    makes path traversal through a component impossible.
    """
    result = unicodedata.normalize("NFC", text)
    # Tabs/newlines are whitespace, not junk: turn them into spaces before
    # stripping the remaining control characters.
    result = _LAYOUT_WHITESPACE.sub(" ", result)
    result = _CONTROL_CHARS.sub("", result)
    result = _COLON_SPACED.sub(" - ", result)
    result = result.translate(_REPLACEMENTS)
    result = _WHITESPACE.sub(" ", result)
    result = _clean_edges(result)

    # Windows reserves "CON" even with an extension ("CON.mkv"), so mark the stem.
    stem, dot, rest = result.partition(".")
    if stem.strip().upper() in _WINDOWS_RESERVED:
        result = f"{stem.strip()}_{dot}{rest}"

    result = _clean_edges(_truncate_utf8(result, max_bytes))
    return result or fallback


def show_display_name(name: str, year: int | None) -> str:
    """Return ``"Name (Year)"``, without doubling a year already in the name."""
    name = name.strip()
    if year is None or _YEAR_SUFFIX.search(name):
        return name
    return f"{name} ({year})"


def show_folder_name(
    name: str, year: int | None, tmdb_id: int | None = None, *, include_tmdb_id: bool = True
) -> str:
    """Build the show folder name, e.g. ``"Show Name (2005) {tmdb-12345}"``.

    The ``{tmdb-ID}`` tag makes Plex match the exact show, so it is on by default.
    """
    folder = show_display_name(name, year)
    if include_tmdb_id and tmdb_id is not None:
        folder = f"{folder} {{tmdb-{tmdb_id}}}"
    return sanitize_component(folder)


def season_folder_name(season: int) -> str:
    """Build ``"Season 01"``. Season 0 (``"Season 00"``) holds specials."""
    if season < 0:
        raise ValueError(f"season must be >= 0, got {season}")
    return f"Season {season:02d}"


def episode_code(season: int, episodes: Sequence[int]) -> str:
    """Build ``"s01e01"`` or, for a multi-episode file, ``"s01e01-e02"``.

    Multi-episode files must cover a contiguous run of episodes; Plex's
    ``eXX-eYY`` range syntax cannot express gaps.
    """
    if season < 0:
        raise ValueError(f"season must be >= 0, got {season}")
    if not episodes:
        raise ValueError("at least one episode number is required")
    eps = sorted(episodes)
    if eps[0] < 0:
        raise ValueError(f"episode numbers must be >= 0, got {eps}")
    if eps != list(range(eps[0], eps[0] + len(eps))):
        raise ValueError(f"episodes must be unique and contiguous, got {list(episodes)}")
    code = f"s{season:02d}e{eps[0]:02d}"
    if len(eps) > 1:
        code += f"-e{eps[-1]:02d}"
    return code


def _join_titles(titles: Sequence[str]) -> str:
    """Join distinct non-empty titles with ``" & "``, preserving order."""
    seen: list[str] = []
    for title in titles:
        title = title.strip()
        if title and title not in seen:
            seen.append(title)
    return " & ".join(seen)


def episode_file_name(
    show_name: str,
    year: int | None,
    season: int,
    episodes: Sequence[int],
    titles: Sequence[str] = (),
    ext: str = ".mkv",
    *,
    include_title: bool = True,
    max_bytes: int = MAX_COMPONENT_BYTES,
) -> str:
    """Build an episode filename like ``"Show (2005) - s01e01 - Title.mkv"``.

    ``titles`` has one entry per episode (multi-episode titles are joined with
    ``" & "``). If the name would exceed ``max_bytes``, only the title part is
    shortened; if even the title-less name is too long, ``ValueError`` is raised.
    """
    if not ext.startswith("."):
        ext = f".{ext}"
    ext = sanitize_component(ext.lower())

    base = f"{show_display_name(show_name, year)} - {episode_code(season, episodes)}"
    base = sanitize_component(base, max_bytes=max_bytes)
    budget = max_bytes - len(ext.encode("utf-8"))
    if len(base.encode("utf-8")) > budget:
        raise ValueError(f"filename too long even without a title: {base!r}")

    title = sanitize_component(_join_titles(titles), fallback="") if include_title else ""
    if title:
        separator = " - "
        room = budget - len(base.encode("utf-8")) - len(separator.encode("utf-8"))
        title = _clean_edges(_truncate_utf8(title, room)) if room > 0 else ""
        if title:
            base = f"{base}{separator}{title}"
    return f"{base}{ext}"


def ensure_within(root: Path, path: Path) -> Path:
    """Return ``path`` if it lies inside ``root``; otherwise raise ``ValueError``.

    Purely lexical (no symlink resolution) so it stays side-effect free;
    ``apply`` should re-check with resolved paths before touching anything.
    """
    root_norm = os.path.normpath(os.path.abspath(root))
    path_norm = os.path.normpath(os.path.abspath(path))
    if os.path.commonpath([root_norm, path_norm]) != root_norm or path_norm == root_norm:
        raise ValueError(f"{path} escapes output root {root}")
    return path


def episode_path(
    library_root: Path,
    show_name: str,
    year: int | None,
    tmdb_id: int | None,
    season: int,
    episodes: Sequence[int],
    titles: Sequence[str] = (),
    ext: str = ".mkv",
    *,
    include_tmdb_id: bool = True,
    include_title: bool = True,
) -> Path:
    """Build the full destination path for an episode file under ``library_root``."""
    path = (
        library_root
        / show_folder_name(show_name, year, tmdb_id, include_tmdb_id=include_tmdb_id)
        / season_folder_name(season)
        / episode_file_name(
            show_name, year, season, episodes, titles, ext, include_title=include_title
        )
    )
    return ensure_within(library_root, path)
