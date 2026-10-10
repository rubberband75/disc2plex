"""Tests for Plex naming and filename sanitization."""

from pathlib import Path

import pytest

from disc2plex.naming import (
    MAX_COMPONENT_BYTES,
    ensure_within,
    episode_code,
    episode_file_name,
    episode_path,
    sanitize_component,
    season_folder_name,
    show_display_name,
    show_folder_name,
)

# --- sanitize_component -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Plain Name", "Plain Name"),
        ("Star Trek: The Next Generation", "Star Trek - The Next Generation"),
        ("Re:Zero", "Re-Zero"),
        ("AC/DC", "AC-DC"),
        ("Back\\Slash", "Back-Slash"),
        ("Pipe|Name", "Pipe-Name"),
        ('Say "Hello"', "Say 'Hello'"),
        ("What If...?", "What If"),
        ("<Tagged> *Star*", "Tagged Star"),
        ("  lots   of\t\nspace  ", "lots of space"),
        ("Trailing dots...", "Trailing dots"),
        ("Tab\x00Null\x1fCtrl", "TabNullCtrl"),
        ("Amélie", "Amélie"),
        ("名探偵コナン", "名探偵コナン"),
    ],
)
def test_sanitize_component(raw: str, expected: str) -> None:
    assert sanitize_component(raw) == expected


def test_sanitize_normalizes_unicode_to_nfc() -> None:
    decomposed = "Amélie"  # 'e' + combining acute, as macOS often stores it
    assert sanitize_component(decomposed) == "Amélie"


@pytest.mark.parametrize("raw", ["", "   ", "...", "..", ".", "???", "<*>", "\x00\x01"])
def test_sanitize_falls_back_when_nothing_left(raw: str) -> None:
    assert sanitize_component(raw) == "_"
    assert sanitize_component(raw, fallback="Unknown") == "Unknown"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("CON", "CON_"), ("nul", "nul_"), ("COM1", "COM1_"), ("aux.mkv", "aux_.mkv")],
)
def test_sanitize_windows_reserved_names(raw: str, expected: str) -> None:
    assert sanitize_component(raw) == expected


def test_sanitize_does_not_flag_names_containing_reserved_words() -> None:
    assert sanitize_component("Constantine") == "Constantine"


def test_sanitize_never_contains_illegal_characters() -> None:
    nasty = 'a<b>c:d"e/f\\g|h?i*j\x00k'
    result = sanitize_component(nasty)
    assert not set('<>:"/\\|?*\x00') & set(result)


def test_sanitize_truncates_on_utf8_boundary() -> None:
    result = sanitize_component("é" * 200)  # 400 bytes
    assert len(result.encode("utf-8")) <= MAX_COMPONENT_BYTES
    result.encode("utf-8").decode("utf-8")  # still valid UTF-8
    assert set(result) == {"é"}


def test_sanitize_truncation_does_not_leave_trailing_space() -> None:
    result = sanitize_component("a" * 9 + " bbbb", max_bytes=10)
    assert result == "a" * 9


# --- show / season folders ----------------------------------------------------


def test_show_display_name() -> None:
    assert show_display_name("Firefly", 2002) == "Firefly (2002)"
    assert show_display_name("Firefly", None) == "Firefly"


def test_show_display_name_does_not_double_year() -> None:
    assert show_display_name("Doctor Who (2005)", 2005) == "Doctor Who (2005)"


def test_show_folder_includes_tmdb_id_by_default() -> None:
    assert show_folder_name("Firefly", 2002, 1437) == "Firefly (2002) {tmdb-1437}"


def test_show_folder_without_tmdb_id() -> None:
    assert show_folder_name("Firefly", 2002, 1437, include_tmdb_id=False) == "Firefly (2002)"
    assert show_folder_name("Firefly", 2002, None) == "Firefly (2002)"


def test_show_folder_sanitizes_name() -> None:
    assert (
        show_folder_name("Star Trek: The Next Generation", 1987, 655)
        == "Star Trek - The Next Generation (1987) {tmdb-655}"
    )


@pytest.mark.parametrize(
    ("season", "expected"),
    [(0, "Season 00"), (1, "Season 01"), (12, "Season 12"), (100, "Season 100")],
)
def test_season_folder_name(season: int, expected: str) -> None:
    assert season_folder_name(season) == expected


def test_season_folder_rejects_negative() -> None:
    with pytest.raises(ValueError):
        season_folder_name(-1)


# --- episode codes ------------------------------------------------------------


@pytest.mark.parametrize(
    ("season", "episodes", "expected"),
    [
        (1, [1], "s01e01"),
        (0, [3], "s00e03"),
        (10, [105], "s10e105"),
        (1, [1, 2], "s01e01-e02"),
        (2, [5, 3, 4], "s02e03-e05"),
    ],
)
def test_episode_code(season: int, episodes: list[int], expected: str) -> None:
    assert episode_code(season, episodes) == expected


@pytest.mark.parametrize("episodes", [[], [1, 3], [2, 2], [-1]])
def test_episode_code_rejects_bad_input(episodes: list[int]) -> None:
    with pytest.raises(ValueError):
        episode_code(1, episodes)


# --- episode file names -------------------------------------------------------


def test_episode_file_name_with_title() -> None:
    assert (
        episode_file_name("Firefly", 2002, 1, [1], ["Serenity"])
        == "Firefly (2002) - s01e01 - Serenity.mkv"
    )


def test_episode_file_name_without_title() -> None:
    name = episode_file_name("Firefly", 2002, 1, [1], ["Serenity"], include_title=False)
    assert name == "Firefly (2002) - s01e01.mkv"


def test_episode_file_name_missing_title_omits_separator() -> None:
    assert episode_file_name("Firefly", 2002, 1, [2]) == "Firefly (2002) - s01e02.mkv"
    assert episode_file_name("Firefly", 2002, 1, [2], ["   "]) == "Firefly (2002) - s01e02.mkv"


def test_episode_file_name_multi_episode_joins_titles() -> None:
    name = episode_file_name("Show", 2000, 1, [1, 2], ["Pilot Part 1", "Pilot Part 2"])
    assert name == "Show (2000) - s01e01-e02 - Pilot Part 1 & Pilot Part 2.mkv"


def test_episode_file_name_multi_episode_dedupes_identical_titles() -> None:
    name = episode_file_name("Show", 2000, 1, [1, 2], ["Pilot", "Pilot"])
    assert name == "Show (2000) - s01e01-e02 - Pilot.mkv"


def test_episode_file_name_special() -> None:
    name = episode_file_name("Firefly", 2002, 0, [1], ["Here's How It Was"])
    assert name == "Firefly (2002) - s00e01 - Here's How It Was.mkv"


def test_episode_file_name_sanitizes_show_and_title() -> None:
    name = episode_file_name(
        "Star Trek: The Next Generation", 1987, 3, [26], ["The Best of Both Worlds: Part I?"]
    )
    assert name == (
        "Star Trek - The Next Generation (1987) - s03e26 - The Best of Both Worlds - Part I.mkv"
    )


def test_episode_file_name_normalizes_extension() -> None:
    assert episode_file_name("Show", 2000, 1, [1], ext="MKV") == "Show (2000) - s01e01.mkv"


def test_episode_file_name_truncates_only_the_title() -> None:
    name = episode_file_name("Show", 2000, 1, [1], ["x" * 500])
    assert len(name.encode("utf-8")) <= MAX_COMPONENT_BYTES
    assert name.startswith("Show (2000) - s01e01 - xxx")
    assert name.endswith(".mkv")


def test_episode_file_name_too_long_without_title_raises() -> None:
    with pytest.raises(ValueError, match="too long"):
        episode_file_name("S" * 300, 2000, 1, [1])


# --- full paths and traversal guard ------------------------------------------


def test_episode_path_layout(tmp_path: Path) -> None:
    path = episode_path(tmp_path, "Firefly", 2002, 1437, 1, [1], ["Serenity"])
    assert path == (
        tmp_path
        / "Firefly (2002) {tmdb-1437}"
        / "Season 01"
        / "Firefly (2002) - s01e01 - Serenity.mkv"
    )


@pytest.mark.parametrize("evil", ["..", "../../etc", "/etc/passwd", "..\\..\\windows"])
def test_episode_path_cannot_escape_root(tmp_path: Path, evil: str) -> None:
    path = episode_path(tmp_path, evil, None, None, 1, [1], [evil])
    assert ensure_within(tmp_path, path) == path
    assert len(path.relative_to(tmp_path).parts) == 3


def test_ensure_within_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ensure_within(tmp_path, tmp_path / ".." / "elsewhere.mkv")
    with pytest.raises(ValueError):
        ensure_within(tmp_path / "lib", tmp_path / "library" / "x.mkv")  # prefix, not child
    with pytest.raises(ValueError):
        ensure_within(tmp_path, tmp_path)


def test_ensure_within_accepts_child(tmp_path: Path) -> None:
    child = tmp_path / "Show" / "Season 01" / "x.mkv"
    assert ensure_within(tmp_path, child) == child
