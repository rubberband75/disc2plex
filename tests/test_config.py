"""Tests for .env parsing and loading."""

from pathlib import Path

from disc2plex.config import load_dotenv, parse_dotenv


def test_parse_dotenv() -> None:
    text = """
# comment
TMDB_API_TOKEN=abc123
export ANTHROPIC_API_KEY="sk-quoted value"
SINGLE='single # not a comment'
TRAILING=value # comment
EMPTY=
not a pair
"""
    assert parse_dotenv(text) == {
        "TMDB_API_TOKEN": "abc123",
        "ANTHROPIC_API_KEY": "sk-quoted value",
        "SINGLE": "single # not a comment",
        "TRAILING": "value",
        "EMPTY": "",
    }


def test_load_dotenv_does_not_override_existing(tmp_path: Path) -> None:
    first = tmp_path / "a.env"
    second = tmp_path / "b.env"
    first.write_text("A=from_first\nB=from_first\n")
    second.write_text("B=from_second\nC=from_second\n")
    env = {"A": "preset"}
    load_dotenv([first, second, tmp_path / "missing.env"], environ=env)
    assert env == {"A": "preset", "B": "from_first", "C": "from_second"}
