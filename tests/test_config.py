"""Tests for .env loading and the config.toml settings."""

from pathlib import Path

import pytest

from disc2plex.config import (
    CONFIG_TEMPLATE,
    ConfigError,
    Settings,
    config_path,
    load_dotenv,
    load_settings,
    parse_dotenv,
    parse_settings,
)


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


# --- config.toml --------------------------------------------------------------------


def test_template_parses_to_defaults() -> None:
    # Every key in the template is commented out, so it must equal the defaults.
    assert parse_settings(CONFIG_TEMPLATE) == Settings()


def test_template_keys_are_all_valid() -> None:
    # Uncommenting every "# key = value" line must still be a valid config.
    lines = []
    for line in CONFIG_TEMPLATE.splitlines():
        body = line.removeprefix("# ")
        if line.startswith("# ") and "=" in body and not body.startswith(("Key", "Optional")):
            lines.append(body.split("  #", 1)[0])
        elif not line.startswith("#"):
            lines.append(line)
    settings = parse_settings("\n".join(lines))
    assert settings.library.root == Path("/mnt/plex/TV Shows")
    assert settings.plex.section == 2


def test_parse_settings_values() -> None:
    settings = parse_settings(
        """
        [library]
        root = "/srv/tv"
        include_titles = false
        review = "leave"
        [matching]
        llm = "claude"
        confident = 0.9
        """
    )
    assert settings.library.root == Path("/srv/tv")
    assert not settings.library.include_titles
    assert settings.library.review == "leave"
    assert settings.matching.llm == "claude"
    assert settings.matching.confident == 0.9
    assert settings.matching.review == 0.5  # untouched keys keep defaults


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("[library]\nroot = ", "invalid TOML"),
        ("[library]\nroots = '/x'", "library.roots"),
        ("[matching]\nllm = 'gpt'", "matching.llm"),
        ("[matching]\nwhisper_device = 'opencl'", "matching.whisper_device"),
        ("[matching]\nconfident = 0.5\nreview = 0.6", "lower than matching.confident"),
        ("[nonsense]\n", "nonsense"),
    ],
)
def test_parse_settings_errors(text: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        parse_settings(text)


def test_load_settings_missing_file_is_defaults(tmp_path: Path) -> None:
    assert load_settings(tmp_path / "nope.toml") == Settings()


def test_load_settings_expands_home(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[library]\nroot = "~/TV"\n')
    assert load_settings(path).library.root == Path.home() / "TV"


def test_config_path_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config_path() == tmp_path / "disc2plex" / "config.toml"
    monkeypatch.setenv("DISC2PLEX_CONFIG", str(tmp_path / "other.toml"))
    assert config_path() == tmp_path / "other.toml"
