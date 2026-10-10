"""Smoke tests for the CLI skeleton."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from disc2plex import __version__, cli
from disc2plex.cli import app

# Wide terminal so Rich does not wrap help text mid-phrase.
runner = CliRunner(env={"COLUMNS": "200"})


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["scan", "--help"],
        ["plan", "--help"],
        ["apply", "--help"],
        ["undo", "--help"],
        ["tmdb", "search", "--help"],
        ["tmdb", "show", "--help"],
        ["tmdb", "season", "--help"],
    ],
)
def test_help(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


def test_apply_help_mentions_dry_run() -> None:
    result = runner.invoke(app, ["apply", "--help"])
    assert "--apply" in result.output
    assert "dry run" in result.output


def test_plan_requires_show(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plan", str(tmp_path)])
    assert result.exit_code != 0


def test_missing_ffprobe_gives_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == 1
    assert "ffprobe not found" in result.output


def test_unimplemented_command_exits_nonzero(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    result = runner.invoke(app, ["undo", str(manifest)])
    assert result.exit_code == 1
    assert "Not implemented" in result.output
