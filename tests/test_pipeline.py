"""End-to-end dry-run tests: fake rip folder + fake TMDB -> plan. No network, no real media."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from conftest import FAKE_TOKEN, FakeRipBuilder, FakeTMDB, synthetic_probe
from disc2plex import cli
from disc2plex.models import AssignmentStatus, PlanAction, ScanResult
from disc2plex.pipeline import PlanRequest, infer_season, parse_disc_range, run_plan
from disc2plex.scan import ProbeFn, scan_directory
from disc2plex.tmdb import AmbiguousShowError, TMDBClient, TMDBError

SHOW_DIR = "Firefly (2002) {tmdb-1437}"
MINUTE = 60.0

# Firefly disc 1 in DVD order: Serenity (86 min, aired #11), The Train Job (#1),
# Bushwhacked (#2), Shindig (#6); plus a Play-All, a featurette, and a logo.
EPISODE_MINUTES = {"t01": 85.4, "t02": 43.6, "t03": 43.9, "t04": 43.2}


def firefly_disc1(fake_rip: FakeRipBuilder) -> tuple[Path, ProbeFn]:
    specs: dict[str, object] = {}
    sizes = {}
    for index, (stem, minutes) in enumerate(EPISODE_MINUTES.items()):
        size = int(minutes * MINUTE * 1_000_000 * (1 + index * 0.07))
        sizes[stem] = size
        # Chapters every ~9 minutes, as on a real disc.
        chapters = [c * 9 * MINUTE for c in range(int(minutes // 9) + 1)]
        specs[f"FIREFLY_D1_{stem}.mkv"] = (
            synthetic_probe(minutes * MINUTE, chapters=chapters),
            size,
        )
    total = sum(EPISODE_MINUTES.values())
    specs["FIREFLY_D1_t00.mkv"] = (synthetic_probe(total * MINUTE), sum(sizes.values()))
    specs["FIREFLY_D1_t05.mkv"] = synthetic_probe(6.5 * MINUTE)
    specs["FIREFLY_D1_t06.mkv"] = synthetic_probe(12)
    return fake_rip(specs)  # type: ignore[arg-type]


def scanner_for(probe: ProbeFn) -> Callable[[Path], ScanResult]:
    def scanner(rip_dir: Path) -> ScanResult:
        return scan_directory(rip_dir, probe=probe)

    return scanner


# --- helpers ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"), [("1-4", (1, 4)), ("5", (5, 5)), (" 2 - 3 ", (2, 3))]
)
def test_parse_disc_range(text: str, expected: tuple[int, int]) -> None:
    assert parse_disc_range(text) == expected


@pytest.mark.parametrize("text", ["", "a-b", "4-2", "0-3", "1-2-3"])
def test_parse_disc_range_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_disc_range(text)


def test_infer_season(fake: FakeTMDB) -> None:
    show = fake.client().get_show(1437)
    assert infer_season(show) == 1  # specials don't count
    two = show.model_copy(
        update={"seasons": (*show.seasons, show.seasons[1].model_copy(update={"number": 2}))}
    )
    with pytest.raises(TMDBError, match="--season"):
        infer_season(two)


# --- pipeline -----------------------------------------------------------------


def test_firefly_disc1_dvd_order(fake_rip: FakeRipBuilder, fake: FakeTMDB, tmp_path: Path) -> None:
    rip_dir, probe = firefly_disc1(fake_rip)
    output = tmp_path / "library"
    run = run_plan(
        PlanRequest(rip_dir=rip_dir, output_root=output, show_name="Firefly", disc=1, order="dvd"),
        scanner=scanner_for(probe),
        client=fake.client(),
    )
    items = {i.source.name: i for i in run.plan.items}
    season_dir = output / SHOW_DIR / "Season 01"

    assert (
        items["FIREFLY_D1_t01.mkv"].destination
        == season_dir / "Firefly (2002) - s01e11 - Serenity.mkv"
    )
    assert (
        items["FIREFLY_D1_t02.mkv"].destination
        == season_dir / "Firefly (2002) - s01e01 - The Train Job.mkv"
    )
    assert (
        items["FIREFLY_D1_t03.mkv"].destination
        == season_dir / "Firefly (2002) - s01e02 - Bushwhacked.mkv"
    )
    assert (
        items["FIREFLY_D1_t04.mkv"].destination
        == season_dir / "Firefly (2002) - s01e06 - Shindig.mkv"
    )
    for stem in EPISODE_MINUTES:
        assert items[f"FIREFLY_D1_{stem}.mkv"].status is AssignmentStatus.CONFIDENT

    assert items["FIREFLY_D1_t00.mkv"].action is PlanAction.SKIP  # Play-All
    assert (
        items["FIREFLY_D1_t05.mkv"].destination
        == output / "_extras_review" / "rip" / "FIREFLY_D1_t05.mkv"
    )
    assert items["FIREFLY_D1_t06.mkv"].action is PlanAction.SKIP  # logo
    assert run.plan.errors == ()


def test_firefly_aired_order_without_hint_needs_review(
    fake_rip: FakeRipBuilder, fake: FakeTMDB, tmp_path: Path
) -> None:
    # In aired order the 86-min episode is #11; the disc's title order doesn't fit
    # any contiguous aired run, so nothing should be confidently placed.
    rip_dir, probe = firefly_disc1(fake_rip)
    run = run_plan(
        PlanRequest(rip_dir=rip_dir, output_root=tmp_path / "lib", show_name="Firefly"),
        scanner=scanner_for(probe),
        client=fake.client(),
    )
    episodes = [a for a in run.match.assignments if a.file.path.stem[-3:] in EPISODE_MINUTES]
    assert all(a.status is not AssignmentStatus.CONFIDENT for a in episodes)


def test_existing_destination_skipped(
    fake_rip: FakeRipBuilder, fake: FakeTMDB, tmp_path: Path
) -> None:
    rip_dir, probe = firefly_disc1(fake_rip)
    output = tmp_path / "library"
    taken = output / SHOW_DIR / "Season 01" / "Firefly (2002) - s01e11 - Serenity.mkv"
    run = run_plan(
        PlanRequest(rip_dir=rip_dir, output_root=output, tmdb_id=1437, disc=1, order="dvd"),
        scanner=scanner_for(probe),
        client=fake.client(),
        exists=lambda p: p == taken,
    )
    serenity = next(i for i in run.plan.items if i.source.name == "FIREFLY_D1_t01.mkv")
    assert serenity.action is PlanAction.SKIP


def test_ambiguous_show_raises(fake_rip: FakeRipBuilder, fake: FakeTMDB, tmp_path: Path) -> None:
    rip_dir, probe = firefly_disc1(fake_rip)
    with pytest.raises(AmbiguousShowError):
        run_plan(
            PlanRequest(rip_dir=rip_dir, output_root=tmp_path, show_name="Doctor Who"),
            scanner=scanner_for(probe),
            client=fake.client(),
        )


def test_pipeline_never_touches_files(
    fake_rip: FakeRipBuilder, fake: FakeTMDB, tmp_path: Path
) -> None:
    rip_dir, probe = firefly_disc1(fake_rip)
    before = sorted(p.name for p in rip_dir.iterdir())
    output = tmp_path / "library"
    run_plan(
        PlanRequest(rip_dir=rip_dir, output_root=output, show_name="Firefly", disc=1, order="dvd"),
        scanner=scanner_for(probe),
        client=fake.client(),
    )
    assert sorted(p.name for p in rip_dir.iterdir()) == before
    assert not output.exists()


# --- CLI ----------------------------------------------------------------------

runner = CliRunner(env={"COLUMNS": "250"})


@pytest.fixture
def cli_disc(
    fake_rip: FakeRipBuilder, fake: FakeTMDB, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    rip_dir, probe = firefly_disc1(fake_rip)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("TMDB_API_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli, "scan_directory", functools.partial(scan_directory, probe=probe))
    real_client = cli.TMDBClient

    def client_factory(token: str, **kwargs: object) -> TMDBClient:
        return real_client(token, transport=httpx.MockTransport(fake.handler), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "TMDBClient", client_factory)
    return rip_dir


def test_cli_plan_table(cli_disc: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["plan", str(cli_disc), "--show", "Firefly", "--disc", "1", "--order", "dvd",
         "--output", str(tmp_path / "lib")],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "s01e11 - Serenity.mkv" in result.output
    assert "Evidence" in result.output
    assert not (tmp_path / "lib").exists()


def test_cli_plan_json(cli_disc: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["plan", str(cli_disc), "--tmdb-id", "1437", "--disc-episodes", "1-4", "--order", "dvd",
         "--output", str(tmp_path / "lib"), "--json"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["show"]["tmdb_id"] == 1437
    assert data["start_source"] == "range"
    names = {Path(i["source"]).name: i for i in data["plan"]["items"]}
    assert names["FIREFLY_D1_t01.mkv"]["episode_label"] == "s01e11"


def test_cli_plan_requires_output(cli_disc: Path) -> None:
    result = runner.invoke(cli.app, ["plan", str(cli_disc), "--show", "Firefly"])
    assert result.exit_code == 1
    assert "--output is required" in result.output


def test_cli_plan_requires_show_or_id(cli_disc: Path, tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["plan", str(cli_disc), "--output", str(tmp_path)])
    assert result.exit_code == 1
    assert "--show" in result.output


def test_cli_plan_bad_range(cli_disc: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["plan", str(cli_disc), "--show", "Firefly", "--disc-episodes", "10-20",
         "--output", str(tmp_path)],
    )  # fmt: skip
    assert result.exit_code == 1
    assert "outside" in result.output


def test_cli_plan_ambiguous_noninteractive(cli_disc: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app, ["plan", str(cli_disc), "--show", "Doctor Who", "--output", str(tmp_path), "--yes"]
    )
    assert result.exit_code == 1
    assert "57243" in result.output and "--tmdb-id" in result.output


def test_cli_plan_ambiguous_prompt(
    cli_disc: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeTMDB
) -> None:
    # Pretend stdin is a terminal and answer "2" (an invalid "9" first).
    monkeypatch.setattr(cli, "_stdin_is_terminal", lambda: True)
    fake.routes["/3/tv/121"] = "tv_1437"  # reuse Firefly's data for the chosen ID
    fake.routes["/3/tv/121/episode_groups"] = "tv_1437_episode_groups"
    fake.routes["/3/tv/121/season/1"] = "tv_1437_season_1"
    result = runner.invoke(
        cli.app,
        ["plan", str(cli_disc), "--show", "Doctor Who", "--season", "1", "--output", str(tmp_path)],
        input="9\n2\n",
    )
    assert "Which one?" in result.output
    assert "Enter a number from 1 to 3" in result.output
    assert "/3/tv/121" in fake.paths()


# --- CLI apply / undo (inside tmp_path only) ----------------------------------


def _apply_args(rip_dir: Path, library: Path, *extra: str) -> list[str]:
    return [
        "apply", str(rip_dir), "--show", "Firefly", "--disc", "1", "--order", "dvd",
        "--output", str(library), *extra,
    ]  # fmt: skip


@pytest.fixture
def library(cli_disc: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    path = tmp_path / "library"
    path.mkdir()
    return path


def _snapshot(folder: Path) -> dict[str, int]:
    return {str(p.relative_to(folder)): p.stat().st_size for p in folder.rglob("*") if p.is_file()}


def test_cli_apply_without_flag_is_dry_run(cli_disc: Path, library: Path) -> None:
    before = _snapshot(cli_disc)
    result = runner.invoke(cli.app, _apply_args(cli_disc, library))
    assert result.exit_code == 0, result.output
    assert "Re-run with --apply" in result.output
    assert _snapshot(cli_disc) == before
    assert list(library.iterdir()) == []


def test_cli_apply_then_undo_round_trip(cli_disc: Path, library: Path, tmp_path: Path) -> None:
    before = _snapshot(cli_disc)
    result = runner.invoke(cli.app, _apply_args(cli_disc, library, "--apply", "--yes"))
    assert result.exit_code == 0, result.output
    season = library / SHOW_DIR / "Season 01"
    assert (season / "Firefly (2002) - s01e11 - Serenity.mkv").exists()
    assert (library / "_extras_review" / "rip" / "FIREFLY_D1_t05.mkv").exists()
    assert (library / "_extras_review" / ".plexignore").exists()
    assert (cli_disc / "FIREFLY_D1_t00.mkv").exists()  # Play-All left in place

    manifests = list((tmp_path / "data" / "disc2plex" / "manifests").glob("*.json"))
    assert len(manifests) == 1
    assert "disc2plex undo" in result.output

    dry = runner.invoke(cli.app, ["undo", str(manifests[0]), "--dry-run"])
    assert dry.exit_code == 0 and "would be reversed" in dry.output
    assert (season / "Firefly (2002) - s01e11 - Serenity.mkv").exists()

    undo = runner.invoke(cli.app, ["undo", str(manifests[0]), "--yes"])
    assert undo.exit_code == 0, undo.output
    assert _snapshot(cli_disc) == before
    assert list(library.iterdir()) == []

    again = runner.invoke(cli.app, ["undo", str(manifests[0]), "--yes"])
    assert again.exit_code == 0 and "Nothing to undo" in again.output


def test_cli_apply_requires_confirmation_when_not_a_terminal(cli_disc: Path, library: Path) -> None:
    result = runner.invoke(cli.app, _apply_args(cli_disc, library, "--apply"))
    assert result.exit_code == 1
    assert "pass --yes" in result.output
    assert list(library.iterdir()) == []


@pytest.mark.parametrize(("answer", "moved"), [("n\n", False), ("y\n", True)])
def test_cli_apply_interactive_confirmation(
    cli_disc: Path, library: Path, monkeypatch: pytest.MonkeyPatch, answer: str, moved: bool
) -> None:
    monkeypatch.setattr(cli, "_stdin_is_terminal", lambda: True)
    result = runner.invoke(cli.app, _apply_args(cli_disc, library, "--apply"), input=answer)
    assert "Apply 5 file operation(s)" in result.output
    assert (library / SHOW_DIR).exists() is moved
    assert (result.exit_code == 0) is moved


def test_cli_apply_refuses_missing_output_root(
    cli_disc: Path, library: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        cli.app, _apply_args(cli_disc, tmp_path / "unmounted-share", "--apply", "--yes")
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert not (tmp_path / "unmounted-share").exists()


def test_cli_undo_bad_manifest(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("nope")
    result = runner.invoke(cli.app, ["undo", str(bad), "--yes"])
    assert result.exit_code == 1
    assert "cannot read manifest" in result.output


# --- CLI content-verification options -----------------------------------------


def test_cli_plan_notes_when_verification_is_off(cli_disc: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app, ["plan", str(cli_disc), "--show", "Firefly", "--output", str(tmp_path)]
    )
    assert "content verification is off" in result.output


def test_cli_plan_verify_none_is_silent(cli_disc: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["plan", str(cli_disc), "--show", "Firefly", "--output", str(tmp_path), "--verify", "none"],
    )
    assert "content verification" not in result.output


def test_cli_plan_local_llm_requires_model(
    cli_disc: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISC2PLEX_LLM_MODEL", raising=False)
    result = runner.invoke(
        cli.app,
        ["plan", str(cli_disc), "--show", "Firefly", "--output", str(tmp_path), "--llm", "local"],
    )
    assert result.exit_code == 1
    assert "model name is required" in result.output


def test_cli_plan_rejects_bad_whisper_model(cli_disc: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["plan", str(cli_disc), "--show", "Firefly", "--output", str(tmp_path),
         "--llm", "claude", "--whisper-model", "enormous"],
    )  # fmt: skip
    assert result.exit_code == 1
    assert "--whisper-model" in result.output
