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
from disc2plex.pipeline import (
    PlanRequest,
    infer_season,
    parse_assign,
    parse_disc_range,
    run_plan,
)
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


# --- --assign (user overrides) ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("t04=6", ("t04", (6,))),
        ("t00=1-2", ("t00", (1, 2))),
        ("t00=e01-e02", ("t00", (1, 2))),
        ("My = Show_t01.mkv=3", ("My = Show_t01.mkv", (3,))),
    ],
)
def test_parse_assign(text: str, expected: tuple[str, tuple[int, ...]]) -> None:
    assert parse_assign(text) == expected


@pytest.mark.parametrize("text", ["t04", "=6", "t04=", "t04=x", "t04=3-1"])
def test_parse_assign_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_assign(text)


def _run_with(
    fake_rip: FakeRipBuilder, fake: FakeTMDB, tmp_path: Path, *assign: str
) -> dict[str, tuple[str, AssignmentStatus]]:
    rip_dir, probe = firefly_disc1(fake_rip)
    run = run_plan(
        PlanRequest(
            rip_dir=rip_dir, output_root=tmp_path / "lib", tmdb_id=1437, disc=1, order="dvd",
            assign=tuple(parse_assign(a) for a in assign),
        ),
        scanner=scanner_for(probe),
        client=fake.client(),
    )  # fmt: skip
    return {a.file.path.stem[-3:]: (a.episode_label, a.status) for a in run.match.assignments}


def test_assign_overrides_and_demotes_the_other_claimant(
    fake_rip: FakeRipBuilder, fake: FakeTMDB, tmp_path: Path
) -> None:
    # Normally t04 -> e06 (Shindig). Say t03 is e06 instead.
    result = _run_with(fake_rip, fake, tmp_path, "t03=6")
    assert result["t03"] == ("s01e06", AssignmentStatus.CONFIDENT)
    assert result["t04"] == ("s01e06", AssignmentStatus.NEEDS_REVIEW)  # conflict, not a winner
    assert result["t02"] == ("s01e01", AssignmentStatus.CONFIDENT)  # untouched


def test_assign_can_promote_a_non_episode(
    fake_rip: FakeRipBuilder, fake: FakeTMDB, tmp_path: Path
) -> None:
    # The 6.5-minute "featurette" (t05) is really the unaired pilot-ish short, say s01e14.
    result = _run_with(fake_rip, fake, tmp_path, "FIREFLY_D1_t05.mkv=14")
    assert result["t05"] == ("s01e14", AssignmentStatus.CONFIDENT)


@pytest.mark.parametrize(
    ("assign", "match"),
    [
        (("t09=1",), "no file on this disc"),
        (("t02=99",), "has no episode 99"),
        (("t02=1", "t03=1"), "episode 1 is given to both"),
        (("t02=1", "FIREFLY_D1_t02=2"), "assigned twice"),
    ],
)
def test_assign_errors(
    fake_rip: FakeRipBuilder, fake: FakeTMDB, tmp_path: Path, assign: tuple[str, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _run_with(fake_rip, fake, tmp_path, *assign)


# --- config file, review flow, JSON, Plex (CLI) -----------------------------------


def _write_config(tmp_path: Path, text: str) -> None:
    # cli_disc points XDG_CONFIG_HOME at tmp_path / "config".
    path = tmp_path / "config" / "disc2plex" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cli_config_supplies_output_and_options(cli_disc: Path, tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        f'[library]\nroot = "{tmp_path / "lib"}"\ninclude_titles = false\n'
        '[matching]\norder = "dvd"\nverify = "none"\n',
    )
    result = runner.invoke(cli.app, ["plan", str(cli_disc), "--show", "Firefly", "--disc", "1"])
    assert result.exit_code == 0, result.output
    assert f"Output root: {tmp_path / 'lib'}" in result.output
    assert "Firefly (2002) - s01e11.mkv" in result.output  # dvd order, no titles
    # Flags beat the config.
    result = runner.invoke(
        cli.app, ["plan", str(cli_disc), "--show", "Firefly", "--disc", "1", "--include-titles"]
    )
    assert "s01e11 - Serenity.mkv" in result.output


def test_cli_bad_config_is_a_clear_error(cli_disc: Path, tmp_path: Path) -> None:
    _write_config(tmp_path, "[library]\nroots = '/x'\n")
    result = runner.invoke(cli.app, ["plan", str(cli_disc), "--show", "Firefly"])
    assert result.exit_code == 1
    assert "library.roots" in result.output


def test_cli_config_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "disc2plex" / "config.toml"
    assert "not created yet" in runner.invoke(cli.app, ["config", "path"]).output
    assert "built-in defaults" in runner.invoke(cli.app, ["config", "show"]).output
    assert runner.invoke(cli.app, ["config", "init"]).exit_code == 0
    assert path.read_text().startswith("# disc2plex configuration")
    again = runner.invoke(cli.app, ["config", "init"])
    assert again.exit_code == 1 and "already exists" in again.output
    path.write_text('[library]\nroot = "/srv/tv"\n')
    shown = runner.invoke(cli.app, ["config", "show"]).output
    assert 'root = "/srv/tv"' in shown and 'llm = "none"' in shown


def test_cli_apply_header_and_summary(cli_disc: Path, library: Path) -> None:
    result = runner.invoke(cli.app, _apply_args(cli_disc, library, "--apply", "--yes"))
    assert result.exit_code == 0, result.output
    assert "Dry run" not in result.output  # it isn't one
    assert "Plan: 4 to library, 1 to extras review, 2 left in place" in result.output


def test_cli_review_leave(cli_disc: Path, library: Path) -> None:
    # Aired order without a hint: every episode needs review, so with --review leave
    # only the extra moves.
    args = ["apply", str(cli_disc), "--show", "Firefly", "--output", str(library),
            "--review", "leave", "--apply", "--yes"]  # fmt: skip
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert not (library / "_review").exists()
    assert (cli_disc / "FIREFLY_D1_t02.mkv").exists()
    assert (library / "_extras_review" / "rip" / "FIREFLY_D1_t05.mkv").exists()


def test_cli_assign_from_review_folder(cli_disc: Path, library: Path) -> None:
    # Review flow: uncertain files land in _review/<disc>/; after checking them,
    # re-run apply on that folder with --assign to file them properly.
    first = runner.invoke(
        cli.app,
        ["apply", str(cli_disc), "--show", "Firefly", "--output", str(library), "--apply", "--yes"],
    )
    assert first.exit_code == 0, first.output
    review = library / "_review" / "rip"
    assert (review / "FIREFLY_D1_t02.mkv").exists()
    second = runner.invoke(
        cli.app,
        ["apply", str(review), "--show", "Firefly", "--output", str(library),
         "--assign", "t02=1", "--apply", "--yes"],
    )  # fmt: skip
    assert second.exit_code == 0, second.output
    assert (
        library / SHOW_DIR / "Season 01" / "Firefly (2002) - s01e01 - The Train Job.mkv"
    ).exists()
    # The others are already in their review folder: skipped, not moved again.
    assert (review / "FIREFLY_D1_t03.mkv").exists()
    assert "1 to library" in second.output and "left in place" in second.output


def test_cli_assign_bad_value(cli_disc: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["plan", str(cli_disc), "--show", "Firefly", "--output", str(tmp_path), "--assign", "t02"],
    )
    assert result.exit_code == 1 and "--assign" in result.output


def test_cli_apply_json(cli_disc: Path, library: Path) -> None:
    needs_yes = runner.invoke(cli.app, _apply_args(cli_disc, library, "--apply", "--json"))
    assert needs_yes.exit_code == 1 and "--yes" in needs_yes.output
    result = runner.invoke(cli.app, _apply_args(cli_disc, library, "--apply", "--yes", "--json"))
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["summary"].startswith("4 to library")
    assert {e["status"] for e in data["manifest"]["entries"]} == {"done"}
    manifest = data["manifest_path"]

    dry = json.loads(runner.invoke(cli.app, ["undo", manifest, "--dry-run", "--json"]).stdout)
    assert dry == {**dry, "undone": False, "pending": 5}
    undone = runner.invoke(cli.app, ["undo", manifest, "--yes", "--json"])
    data = json.loads(undone.stdout)
    assert data["undone"] is True
    assert {e["status"] for e in data["manifest"]["entries"]} == {"undone"}


def _fake_plex(
    monkeypatch: pytest.MonkeyPatch, sections: list[dict[str, object]]
) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/library/sections":
            return httpx.Response(200, json={"MediaContainer": {"Directory": sections}})
        return httpx.Response(200)

    real = cli.PlexClient
    monkeypatch.setattr(
        cli,
        "PlexClient",
        lambda url, token: real(url, token, transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setenv("PLEX_TOKEN", "plex-secret")
    return seen


def test_cli_apply_triggers_plex_scan(
    cli_disc: Path, library: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(
        tmp_path,
        '[plex]\nurl = "http://plex.local:32400"\nlibrary_path = "/data/TV"\n'
        "scan_after_apply = true\n",
    )
    seen = _fake_plex(
        monkeypatch,
        [{"key": "7", "type": "show", "title": "TV", "Location": [{"path": "/data/TV"}]}],
    )
    result = runner.invoke(cli.app, _apply_args(cli_disc, library, "--apply", "--yes"))
    assert result.exit_code == 0, result.output
    refreshes = [r for r in seen if r.url.path.endswith("/refresh")]
    assert len(refreshes) == 1  # one show folder; review folders aren't scanned
    assert refreshes[0].url.path == "/library/sections/7/refresh"
    assert refreshes[0].url.params["path"] == f"/data/TV/{SHOW_DIR}"
    assert "Plex: asked library 7 to scan" in result.output


def test_cli_plex_scan_failure_is_only_a_warning(
    cli_disc: Path, library: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, '[plex]\nurl = "http://plex.local:32400"\n')
    _fake_plex(monkeypatch, [])  # no library contains the output root
    result = runner.invoke(
        cli.app, _apply_args(cli_disc, library, "--apply", "--yes", "--plex-scan")
    )
    assert result.exit_code == 0, result.output
    assert "Plex scan not started" in result.output and "plex.section" in result.output
    assert (library / SHOW_DIR).exists()


def test_cli_no_plex_scan_by_default(
    cli_disc: Path, library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _fake_plex(monkeypatch, [])
    result = runner.invoke(cli.app, _apply_args(cli_disc, library, "--apply", "--yes"))
    assert result.exit_code == 0 and seen == [] and "Plex" not in result.output


def test_assigning_every_episode_drops_the_ambiguous_start_warning(
    fake_rip: FakeRipBuilder, fake: FakeTMDB, tmp_path: Path
) -> None:
    rip_dir, probe = firefly_disc1(fake_rip)

    def warnings(*assign: str) -> tuple[str, ...]:
        request = PlanRequest(
            rip_dir=rip_dir, output_root=tmp_path, tmdb_id=1437,
            assign=tuple(parse_assign(a) for a in assign),
        )  # fmt: skip
        return run_plan(request, scanner=scanner_for(probe), client=fake.client()).plan.warnings

    assert any("ambiguous" in w for w in warnings())
    assert any("ambiguous" in w for w in warnings("t02=1"))
    assert not any("ambiguous" in w for w in warnings("t01=11", "t02=1", "t03=2", "t04=6"))
