"""Regression tests built from real disc metadata (ffprobe + TMDB JSON only, no video).

The Good Place, Season 1, Blu-ray disc 1 (MakeMKV, title 5 missing). Facts this
disc taught us:

- episodes run ~5-20% longer than TMDB's 22-minute runtimes (needs calibration);
- t00 (42:43) holds episodes 1+2, shorter than 2x because one set of credits is dropped;
- TMDB's "Bluray Release" episode group merges e01+e02 into its first entry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import FIXTURES, FakeRipBuilder, FakeTMDB
from disc2plex.models import AssignmentStatus, ScanResult
from disc2plex.pipeline import PlanRequest, PlanRun, run_plan
from disc2plex.scan import ProbeFn, scan_directory

GOOD_PLACE_ROUTES = {
    "/3/tv/66573": "tv_66573",
    "/3/tv/66573/episode_groups": "tv_66573_episode_groups",
    "/3/tv/66573/season/1": "tv_66573_season_1",
    "/3/tv/episode_group/6772fc80cfe62644df133b27": "episode_group_66573_bluray",
}


def good_place_disc(fake_rip: FakeRipBuilder) -> tuple[Path, ProbeFn]:
    data = json.loads((FIXTURES / "real" / "good_place_s1_d1.json").read_text())
    return fake_rip({name: (info["probe"], info["size_bytes"]) for name, info in data.items()})


def plan(
    fake_rip: FakeRipBuilder, tmp_path: Path, *, disc: int | None = None, order: str = "aired"
) -> PlanRun:
    rip_dir, probe = good_place_disc(fake_rip)

    def scanner(path: Path) -> ScanResult:
        return scan_directory(path, probe=probe)

    return run_plan(
        PlanRequest(
            rip_dir=rip_dir,
            output_root=tmp_path / "library",
            tmdb_id=66573,
            season=1,
            disc=disc,
            order=order,
        ),
        scanner=scanner,
        client=FakeTMDB(GOOD_PLACE_ROUTES).client(),
    )


def by_title(run: PlanRun) -> dict[str, tuple[str, AssignmentStatus]]:
    return {a.file.path.stem[-3:]: (a.episode_label, a.status) for a in run.match.assignments}


def test_disc1_aired_order(fake_rip: FakeRipBuilder, tmp_path: Path) -> None:
    result = by_title(plan(fake_rip, tmp_path, disc=1))
    assert result["t00"][0] == "s01e01-e02"
    assert result["t00"][1] is not AssignmentStatus.UNMATCHED
    assert result["t01"] == ("s01e03", AssignmentStatus.CONFIDENT)
    assert result["t02"] == ("s01e04", AssignmentStatus.CONFIDENT)
    assert result["t03"] == ("s01e05", AssignmentStatus.CONFIDENT)
    assert result["t04"] == ("s01e06", AssignmentStatus.CONFIDENT)
    assert result["t06"] == ("s01e07", AssignmentStatus.CONFIDENT)


def test_disc1_runtime_calibration_noted(fake_rip: FakeRipBuilder, tmp_path: Path) -> None:
    run = plan(fake_rip, tmp_path, disc=1)
    assert any("runtimes were scaled" in w for w in run.plan.warnings)


def test_without_disc_hint_nothing_is_confident(fake_rip: FakeRipBuilder, tmp_path: Path) -> None:
    # Uniform 22-minute runtimes give no way to place the disc in the season.
    run = plan(fake_rip, tmp_path)
    assert all(a.status is not AssignmentStatus.CONFIDENT for a in run.match.assignments)
    assert any("ambiguous" in w for w in run.plan.warnings)


@pytest.mark.parametrize("disc", [1, None])
def test_bluray_group_order_never_confidently_wrong(
    fake_rip: FakeRipBuilder, tmp_path: Path, disc: int | None
) -> None:
    # In the Blu-ray group the first entry is s01e01 alone (23 min) though the
    # disc's t00 holds e01+e02. Whatever we decide, never place t00 confidently
    # as anything that isn't e01-e02.
    run = plan(fake_rip, tmp_path, disc=disc, order="dvd")
    label, status = by_title(run)["t00"]
    assert status is not AssignmentStatus.CONFIDENT or label == "s01e01-e02"
