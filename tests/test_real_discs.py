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
from disc2plex.models import AssignmentStatus, FileClass, ScanResult
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


# --- Parks and Recreation, Season 1, DVD (MakeMKV, one disc) -----------------------
#
# Seven episode-length titles for six episodes: t06 (21:49) and t07 (26:50) are both
# "Rock Show" (broadcast and extended cut; TMDB lists 27 min), and t12 (22:53,
# stereo) is a compilation whose excerpts come from several episodes. Runtimes
# alone can't tell which 22-minute title is the odd one out.

PARKS_ROUTES = {
    "/3/tv/8592": "tv_8592",
    "/3/tv/8592/episode_groups": "tv_8592_episode_groups",
    "/3/tv/8592/season/1": "tv_8592_season_1",
}


def parks_disc(fake_rip: FakeRipBuilder) -> tuple[Path, ProbeFn]:
    data = json.loads((FIXTURES / "real" / "parks_s1_d1.json").read_text())
    return fake_rip({name: (info["probe"], info["size_bytes"]) for name, info in data.items()})


def test_parks_left_over_title_makes_nothing_confident(
    fake_rip: FakeRipBuilder, tmp_path: Path
) -> None:
    # Before the "left over" cap, t02-t04 and t06 were filed one episode early at 87%+.
    rip_dir, probe = parks_disc(fake_rip)
    run = run_plan(
        PlanRequest(rip_dir=rip_dir, output_root=tmp_path / "lib", tmdb_id=8592, season=1),
        scanner=lambda path: scan_directory(path, probe=probe),
        client=FakeTMDB(PARKS_ROUTES).client(),
    )
    assert all(a.status is not AssignmentStatus.CONFIDENT for a in run.match.assignments)
    assert any("left over" in n for n in run.match.notes)


# --- Parks and Recreation, Season 6, DVD disc 1 ------------------------------------
#
# t01 (43:05) is the hour-long premiere "London", one 44-minute episode on TMDB.
# t02-t08 are 21:35 each at nearly the same bitrate, so any two of them add up to
# t01 in length and size; it was once called their Play-All and left behind.
# t09 (20:22, stereo) is not an episode. t10 and t13 (4.4 min) each happen to
# equal two shorter featurettes.


def parks_s6_run(fake_rip: FakeRipBuilder, tmp_path: Path, disc: int | None = None) -> PlanRun:
    data = json.loads((FIXTURES / "real" / "parks_s6_d1.json").read_text())
    rip_dir, probe = fake_rip({n: (i["probe"], i["size_bytes"]) for n, i in data.items()})
    return run_plan(
        PlanRequest(
            rip_dir=rip_dir, output_root=tmp_path / "lib", tmdb_id=8592, season=6, disc=disc
        ),
        scanner=lambda path: scan_directory(path, probe=probe),
        client=FakeTMDB({**PARKS_ROUTES, "/3/tv/8592/season/6": "tv_8592_season_6"}).client(),
    )


def test_parks_s6_hour_long_premiere_is_an_episode_not_a_play_all(
    fake_rip: FakeRipBuilder, tmp_path: Path
) -> None:
    run = parks_s6_run(fake_rip, tmp_path, disc=1)
    classes = {i.file.path.stem[-3:]: i.file_class for i in run.classified.items}
    assert FileClass.PLAY_ALL not in classes.values()
    assert classes["t01"] is FileClass.EPISODE
    result = by_title(run)
    assert result["t01"] == ("s06e01", AssignmentStatus.CONFIDENT)
    for number in range(2, 9):
        assert result[f"t{number:02d}"] == (f"s06e{number:02d}", AssignmentStatus.CONFIDENT)


def test_parks_s6_short_sums_of_featurettes_are_extras(
    fake_rip: FakeRipBuilder, tmp_path: Path
) -> None:
    run = parks_s6_run(fake_rip, tmp_path)
    classes = {i.file.path.stem[-3:]: i.file_class for i in run.classified.items}
    assert classes["t10"] is FileClass.EXTRA
    assert classes["t13"] is FileClass.EXTRA


def test_parks_s6_without_a_hint_runtimes_alone_are_not_confident(
    fake_rip: FakeRipBuilder, tmp_path: Path
) -> None:
    # t01 could be the 44-minute premiere or two 22-minute episodes.
    run = parks_s6_run(fake_rip, tmp_path)
    assert all(a.status is not AssignmentStatus.CONFIDENT for a in run.match.assignments)
