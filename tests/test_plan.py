"""Tests for plan building (stage 7): destinations, review routing, collisions."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_media, make_show
from disc2plex.models import (
    Assignment,
    AssignmentStatus,
    EpisodeInfo,
    FileClass,
    MatchResult,
    PlanAction,
    PlanItem,
)
from disc2plex.plan import ExtrasMode, PlanOptions, build_plan

ROOT = Path("/library/TV Shows")
SHOW = make_show("Star Trek: The Next Generation", 1987, 655)


def ep(n: int, title: str = "", season: int = 3) -> EpisodeInfo:
    return EpisodeInfo(season=season, episode=n, title=title or f"Episode {n}")


def assignment(
    name: str,
    file_class: FileClass = FileClass.EPISODE,
    episodes: tuple[EpisodeInfo, ...] = (),
    status: AssignmentStatus = AssignmentStatus.CONFIDENT,
    confidence: float = 0.95,
) -> Assignment:
    return Assignment(
        file=make_media(name, 44),
        file_class=file_class,
        episodes=episodes,
        status=status
        if episodes or file_class is not FileClass.EPISODE
        else AssignmentStatus.UNMATCHED,
        confidence=confidence,
    )


def plan_for(*assignments: Assignment, **kwargs: object) -> dict[str, PlanItem]:
    plan = build_plan(MatchResult(assignments=assignments), SHOW, ROOT, "STTNG_S3_D1", **kwargs)  # type: ignore[arg-type]
    return {i.source.name: i for i in plan.items}


SHOW_DIR = ROOT / "Star Trek - The Next Generation (1987) {tmdb-655}"


def test_confident_episode_goes_to_library() -> None:
    items = plan_for(assignment("t00.mkv", episodes=(ep(26, "The Best of Both Worlds: Part I"),)))
    item = items["t00.mkv"]
    assert item.action is PlanAction.MOVE
    assert item.destination == (
        SHOW_DIR
        / "Season 03"
        / "Star Trek - The Next Generation (1987) - s03e26 - The Best of Both Worlds - Part I.mkv"
    )
    assert item.episode_label == "s03e26"


def test_multi_episode_file_name() -> None:
    items = plan_for(assignment("t00.mkv", episodes=(ep(1, "A"), ep(2, "B"))))
    assert items["t00.mkv"].destination is not None
    assert items["t00.mkv"].destination.name.endswith("- s03e01-e02 - A & B.mkv")


def test_non_consecutive_multi_episode_goes_to_review() -> None:
    # DVD order can pair episodes that aren't adjacent in aired numbering.
    items = plan_for(assignment("t00.mkv", episodes=(ep(11), ep(1))))
    item = items["t00.mkv"]
    assert item.destination == ROOT / "_review" / "STTNG_S3_D1" / "t00.mkv"
    assert "not consecutive" in item.reason


def test_options_titles_and_tmdb_id() -> None:
    items = plan_for(
        assignment("t00.mkv", episodes=(ep(5, "Title"),)),
        options=PlanOptions(include_titles=False, include_tmdb_id=False, action=PlanAction.COPY),
    )
    item = items["t00.mkv"]
    assert item.action is PlanAction.COPY
    assert item.destination == (
        ROOT
        / "Star Trek - The Next Generation (1987)"
        / "Season 03"
        / "Star Trek - The Next Generation (1987) - s03e05.mkv"
    )


def test_specials_go_to_season_00() -> None:
    items = plan_for(assignment("t00.mkv", episodes=(ep(1, "Special", season=0),)))
    assert items["t00.mkv"].destination is not None
    assert items["t00.mkv"].destination.parent.name == "Season 00"


@pytest.mark.parametrize(
    ("status", "confidence", "reason"),
    [
        (AssignmentStatus.NEEDS_REVIEW, 0.6, "needs review: best guess s03e05 (60%)"),
        (AssignmentStatus.UNMATCHED, 0.2, "unmatched episode"),
    ],
)
def test_non_confident_episodes_go_to_review(
    status: AssignmentStatus, confidence: float, reason: str
) -> None:
    items = plan_for(assignment("t00.mkv", episodes=(ep(5),), status=status, confidence=confidence))
    item = items["t00.mkv"]
    assert item.destination == ROOT / "_review" / "STTNG_S3_D1" / "t00.mkv"
    assert item.reason == reason


def test_unknown_goes_to_review() -> None:
    items = plan_for(assignment("t09.mkv", FileClass.UNKNOWN))
    assert items["t09.mkv"].destination == ROOT / "_review" / "STTNG_S3_D1" / "t09.mkv"


def test_play_all_and_junk_are_skipped() -> None:
    items = plan_for(
        assignment("t00.mkv", FileClass.PLAY_ALL), assignment("t08.mkv", FileClass.JUNK)
    )
    for name in ("t00.mkv", "t08.mkv"):
        assert items[name].action is PlanAction.SKIP
        assert items[name].destination is None


@pytest.mark.parametrize(
    ("mode", "destination", "action"),
    [
        (ExtrasMode.REVIEW, ROOT / "_extras_review" / "STTNG_S3_D1" / "t05.mkv", PlanAction.MOVE),
        (ExtrasMode.SEASON0, ROOT / "_extras_review" / "STTNG_S3_D1" / "t05.mkv", PlanAction.MOVE),
        (ExtrasMode.SKIP, None, PlanAction.SKIP),
    ],
)
def test_extras_modes(mode: ExtrasMode, destination: Path | None, action: PlanAction) -> None:
    items = plan_for(assignment("t05.mkv", FileClass.EXTRA), options=PlanOptions(extras=mode))
    assert items["t05.mkv"].destination == destination
    assert items["t05.mkv"].action is action


def test_collision_is_an_error() -> None:
    plan = build_plan(
        MatchResult(
            assignments=(
                assignment("t00.mkv", episodes=(ep(5),)),
                assignment("t01.mkv", episodes=(ep(5),)),
            )
        ),
        SHOW,
        ROOT,
        "disc",
    )
    assert len(plan.errors) == 1
    assert "t00.mkv" in plan.errors[0] and "t01.mkv" in plan.errors[0]


def test_existing_destination_is_skipped() -> None:
    target = (
        SHOW_DIR / "Season 03" / "Star Trek - The Next Generation (1987) - s03e05 - Episode 5.mkv"
    )
    plan = build_plan(
        MatchResult(assignments=(assignment("t00.mkv", episodes=(ep(5),)),)),
        SHOW,
        ROOT,
        "disc",
        exists=lambda p: p == target,
    )
    assert plan.items[0].action is PlanAction.SKIP
    assert "destination exists" in plan.items[0].reason
    assert any("already exists" in w for w in plan.warnings)


def test_weird_disc_name_is_sanitized_and_contained() -> None:
    plan = build_plan(
        MatchResult(assignments=(assignment("t09.mkv", FileClass.UNKNOWN),)),
        SHOW,
        ROOT,
        "../../etc",
    )
    destination = plan.items[0].destination
    assert destination is not None
    assert destination.is_relative_to(ROOT / "_review")


def test_match_notes_become_warnings() -> None:
    plan = build_plan(MatchResult(notes=("episode 3 missing",)), SHOW, ROOT, "disc")
    assert plan.warnings == ("episode 3 missing",)
