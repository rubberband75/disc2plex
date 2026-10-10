"""Tests for heuristic matching (stage 4) and decision/conflicts (stage 6)."""

from __future__ import annotations

import pytest

from conftest import make_media, make_season
from disc2plex.classify import classify
from disc2plex.match.combine import Thresholds, decide
from disc2plex.match.heuristic import match_heuristic
from disc2plex.models import (
    Assignment,
    AssignmentStatus,
    EpisodeInfo,
    FileClass,
    MatchResult,
    MediaFile,
    SeasonEpisodes,
)
from disc2plex.tmdb import expected_runtime_seconds, season_runtimes_seconds

CONFIDENT = AssignmentStatus.CONFIDENT
REVIEW = AssignmentStatus.NEEDS_REVIEW
UNMATCHED = AssignmentStatus.UNMATCHED


def run(
    files: list[MediaFile],
    season: SeasonEpisodes,
    **kwargs: object,
) -> MatchResult:
    classified = classify(
        files,
        expected_runtime_seconds=expected_runtime_seconds(season),
        season_runtimes_seconds=season_runtimes_seconds(season),
    )
    return decide(match_heuristic(classified.items, season, **kwargs))  # type: ignore[arg-type]


def by_name(result: MatchResult) -> dict[str, Assignment]:
    return {a.file.path.name: a for a in result.assignments}


def labels(result: MatchResult) -> dict[str, str]:
    return {name: a.episode_label for name, a in by_name(result).items() if a.episodes}


def statuses(result: MatchResult) -> dict[str, AssignmentStatus]:
    return {name: a.status for name, a in by_name(result).items()}


# --- whole season / pinned starts ---------------------------------------------


def test_whole_season_on_one_disc_is_confident() -> None:
    season = make_season([44, 43, 45, 44])
    files = [make_media(f"t{i:02d}.mkv", m) for i, m in enumerate([43.5, 42.8, 44.6, 43.9])]
    result = run(files, season)
    assert labels(result) == {
        "t00.mkv": "s01e01",
        "t01.mkv": "s01e02",
        "t02.mkv": "s01e03",
        "t03.mkv": "s01e04",
    }
    assert set(statuses(result).values()) == {CONFIDENT}
    assert result.start_position == 1


def test_disc_range_pins_start() -> None:
    season = make_season([44] * 12)
    files = [make_media(f"t{i:02d}.mkv", 43.5 + i * 0.1) for i in range(4)]
    result = run(files, season, disc_range=(5, 8))
    assert list(labels(result).values()) == ["s01e05", "s01e06", "s01e07", "s01e08"]
    assert set(statuses(result).values()) == {CONFIDENT}
    assert result.start_source == "range"


def test_disc_one_pins_start() -> None:
    season = make_season([44] * 12)
    files = [make_media(f"t{i:02d}.mkv", 43.5) for i in range(4)]
    result = run(files, season, disc_number=1)
    assert list(labels(result).values()) == ["s01e01", "s01e02", "s01e03", "s01e04"]
    assert set(statuses(result).values()) == {CONFIDENT}


def test_uniform_runtimes_without_hint_need_review() -> None:
    season = make_season([44] * 12)
    files = [make_media(f"t{i:02d}.mkv", 43.5 + i * 0.1) for i in range(4)]
    result = run(files, season, disc_number=2)
    assert set(statuses(result).values()) == {REVIEW}
    assert any("ambiguous" in n for n in result.notes)


def test_distinctive_runtimes_infer_start() -> None:
    # A 22-min run inside a 44-min season pins disc 2's position.
    season = make_season([44, 44, 44, 44, 22, 23, 21, 44, 44, 44])
    files = [make_media(f"t{i:02d}.mkv", m) for i, m in enumerate([21.8, 22.9, 21.1])]
    result = run(files, season)
    assert list(labels(result).values()) == ["s01e05", "s01e06", "s01e07"]
    assert result.start_source == "inferred"
    assert result.start_position == 5


def test_disc_range_outside_season_raises() -> None:
    season = make_season([44] * 4)
    files = [make_media("t00.mkv", 44)]
    with pytest.raises(ValueError, match="outside"):
        run(files, season, disc_range=(3, 6))


def test_disc_running_longer_than_tmdb_is_calibrated() -> None:
    # Modelled on a real Blu-ray: TMDB says 22 min, the disc's episodes run 22.5-26.5.
    season = make_season([23, 22, 22, 22, 22, 22, 22, 22])
    files = [
        make_media(f"t{i:02d}.mkv", m) for i, m in enumerate([42.7, 24.9, 22.6, 26.3, 26.4, 24.3])
    ]
    result = run(files, season, disc_number=1)
    assert labels(result)["t00.mkv"] == "s01e01-e02"
    assert labels(result)["t05.mkv"] == "s01e07"
    # Single episodes are confident; the merged pilot (shorter than 2x, as
    # two-episode files drop a set of credits) is labelled right but may need review.
    assert {statuses(result)[f"t{i:02d}.mkv"] for i in range(1, 6)} == {CONFIDENT}
    assert statuses(result)["t00.mkv"] is not UNMATCHED
    assert any("runtimes were scaled" in n for n in result.notes)
    singles = [a for a in result.assignments if a.file.path.name != "t00.mkv"]
    assert not any("mismatch" in r for a in singles for r in a.reasons)


# --- double-length / two-part -------------------------------------------------


def test_double_file_takes_two_episodes() -> None:
    season = make_season([44, 44, 44, 44, 44])
    files = [
        # Two episodes in one file; chapters (every 10 min) rule out a Play-All.
        make_media("t00.mkv", 87.5, chapters=[0, 10, 20, 30, 40, 50, 60, 70, 80]),
        make_media("t01.mkv", 43.8),
        make_media("t02.mkv", 44.2),
        make_media("t03.mkv", 43.1),
    ]
    result = run(files, season)
    assert labels(result) == {
        "t00.mkv": "s01e01-e02",
        "t01.mkv": "s01e03",
        "t02.mkv": "s01e04",
        "t03.mkv": "s01e05",
    }
    assert set(statuses(result).values()) == {CONFIDENT}


def test_double_file_without_chapters_is_left_for_review() -> None:
    # Same length as two episodes, no chapters, sizes don't match exactly:
    # Play-All or two-parter? Don't guess.
    season = make_season([44, 44, 44, 44, 44])
    files = [
        make_media("t00.mkv", 87.5),
        make_media("t01.mkv", 43.8),
        make_media("t02.mkv", 44.2),
    ]
    result = run(files, season)
    assert by_name(result)["t00.mkv"].file_class is FileClass.UNKNOWN


def test_long_pilot_listed_as_one_episode_takes_one() -> None:
    # TMDB lists the 86-minute pilot as a single episode.
    season = make_season([86, 44, 44, 44])
    files = [
        make_media("t00.mkv", 85.2),
        make_media("t01.mkv", 43.8),
        make_media("t02.mkv", 44.2),
        make_media("t03.mkv", 43.1),
    ]
    result = run(files, season)
    assert labels(result)["t00.mkv"] == "s01e01"
    assert labels(result)["t03.mkv"] == "s01e04"
    assert statuses(result)["t00.mkv"] is CONFIDENT


# --- out of order, gaps, mismatches -------------------------------------------


def test_out_of_order_titles_are_never_confident() -> None:
    # The disc puts the 86-min pilot last. Title order can't fit it, so the
    # misfits are left unmatched rather than forced, and nothing is confident.
    season = make_season([86, 44, 44, 44])
    files = [
        make_media("t00.mkv", 43.8),
        make_media("t01.mkv", 44.2),
        make_media("t02.mkv", 43.1),
        make_media("t03.mkv", 85.2),
    ]
    result = run(files, season, disc_number=1)
    assert CONFIDENT not in statuses(result).values()
    pilot = by_name(result)["t03.mkv"]
    assert pilot.status is UNMATCHED
    assert any("out of order" in r for r in pilot.reasons)
    assert result.notes  # the gap at position 1 is reported


def test_missing_episode_detected_as_gap() -> None:
    # Episode 3 (a distinctive 30-min episode) failed to rip.
    season = make_season([44, 44, 30, 44, 44])
    files = [
        make_media("t00.mkv", 44.0),
        make_media("t01.mkv", 43.6),
        make_media("t02.mkv", 43.9),
        make_media("t03.mkv", 44.1),
    ]
    result = run(files, season, disc_range=(1, 5))
    assert labels(result) == {
        "t00.mkv": "s01e01",
        "t01.mkv": "s01e02",
        "t02.mkv": "s01e04",
        "t03.mkv": "s01e05",
    }
    assert any("missing" in n for n in result.notes)
    assert CONFIDENT not in statuses(result).values()


def test_range_with_fewer_files_than_episodes_is_flagged() -> None:
    season = make_season([44] * 8)
    files = [make_media(f"t{i:02d}.mkv", 44.0 + i * 0.1) for i in range(3)]
    result = run(files, season, disc_range=(1, 4))
    assert any("more episodes than matched files" in n for n in result.notes)
    assert CONFIDENT not in statuses(result).values()


def test_more_files_than_episodes_leaves_one_unmatched() -> None:
    season = make_season([44, 44])
    files = [make_media(f"t{i:02d}.mkv", 44.0 + i * 0.2) for i in range(3)]
    result = run(files, season)
    matched = [a for a in result.assignments if a.episodes]
    assert len(matched) == 2
    assert sum(1 for a in result.assignments if a.status is UNMATCHED) == 1


def test_two_files_with_near_identical_durations_get_distinct_episodes() -> None:
    season = make_season([44, 44, 44])
    files = [
        make_media("t00.mkv", 44.0, size=1_500_000_000),
        make_media("t01.mkv", 44.0, size=1_610_000_000),
        make_media("t02.mkv", 43.0),
    ]
    result = run(files, season)
    assert sorted(labels(result).values()) == ["s01e01", "s01e02", "s01e03"]


# --- DVD order ---------------------------------------------------------------


def test_dvd_order_assigns_aired_numbers() -> None:
    # DVD order puts aired episode 11 first.
    season = make_season([86, 44, 44], numbers=[11, 1, 2])
    files = [
        make_media("t00.mkv", 85.0),
        make_media("t01.mkv", 44.0),
        make_media("t02.mkv", 43.5),
    ]
    result = run(files, season, disc_number=1)
    assert labels(result) == {"t00.mkv": "s01e11", "t01.mkv": "s01e01", "t02.mkv": "s01e02"}


# --- non-episode files and runtime-less seasons -------------------------------


def test_non_episode_files_pass_through_unassigned() -> None:
    season = make_season([44, 44])
    files = [
        make_media("t00.mkv", 44.0),
        make_media("t01.mkv", 43.5),
        make_media("t02.mkv", 4.0),  # extra
        make_media("t03.mkv", 0.3),  # junk
    ]
    result = run(files, season)
    a = by_name(result)
    assert a["t02.mkv"].file_class is FileClass.EXTRA
    assert a["t02.mkv"].episodes == ()
    assert a["t03.mkv"].file_class is FileClass.JUNK
    assert a["t02.mkv"].status is UNMATCHED


def test_season_without_runtimes_relies_on_order() -> None:
    season = make_season([None, None, None])
    files = [make_media(f"t{i:02d}.mkv", 44.0 + i) for i in range(3)]
    result = run(files, season, disc_number=1)
    assert list(labels(result).values()) == ["s01e01", "s01e02", "s01e03"]
    a = by_name(result)["t00.mkv"]
    assert "runtime_fit" not in a.signals
    assert any("no runtime" in r for r in a.reasons)


def test_empty_season() -> None:
    season = make_season([])
    result = run([make_media("t00.mkv", 44.0), make_media("t01.mkv", 44.5)], season)
    assert all(a.status is UNMATCHED for a in result.assignments)
    assert any("no episodes" in n for n in result.notes)


def test_every_assignment_has_reasons_and_signals() -> None:
    season = make_season([44, 44, 44])
    files = [make_media(f"t{i:02d}.mkv", 44.0) for i in range(3)]
    for a in run(files, season).assignments:
        assert a.reasons
        assert "order" in a.signals


# --- combine.decide -----------------------------------------------------------


def _assignment(name: str, episode: int | None, confidence: float) -> Assignment:
    eps = (EpisodeInfo(season=1, episode=episode, title="x"),) if episode is not None else ()
    return Assignment(
        file=make_media(name, 44),
        file_class=FileClass.EPISODE,
        episodes=eps,
        confidence=confidence,
    )


def test_decide_thresholds() -> None:
    result = decide(
        MatchResult(
            assignments=(
                _assignment("a.mkv", 1, 0.9),
                _assignment("b.mkv", 2, 0.6),
                _assignment("c.mkv", 3, 0.3),
                _assignment("d.mkv", None, 0.99),
            )
        )
    )
    assert [a.status for a in result.assignments] == [CONFIDENT, REVIEW, UNMATCHED, UNMATCHED]


def test_decide_custom_thresholds() -> None:
    result = decide(
        MatchResult(assignments=(_assignment("a.mkv", 1, 0.7),)),
        Thresholds(confident=0.6, review=0.4),
    )
    assert result.assignments[0].status is CONFIDENT


def test_decide_conflict_demotes_both() -> None:
    result = decide(
        MatchResult(assignments=(_assignment("a.mkv", 1, 0.95), _assignment("b.mkv", 1, 0.9)))
    )
    assert [a.status for a in result.assignments] == [REVIEW, REVIEW]
    assert all(any("conflict" in r for r in a.reasons) for a in result.assignments)
    assert any("more than one file" in n for n in result.notes)
