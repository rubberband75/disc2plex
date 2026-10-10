"""Tests for Stage 2 classification, covering the messy-disc edge cases in CLAUDE.md."""

from __future__ import annotations

import pytest

from conftest import make_media
from disc2plex.classify import classify
from disc2plex.models import Classification, ClassifyResult, FileClass, MediaFile

EPISODE = FileClass.EPISODE
PLAY_ALL = FileClass.PLAY_ALL
EXTRA = FileClass.EXTRA
JUNK = FileClass.JUNK
UNKNOWN = FileClass.UNKNOWN


def classes(result: ClassifyResult) -> dict[str, FileClass]:
    return {item.file.path.name: item.file_class for item in result.items}


def item(result: ClassifyResult, name: str) -> Classification:
    return next(i for i in result.items if i.file.path.name == name)


def four_episodes() -> list[MediaFile]:
    return [
        make_media("t00.mkv", 43.1),
        make_media("t01.mkv", 44.0),
        make_media("t02.mkv", 42.6),
        make_media("t03.mkv", 43.8),
    ]


# --- basics -------------------------------------------------------------------


def test_clean_disc_all_episodes() -> None:
    result = classify(four_episodes())
    assert set(classes(result).values()) == {EPISODE}
    assert result.reference_source == "cluster"
    assert result.reference_seconds == pytest.approx(43.45 * 60)
    assert all(i.episode_count_hint == 1 for i in result.items)
    assert all(i.reasons for i in result.items)


def test_output_preserves_input_order_and_covers_every_file() -> None:
    files = [*four_episodes(), make_media("t04.mkv", 0.2), make_media("t05.mkv", 5)]
    result = classify(files)
    assert [i.file for i in result.items] == files


def test_empty_input() -> None:
    result = classify([])
    assert result.items == ()
    assert result.reference_seconds is None


# --- junk ---------------------------------------------------------------------


def test_junk_short_silent_and_zero_duration() -> None:
    files = [
        *four_episodes(),
        make_media("short.mkv", 0.5),
        make_media("menu.mkv", 10, audio=False),
        make_media("broken.mkv", 0),
    ]
    result = classify(files)
    assert classes(result)["short.mkv"] is JUNK
    assert classes(result)["menu.mkv"] is JUNK
    assert "no audio" in item(result, "menu.mkv").reasons[0]
    assert classes(result)["broken.mkv"] is JUNK


def test_exact_duplicate_is_junk() -> None:
    a = make_media("t00.mkv", 44.0, size=1_500_000_000, chapters=[0, 10, 20, 30])
    dup = make_media("t05.mkv", 44.0, size=1_500_100_000, chapters=[0, 10, 20, 30])
    result = classify([a, make_media("t01.mkv", 43.5), dup])
    assert classes(result)["t00.mkv"] is EPISODE
    assert classes(result)["t05.mkv"] is JUNK
    assert item(result, "t05.mkv").duplicate_of == a.path


def test_near_identical_duration_different_size_is_not_duplicate() -> None:
    a = make_media("t00.mkv", 44.0, size=1_500_000_000)
    b = make_media("t01.mkv", 44.0, size=1_620_000_000)
    result = classify([a, b])
    assert classes(result) == {"t00.mkv": EPISODE, "t01.mkv": EPISODE}


def test_same_length_and_size_but_different_chapters_is_not_duplicate() -> None:
    a = make_media("t00.mkv", 44.0, size=1_500_000_000, chapters=[0, 9, 21, 33])
    b = make_media("t01.mkv", 44.0, size=1_500_000_000, chapters=[0, 12, 25, 31])
    result = classify([a, b])
    assert classes(result) == {"t00.mkv": EPISODE, "t01.mkv": EPISODE}


# --- Play-All -----------------------------------------------------------------


def test_play_all_confirmed_by_size() -> None:
    eps = four_episodes()
    total_minutes = sum(e.duration_seconds for e in eps) / 60
    play_all = make_media("t04.mkv", total_minutes, size=sum(e.size_bytes for e in eps))
    result = classify([*eps, play_all])

    assert classes(result)["t04.mkv"] is PLAY_ALL
    assert set(item(result, "t04.mkv").play_all_of) == {e.path for e in eps}
    assert all(classes(result)[e.path.name] is EPISODE for e in eps)
    # The Play-All must not drag the reference length up.
    assert result.reference_seconds == pytest.approx(43.45 * 60)


def test_play_all_listed_first_out_of_order_confirmed_by_chapters() -> None:
    # Titles out of order: the Play-All is t00 and plays t03, t01, t04, t02.
    eps = {
        "t01.mkv": 42.0,
        "t02.mkv": 45.0,
        "t03.mkv": 43.5,
        "t04.mkv": 44.2,
    }
    order = ["t03.mkv", "t01.mkv", "t04.mkv", "t02.mkv"]
    boundaries, offset = [0.0], 0.0
    for name in order[:-1]:
        offset += eps[name]
        boundaries.append(offset)
    # Real Play-Alls drift a few seconds per episode and have extra chapters.
    chapters = sorted({*(b + 0.03 for b in boundaries), 10.0, 60.0, 100.0})
    play_all = make_media("t00.mkv", sum(eps.values()) + 0.1, size=1, chapters=chapters)
    files = [play_all, *(make_media(n, m) for n, m in eps.items())]
    result = classify(files)
    assert classes(result)["t00.mkv"] is PLAY_ALL
    assert any("chapter" in r for r in item(result, "t00.mkv").reasons)


def test_play_all_without_confirmation_is_unknown() -> None:
    eps = four_episodes()
    total_minutes = sum(e.duration_seconds for e in eps) / 60
    suspect = make_media("t04.mkv", total_minutes, size=10)  # no chapters, size mismatch
    result = classify([*eps, suspect])
    assert classes(result)["t04.mkv"] is UNKNOWN
    assert "possible Play-All" in " ".join(item(result, "t04.mkv").reasons)
    assert item(result, "t04.mkv").play_all_of  # suspected components still reported
    assert all(classes(result)[e.path.name] is EPISODE for e in eps)


def test_play_all_of_subset_of_titles() -> None:
    eps = four_episodes()
    first_two = eps[:2]
    play_all = make_media(
        "t04.mkv",
        sum(e.duration_seconds for e in first_two) / 60,
        size=sum(e.size_bytes for e in first_two),
    )
    result = classify([*eps, play_all])
    assert classes(result)["t04.mkv"] is PLAY_ALL
    assert set(item(result, "t04.mkv").play_all_of) == {e.path for e in first_two}


# --- double-length / two-part episodes -----------------------------------------


def test_double_length_episode_with_no_matching_pair() -> None:
    files = [
        make_media("t00.mkv", 80.0),  # 1.78x; no two episodes sum to ~80
        make_media("t01.mkv", 44.0),
        make_media("t02.mkv", 45.0),
        make_media("t03.mkv", 46.0),
    ]
    result = classify(files)
    double = item(result, "t00.mkv")
    assert double.file_class is EPISODE
    assert double.episode_count_hint == 2
    assert "two-part" in double.reasons[-1]


def test_double_length_whose_chapters_rule_out_play_all() -> None:
    # 42+45 and 43+44 both equal 87 min, but chapters every 10 minutes miss
    # every possible boundary, so it's a genuine long episode.
    files = [
        make_media("t00.mkv", 42.0),
        make_media("t01.mkv", 43.0),
        make_media("t02.mkv", 44.0),
        make_media("t03.mkv", 45.0),
        make_media("t04.mkv", 87.0, size=5, chapters=[0, 10, 20, 30, 40, 50, 60, 70, 80]),
    ]
    result = classify(files)
    double = item(result, "t04.mkv")
    assert double.file_class is EPISODE
    assert double.episode_count_hint == 2
    assert any("chapters don't line up" in r for r in double.reasons)


def test_double_length_equal_to_two_episodes_without_chapters_is_unknown() -> None:
    # Could be a two-episode Play-All or a two-parter; don't guess.
    files = [*four_episodes(), make_media("t04.mkv", 43.1 + 44.0, size=5)]
    result = classify(files)
    assert classes(result)["t04.mkv"] is UNKNOWN


def test_extended_pilot_between_bands_is_unknown() -> None:
    result = classify([*four_episodes(), make_media("t04.mkv", 62.0)])
    assert classes(result)["t04.mkv"] is UNKNOWN
    assert "longer" in item(result, "t04.mkv").reasons[-1]


# --- extras -------------------------------------------------------------------


def test_extras_and_in_between_lengths() -> None:
    files = [
        *four_episodes(),
        make_media("featurette.mkv", 5.0),
        make_media("documentary.mkv", 12.0),
        make_media("half.mkv", 30.0),
    ]
    result = classify(files)
    assert classes(result)["featurette.mkv"] is EXTRA
    assert classes(result)["documentary.mkv"] is EXTRA
    assert classes(result)["half.mkv"] is UNKNOWN


def test_many_similar_featurettes_do_not_become_the_episode_cluster() -> None:
    files = [
        make_media("t00.mkv", 44.0),
        make_media("t01.mkv", 43.0),
        *(make_media(f"t{i:02d}.mkv", 3.0 + i * 0.05) for i in range(2, 8)),
    ]
    result = classify(files)
    assert result.reference_seconds == pytest.approx(43.5 * 60)
    assert [classes(result)[f"t{i:02d}.mkv"] for i in range(2, 8)] == [EXTRA] * 6


# --- reference length and expected runtime ------------------------------------


def test_single_file_without_expected_runtime_is_unknown() -> None:
    result = classify([make_media("t00.mkv", 44.0), make_media("t01.mkv", 2.0)])
    assert classes(result) == {"t00.mkv": UNKNOWN, "t01.mkv": EXTRA}
    assert result.reference_seconds is None
    assert any("could not infer episode length" in n for n in result.notes)


def test_single_file_with_expected_runtime() -> None:
    result = classify([make_media("t00.mkv", 44.0)], expected_runtime_seconds=45 * 60)
    assert classes(result)["t00.mkv"] is EPISODE
    assert result.reference_source == "expected"


def test_observed_cluster_preferred_over_rounded_expected_runtime() -> None:
    result = classify(four_episodes(), expected_runtime_seconds=45 * 60)
    assert result.reference_source == "cluster"
    assert result.reference_seconds == pytest.approx(43.45 * 60)


def test_expected_runtime_picks_the_right_cluster() -> None:
    # Two pairs of similar titles: 22-min episodes and 44-min "double" files.
    files = [
        make_media("t00.mkv", 22.0),
        make_media("t01.mkv", 22.5),
        make_media("t02.mkv", 44.0),
        make_media("t03.mkv", 46.0),
    ]
    result = classify(files, expected_runtime_seconds=22 * 60)
    assert result.reference_seconds == pytest.approx(22.25 * 60)
    assert classes(result)["t00.mkv"] is EPISODE
    assert item(result, "t03.mkv").episode_count_hint == 2


def test_short_episode_show_with_expected_runtime() -> None:
    # 7-minute cartoons would look like extras without context.
    files = [make_media(f"t{i:02d}.mkv", 7.0 + i * 0.1) for i in range(6)]
    without = classify(files)
    assert EPISODE not in classes(without).values()

    with_expected = classify(files, expected_runtime_seconds=7 * 60)
    assert set(classes(with_expected).values()) == {EPISODE}


def test_expected_runtime_far_from_every_cluster_adds_a_note() -> None:
    result = classify(four_episodes(), expected_runtime_seconds=22 * 60)
    assert result.reference_source == "expected"
    assert any("check the show/season" in n for n in result.notes)


# --- a realistic messy disc ---------------------------------------------------


def test_messy_disc() -> None:
    eps = [
        make_media("DISC1_t01.mkv", 43.2, chapters=[0, 8, 19, 31]),
        make_media("DISC1_t02.mkv", 44.1, chapters=[0, 11, 23, 35]),
        make_media("DISC1_t03.mkv", 42.8, chapters=[0, 9, 18, 30]),
    ]
    play_all = make_media(
        "DISC1_t00.mkv",
        sum(e.duration_seconds for e in eps) / 60,
        size=sum(e.size_bytes for e in eps) + 2_000_000,
    )
    dup = make_media("DISC1_t04.mkv", 43.2, size=eps[0].size_bytes, chapters=[0, 8, 19, 31])
    files = [
        play_all,
        *eps,
        dup,
        make_media("DISC1_t05.mkv", 6.5),  # featurette
        make_media("DISC1_t06.mkv", 0.4),  # logo
        make_media("DISC1_t07.mkv", 3.0, audio=False),  # menu
    ]
    result = classify(files)
    assert classes(result) == {
        "DISC1_t00.mkv": PLAY_ALL,
        "DISC1_t01.mkv": EPISODE,
        "DISC1_t02.mkv": EPISODE,
        "DISC1_t03.mkv": EPISODE,
        "DISC1_t04.mkv": JUNK,
        "DISC1_t05.mkv": EXTRA,
        "DISC1_t06.mkv": JUNK,
        "DISC1_t07.mkv": JUNK,
    }
