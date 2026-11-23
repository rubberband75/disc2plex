"""Tests for content verification: prompts, reply validation, scoring, combining.

The LLM and transcription are faked. The pipeline test replays the answers
Claude actually gave for The Good Place S1 disc 1 (including one wrong answer).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from conftest import FIXTURES, FakeRipBuilder, FakeTMDB, make_media, make_season
from disc2plex.llm.base import LLMError, LLMResponse
from disc2plex.match import content as content_mod
from disc2plex.match.content import (
    NONE_CODE,
    ContentConfig,
    ContentUsage,
    ContentVerifier,
    Evidence,
    ExcerptAnswer,
    FileEvidence,
    VerifyMode,
    answer_schema,
    apply_evidence,
    ask_llm,
    build_user_prompt,
    parse_reply,
)
from disc2plex.match.heuristic import HeuristicConfig, content_score, match_heuristic
from disc2plex.models import (
    Assignment,
    AssignmentStatus,
    EpisodeInfo,
    FileClass,
    FileTranscript,
    MatchResult,
    MediaFile,
    ScanResult,
    SliceTranscript,
)
from disc2plex.pipeline import PlanRequest, run_plan
from disc2plex.scan import scan_directory
from disc2plex.transcribe import SliceSpec

SEASON = make_season([22] * 6)
EP = {e.episode: e for e in SEASON.episodes}


def transcript(*texts: str, path: Path = Path("/rip/t00.mkv")) -> FileTranscript:
    return FileTranscript(
        file=path,
        audio_stream_index=1,
        model="fake",
        slices=tuple(
            SliceTranscript(start_seconds=600 * i, duration_seconds=240, text=t)
            for i, t in enumerate(texts)
        ),
    )


# --- prompt and schema ------------------------------------------------------------


def test_schema_restricts_episode_codes() -> None:
    schema = answer_schema(["s01e01", "s01e02"])
    item = schema["properties"]["excerpts"]["items"]
    assert item["properties"]["episode"]["enum"] == ["s01e01", "s01e02", NONE_CODE]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"excerpt", "episode", "confidence", "reasoning"}


def test_user_prompt_lists_candidates_and_excerpts() -> None:
    season = SEASON.model_copy(
        update={
            "episodes": (
                EpisodeInfo(season=1, episode=1, title="Pilot", overview="Eleanor arrives."),
                EpisodeInfo(season=1, episode=2, title="Flying", overview=""),
            )
        }
    )
    prompt = build_user_prompt(season, transcript("hello there", "general kenobi"), 1500)
    assert 's01e01 "Pilot": Eleanor arrives.' in prompt
    assert "(no synopsis available)" in prompt
    assert "Excerpt 1 (from 0:00 to 4:00):\nhello there" in prompt
    assert "Excerpt 2" in prompt and "general kenobi" in prompt
    assert "25:00 long" in prompt


# --- reply parsing ------------------------------------------------------------


def reply(*items: tuple[int, str, float]) -> str:
    return json.dumps(
        {
            "excerpts": [
                {"excerpt": n, "episode": code, "confidence": c, "reasoning": "because"}
                for n, code, c in items
            ]
        }
    )


def test_parse_reply_valid() -> None:
    answers = parse_reply(reply((1, "s01e03", 0.9)), SEASON, 1)
    assert answers == [ExcerptAnswer(1, EP[3], 0.9, "because")]


def test_parse_reply_normalizes() -> None:
    text = reply((2, "S01E02", 1.7), (1, "none", -0.2), (1, "s01e04", 0.5), (5, "s01e01", 0.9))
    answers = parse_reply(text, SEASON, 2)
    assert [(a.excerpt, a.episode, a.confidence) for a in answers] == [
        (1, None, 0.0),  # "none"; the duplicate excerpt 1 answer is ignored
        (2, EP[2], 1.0),  # case-insensitive code, confidence clamped
    ]


def test_parse_reply_unknown_code_is_no_match() -> None:
    assert parse_reply(reply((1, "s09e99", 0.9)), SEASON, 1)[0].episode is None


@pytest.mark.parametrize("text", ["not json", '{"excerpts": "nope"}', reply((7, "s01e01", 0.9))])
def test_parse_reply_rejects_unusable(text: str) -> None:
    with pytest.raises(ValueError):
        parse_reply(text, SEASON, 1)


class ScriptedLLM:
    """Returns queued replies; records prompts."""

    name = "fake/llm"

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        self.prompts.append(user)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item, model="fake", input_tokens=100, output_tokens=10)


def test_ask_llm_retries_once_then_gives_up() -> None:
    usage = ContentUsage()
    llm = ScriptedLLM("garbage", reply((1, "s01e02", 0.8)))
    answers = ask_llm(llm, SEASON, transcript("x" * 200), 1500, usage)
    assert answers is not None and answers[0].episode == EP[2]
    assert (usage.llm_calls, usage.input_tokens) == (2, 200)

    assert (
        ask_llm(ScriptedLLM("bad", "worse"), SEASON, transcript("x"), 1500, ContentUsage()) is None
    )


# --- content score in the aligner -----------------------------------------------


def test_content_score_signs() -> None:
    cfg = HeuristicConfig()
    votes = [((1, 3), 0.9)]
    assert content_score(votes, [EP[3]], 6, cfg) > 0
    assert content_score(votes, [EP[4]], 6, cfg) < 0
    assert content_score([], [EP[3]], 6, cfg) == 0.0
    # A two-episode file whose excerpts name both of its episodes scores high.
    both = [((1, 1), 0.9), ((1, 2), 0.9)]
    assert content_score(both, [EP[1], EP[2]], 6, cfg) > content_score(both, [EP[2], EP[3]], 6, cfg)


def _classified(files: list[MediaFile]) -> Any:
    from disc2plex.classify import classify

    return classify(files, expected_runtime_seconds=22 * 60).items


def test_votes_pin_an_ambiguous_start() -> None:
    files = [make_media(f"t{i:02d}.mkv", 22.0 + i * 0.1) for i in range(3)]
    plain = match_heuristic(_classified(files), SEASON)
    assert plain.start_source == "inferred"
    votes = {files[0].path: [((1, 4), 0.9)], files[2].path: [((1, 6), 0.9)]}
    voted = match_heuristic(_classified(files), SEASON, content_votes=votes)
    assert voted.start_position == 4
    assert [a.episode_label for a in voted.assignments] == ["s01e04", "s01e05", "s01e06"]
    assert voted.assignments[0].confidence > plain.assignments[0].confidence


def test_one_wrong_vote_does_not_break_the_sequence() -> None:
    files = [make_media(f"t{i:02d}.mkv", 22.0 + i * 0.1) for i in range(4)]
    votes = {
        files[0].path: [((1, 1), 0.93)],
        files[1].path: [((1, 2), 0.93)],
        files[2].path: [((1, 6), 0.85)],  # confidently wrong
        files[3].path: [((1, 4), 0.95)],
    }
    voted = match_heuristic(_classified(files), SEASON, content_votes=votes)
    assert [a.episode_label for a in voted.assignments] == ["s01e01", "s01e02", "s01e03", "s01e04"]


# --- apply_evidence -----------------------------------------------------------


def assignment(
    name: str, episodes: Sequence[int], confidence: float, minutes: float = 22
) -> Assignment:
    return Assignment(
        file=make_media(f"{name}.mkv", minutes),
        file_class=FileClass.EPISODE if episodes else FileClass.UNKNOWN,
        episodes=tuple(EP[n] for n in episodes),
        confidence=confidence,
    )


def evidence_for(**answers: list[tuple[int | None, float]]) -> Evidence:
    ev = Evidence()
    ev.usage.backend = "fake"
    for name, items in answers.items():
        ev.by_file[Path(f"/rip/{name}.mkv")] = FileEvidence(
            answers=tuple(
                ExcerptAnswer(i + 1, EP[n] if n else None, c, "r") for i, (n, c) in enumerate(items)
            )
        )
    return ev


def applied(assignments: list[Assignment], ev: Evidence) -> dict[str, Assignment]:
    result = apply_evidence(MatchResult(assignments=tuple(assignments)), ev)
    return {a.file.path.stem: a for a in result.assignments}


def test_agreement_boosts_confidence() -> None:
    out = applied([assignment("t00", [3], 0.5)], evidence_for(t00=[(3, 0.9)]))["t00"]
    assert out.confidence == pytest.approx(1 - 0.5 * (1 - 0.9 * 0.9), abs=1e-3)
    assert out.signals["content"] == 0.9
    assert any("agrees" in r for r in out.reasons)


def test_partial_agreement_for_two_episode_file() -> None:
    out = applied([assignment("t00", [1, 2], 0.6, 44)], evidence_for(t00=[(1, 0.9), (None, 0.0)]))
    assert 0.6 < out["t00"].confidence < 1 - 0.4 * (1 - 0.9 * 0.9)
    assert any("partly agrees" in r for r in out["t00"].reasons)


def test_weak_answers_are_no_signal() -> None:
    out = applied([assignment("t00", [3], 0.8)], evidence_for(t00=[(5, 0.3)]))["t00"]
    assert out.confidence == 0.8 and out.episodes == (EP[3],)
    assert any("no confident identification" in r for r in out.reasons)


def test_confident_disagreement_reassigns_but_caps_at_review() -> None:
    out = applied([assignment("t00", [3], 0.9)], evidence_for(t00=[(5, 0.95)]))["t00"]
    assert out.episodes == (EP[5],)
    assert out.confidence <= ContentConfig().review_cap
    assert any("reassigned to s01e05" in r for r in out.reasons)


def test_disagreement_with_supported_owner_is_demoted_not_reassigned() -> None:
    # Mirrors the real disc: t04 claims e02, but t00 holds e02 with agreeing evidence.
    out = applied(
        [assignment("t00", [1, 2], 0.8, 44), assignment("t04", [6], 0.9)],
        evidence_for(t00=[(1, 0.92), (2, 0.9)], t04=[(2, 0.85)]),
    )
    assert out["t04"].episodes == (EP[6],)
    assert out["t04"].confidence <= ContentConfig().disagree_cap
    assert any("t00.mkv has stronger evidence" in r for r in out["t04"].reasons)


def test_weak_disagreement_goes_to_review() -> None:
    out = applied([assignment("t00", [3], 0.95)], evidence_for(t00=[(5, 0.7)]))["t00"]
    assert out.episodes == (EP[3],)
    assert out.confidence == ContentConfig().disagree_cap


def test_unknown_file_can_get_a_capped_guess() -> None:
    out = applied([assignment("t09", [], 0.0)], evidence_for(t09=[(4, 0.95)]))["t09"]
    assert out.file_class is FileClass.EPISODE and out.episodes == (EP[4],)
    assert out.confidence <= ContentConfig().review_cap


def test_problems_and_usage_are_reported() -> None:
    ev = Evidence()
    ev.usage.backend = "fake"
    ev.problems[Path("/rip/t00.mkv")] = "too little speech to judge"
    result = apply_evidence(MatchResult(assignments=(assignment("t00", [3], 0.7),)), ev)
    assert "content: too little speech to judge" in result.assignments[0].reasons
    assert any("content verification (fake)" in n for n in result.notes)


# --- gather (fake transcription) ------------------------------------------------


@pytest.fixture
def fake_transcribe(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Replace transcription: text embeds the file stem; records slice counts."""
    calls: list[tuple[str, int]] = []

    def fake(
        media: MediaFile, transcriber: object, *, lang: str, slices: list[SliceSpec], cache: object
    ) -> FileTranscript:
        calls.append((media.path.stem, len(slices)))
        filler = " words" * 40
        return FileTranscript(
            file=media.path,
            audio_stream_index=1,
            model="fake",
            slices=tuple(
                SliceTranscript(
                    start_seconds=s.start_seconds,
                    duration_seconds=s.duration_seconds,
                    text=f"FILE {media.path.stem} PART {i + 1}{filler}",
                )
                for i, s in enumerate(slices)
            ),
        )

    monkeypatch.setattr(content_mod, "transcribe_file", fake)
    return calls


class OracleLLM:
    """Answers from a table keyed by file stem: stem -> [(episode code, confidence)]."""

    name = "fake/oracle"

    def __init__(self, table: dict[str, list[tuple[str, float]]]) -> None:
        self.table = table
        self.calls = 0

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        self.calls += 1
        stem = re.search(r"FILE (\S+) PART", user)
        assert stem is not None
        items = self.table[stem.group(1)]
        text = json.dumps(
            {
                "excerpts": [
                    {"excerpt": i + 1, "episode": code, "confidence": c, "reasoning": "plot match"}
                    for i, (code, c) in enumerate(items)
                ]
            }
        )
        return LLMResponse(text=text, model="fake", input_tokens=1000, output_tokens=50)


def _verifier(llm: Any, **config: Any) -> ContentVerifier:
    return ContentVerifier(transcriber=object(), llm=llm, config=ContentConfig(**config))  # type: ignore[arg-type]


def test_targets_low_vs_all() -> None:
    match = MatchResult(
        assignments=(
            assignment("t00", [1], 0.95),
            assignment("t01", [2], 0.6),
            assignment("t02", [], 0.0, minutes=22),  # plausible UNKNOWN
            assignment("t03", [], 0.0, minutes=3),  # implausibly short UNKNOWN
        )
    )
    verifier = _verifier(OracleLLM({}))
    assert [a.file.path.stem for a in verifier.targets(match, VerifyMode.LOW)] == ["t01", "t02"]
    assert [a.file.path.stem for a in verifier.targets(match, VerifyMode.ALL)] == [
        "t00",
        "t01",
        "t02",
    ]
    assert verifier.targets(match, VerifyMode.NONE) == []


def test_targets_skip_user_assigned_files() -> None:
    assigned = assignment("t00", [1], 1.0).model_copy(update={"signals": {"assigned": 1.0}})
    match = MatchResult(assignments=(assigned, assignment("t01", [2], 0.95)))
    verifier = _verifier(OracleLLM({}))
    assert [a.file.path.stem for a in verifier.targets(match, VerifyMode.ALL)] == ["t01"]


def test_usage_note_only_when_verification_did_something() -> None:
    match = MatchResult(assignments=(assignment("t00", [1], 0.95),))
    idle = Evidence()
    idle.usage.backend = "fake"
    assert apply_evidence(match, idle).notes == ()


def test_gather_uses_two_slices_for_double_files(fake_transcribe: list[tuple[str, int]]) -> None:
    match = MatchResult(
        assignments=(assignment("t00", [1, 2], 0.5, 44), assignment("t01", [3], 0.5)),
    )
    llm = OracleLLM({"t00": [("s01e01", 0.9), ("s01e02", 0.9)], "t01": [("s01e03", 0.9)]})
    evidence = _verifier(llm).gather(match, SEASON, VerifyMode.LOW)
    assert fake_transcribe == [("t00", 2), ("t01", 1)]
    assert len(evidence.by_file) == 2
    assert evidence.usage.llm_calls == 2 and evidence.usage.input_tokens == 2000


def test_gather_respects_call_cap(fake_transcribe: list[tuple[str, int]]) -> None:
    match = MatchResult(assignments=tuple(assignment(f"t0{i}", [i + 1], 0.5) for i in range(3)))
    llm = OracleLLM({f"t0{i}": [(f"s01e0{i + 1}", 0.9)] for i in range(3)})
    evidence = _verifier(llm, max_llm_calls=2).gather(match, SEASON, VerifyMode.LOW)
    assert llm.calls == 2
    assert any("call cap" in n for n in evidence.usage.notes)


def test_gather_stops_on_llm_error(fake_transcribe: list[tuple[str, int]]) -> None:
    match = MatchResult(assignments=tuple(assignment(f"t0{i}", [i + 1], 0.5) for i in range(3)))
    evidence = _verifier(ScriptedLLM(LLMError("bad key"))).gather(match, SEASON, VerifyMode.LOW)
    assert evidence.by_file == {}
    assert any("stopped: bad key" in n for n in evidence.usage.notes)


def test_gather_skips_files_without_enough_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        content_mod, "transcribe_file", lambda media, *_a, **_k: transcript("uh", path=media.path)
    )
    match = MatchResult(assignments=(assignment("t00", [1], 0.5),))
    evidence = _verifier(OracleLLM({})).gather(match, SEASON, VerifyMode.LOW)
    assert evidence.problems[Path("/rip/t00.mkv")] == "too little speech to judge"


# --- end to end on the real disc's metadata -------------------------------------

# What Claude answered for each title in the live run (t04 was wrong: it is e06).
REAL_ANSWERS = {
    "GOOD_PLACE_S1_D1_t00": [("s01e01", 0.92), ("s01e02", 0.9)],
    "GOOD_PLACE_S1_D1_t01": [("s01e03", 0.93)],
    "GOOD_PLACE_S1_D1_t02": [("s01e04", 0.93)],
    "GOOD_PLACE_S1_D1_t03": [("s01e05", 0.72)],
    "GOOD_PLACE_S1_D1_t04": [("s01e02", 0.85)],
    "GOOD_PLACE_S1_D1_t06": [("s01e07", 0.95)],
}
GOOD_PLACE_ROUTES = {
    "/3/tv/66573": "tv_66573",
    "/3/tv/66573/episode_groups": "tv_66573_episode_groups",
    "/3/tv/66573/season/1": "tv_66573_season_1",
}


def test_real_disc_without_hint_is_resolved_by_transcripts(
    fake_rip: FakeRipBuilder, fake_transcribe: list[tuple[str, int]], tmp_path: Path
) -> None:
    data = json.loads((FIXTURES / "real" / "good_place_s1_d1.json").read_text())
    rip_dir, probe = fake_rip({n: (i["probe"], i["size_bytes"]) for n, i in data.items()})

    def scanner(path: Path) -> ScanResult:
        return scan_directory(path, probe=probe)

    run = run_plan(
        PlanRequest(
            rip_dir=rip_dir,
            output_root=tmp_path / "lib",
            tmdb_id=66573,
            season=1,
            verify=VerifyMode.LOW,
        ),
        scanner=scanner,
        client=FakeTMDB(GOOD_PLACE_ROUTES).client(),
        verifier=_verifier(OracleLLM(REAL_ANSWERS)),
    )
    result = {a.file.path.stem[-3:]: (a.episode_label, a.status) for a in run.match.assignments}
    confident = AssignmentStatus.CONFIDENT
    assert result["t00"] == ("s01e01-e02", confident)
    assert result["t01"] == ("s01e03", confident)
    assert result["t02"] == ("s01e04", confident)
    assert result["t03"] == ("s01e05", confident)
    assert result["t06"] == ("s01e07", confident)
    # The wrong answer can't move t04 out of sequence; it just sends it to review.
    assert result["t04"] == ("s01e06", AssignmentStatus.NEEDS_REVIEW)
    assert run.match.start_position == 1
    # Five files back the start, so only the odd one out was listened to again.
    stems = [stem[-3:] for stem, _ in fake_transcribe]
    assert stems.count("t04") == 2 and all(stems.count(t) == 1 for t in ("t00", "t03", "t06"))


def test_two_titles_with_contradicting_transcripts_are_not_confident(
    fake_rip: FakeRipBuilder, fake_transcribe: list[tuple[str, int]], tmp_path: Path
) -> None:
    # Found by a real apply on copies of t03 + t04 only: t04's wrong "e02" answer
    # pulled the inferred start to 1 and then "agreed" with it, at 91%.
    data = json.loads((FIXTURES / "real" / "good_place_s1_d1.json").read_text())
    subset = {n: i for n, i in data.items() if n.endswith(("_t03.mkv", "_t04.mkv"))}
    rip_dir, probe = fake_rip({n: (i["probe"], i["size_bytes"]) for n, i in subset.items()})

    def scanner(path: Path) -> ScanResult:
        return scan_directory(path, probe=probe)

    run = run_plan(
        PlanRequest(
            rip_dir=rip_dir,
            output_root=tmp_path / "lib",
            tmdb_id=66573,
            season=1,
            verify=VerifyMode.LOW,
        ),
        scanner=scanner,
        client=FakeTMDB(GOOD_PLACE_ROUTES).client(),
        verifier=_verifier(OracleLLM(REAL_ANSWERS)),
    )
    assert all(a.status is not AssignmentStatus.CONFIDENT for a in run.match.assignments)
    assert any("disagree about where the disc starts" in n for n in run.match.notes)


def test_agreement_needs_a_majority_when_the_start_is_inferred() -> None:
    def run(start_source: str | None) -> dict[str, Assignment]:
        match = MatchResult(
            assignments=(assignment("t00", [1], 0.6), assignment("t01", [2], 0.6)),
            start_source=start_source,
        )
        ev = evidence_for(t00=[(1, 0.9)], t01=[(5, 0.8)])
        return {a.file.path.stem: a for a in apply_evidence(match, ev).assignments}

    inferred = run("inferred")["t00"]
    assert inferred.confidence == ContentConfig().review_cap
    assert any("contradict the inferred start" in r for r in inferred.reasons)
    # A start the user pinned (--disc / --disc-episodes) isn't in question.
    assert run("disc1")["t00"].confidence > ContentConfig().confident_threshold


# --- second listen ------------------------------------------------------------


def _overlaps(a: SliceSpec, b: SliceSpec) -> bool:
    return a.start_seconds < b.start_seconds + b.duration_seconds and b.start_seconds < (
        a.start_seconds + a.duration_seconds
    )


@pytest.mark.parametrize(("episodes", "minutes", "count"), [([3], 22, 2), ([1, 2], 44, 4)])
def test_second_slices_cover_new_ground(episodes: list[int], minutes: float, count: int) -> None:
    a = assignment("t00", episodes, 0.5, minutes)
    first = content_mod._slices_for(a, 22 * 60)
    second = content_mod._second_slices(a, 22 * 60, 150)
    assert len(second) == count
    assert not any(_overlaps(x, y) for x in first for y in second)
    duration = minutes * 60
    # Clear of the first minute (recaps) and the last minute (credits).
    assert all(
        s.start_seconds > 60 and s.start_seconds + s.duration_seconds < duration - 60
        for s in second
    )


def test_prompt_says_whether_excerpts_share_an_episode() -> None:
    two = transcript("x" * 200, "y" * 200)
    assert "come from the same episode" in build_user_prompt(SEASON, two, 1320)
    assert "two consecutive episodes" in build_user_prompt(SEASON, two, 2640, double=True)
    assert "same episode" not in build_user_prompt(SEASON, transcript("x" * 200), 1320)


@pytest.mark.parametrize(
    ("episodes", "answers", "expected"),
    [
        ([3], [(3, 0.9)], True),
        ([1, 2], [(1, 0.9), (None, 0.0)], True),
        ([3], [(5, 0.9)], False),  # contradicts the match
        ([3], [(3, 0.3)], False),  # nothing clear
        ([], [(4, 0.95)], False),  # no match to back up
    ],
)
def test_settled(
    episodes: list[int], answers: list[tuple[int | None, float]], expected: bool
) -> None:
    ev = evidence_for(t00=answers).by_file[Path("/rip/t00.mkv")]
    assert content_mod.settled(assignment("t00", episodes, 0.5), ev, ContentConfig()) is expected


class TwoPassLLM:
    """Answers by file stem; a prompt with more than one excerpt of a single
    file gets the ``second`` answer (one per excerpt)."""

    name = "fake/two-pass"

    def __init__(self, first: dict[str, str], second: dict[str, str], confidence: float = 0.9):
        self.first, self.second, self.confidence = first, second, confidence
        self.prompts: list[str] = []

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        self.prompts.append(user)
        stem = re.search(r"FILE (\S+) PART", user)
        assert stem is not None
        count = len(re.findall(r"^Excerpt \d+ ", user, flags=re.MULTILINE))
        table = self.second if count > 1 else self.first
        code = table[stem.group(1)]
        text = json.dumps(
            {
                "excerpts": [
                    {
                        "excerpt": i + 1,
                        "episode": code,
                        "confidence": self.confidence,
                        "reasoning": "plot",
                    }
                    for i in range(count)
                ]
            }
        )
        return LLMResponse(text=text, model="fake", input_tokens=1000, output_tokens=50)


def _gathered(verifier: ContentVerifier, match: MatchResult) -> Evidence:
    evidence = verifier.gather(match, SEASON, VerifyMode.ALL)
    verifier.listen_again(match, SEASON, evidence)
    return evidence


def test_listen_again_only_for_unsettled_files(fake_transcribe: list[tuple[str, int]]) -> None:
    match = MatchResult(assignments=(assignment("t00", [3], 0.5), assignment("t01", [4], 0.5)))
    llm = TwoPassLLM({"t00": "s01e03", "t01": "s01e02"}, {"t01": "s01e04"})
    evidence = _gathered(_verifier(llm), match)
    # t00 agreed, so only t01 was transcribed again (two more slices) and re-asked.
    assert fake_transcribe == [("t00", 1), ("t01", 1), ("t01", 2)]
    assert "Excerpt 3" in llm.prompts[-1] and "same episode" in llm.prompts[-1]
    t01 = evidence.by_file[Path("/rip/t01.mkv")]
    assert [a.episode for a in t01.answers] == [EP[4]] * 3
    assert [a.episode for a in t01.first_answers] == [EP[2]]
    assert evidence.usage.files_relistened == 1 and evidence.usage.llm_calls == 3
    assert "(1 listened to twice)" in evidence.usage.summary_line

    out = applied(list(match.assignments), evidence)["t01"]
    assert out.episodes == (EP[4],) and out.confidence > 0.85
    assert any(r.startswith("first excerpt: excerpt 1 -> s01e02") for r in out.reasons)
    assert any(r.startswith("listened again with more excerpts:") for r in out.reasons)


def test_listen_again_can_be_turned_off_and_respects_the_cap(
    fake_transcribe: list[tuple[str, int]],
) -> None:
    match = MatchResult(assignments=(assignment("t00", [3], 0.5), assignment("t01", [4], 0.5)))
    wrong = {"t00": "s01e05", "t01": "s01e06"}
    off = _gathered(_verifier(TwoPassLLM(wrong, wrong), second_listen=False), match)
    assert off.usage.files_relistened == 0 and off.usage.llm_calls == 2

    capped = _gathered(_verifier(TwoPassLLM(wrong, wrong), max_llm_calls=3), match)
    assert capped.usage.files_relistened == 1 and capped.usage.llm_calls == 3
    assert any("1 file(s) not listened to again" in n for n in capped.usage.notes)


def test_second_listen_fixes_a_wrong_first_answer_end_to_end(
    fake_rip: FakeRipBuilder, fake_transcribe: list[tuple[str, int]], tmp_path: Path
) -> None:
    # The t03 + t04 pair again: t04's middle excerpt fooled Claude into "e02".
    # If more excerpts tell the truth, both files are identified.
    data = json.loads((FIXTURES / "real" / "good_place_s1_d1.json").read_text())
    subset = {n: i for n, i in data.items() if n.endswith(("_t03.mkv", "_t04.mkv"))}
    rip_dir, probe = fake_rip({n: (i["probe"], i["size_bytes"]) for n, i in subset.items()})
    llm = TwoPassLLM(
        {"GOOD_PLACE_S1_D1_t03": "s01e05", "GOOD_PLACE_S1_D1_t04": "s01e02"},
        {"GOOD_PLACE_S1_D1_t03": "s01e05", "GOOD_PLACE_S1_D1_t04": "s01e06"},
    )
    run = run_plan(
        PlanRequest(
            rip_dir=rip_dir,
            output_root=tmp_path / "lib",
            tmdb_id=66573,
            season=1,
            verify=VerifyMode.LOW,
        ),
        scanner=lambda path: scan_directory(path, probe=probe),
        client=FakeTMDB(GOOD_PLACE_ROUTES).client(),
        verifier=_verifier(llm),
    )
    # The wrong answer won the first alignment, so t03 looked like the odd one out;
    # with the start contested, both files were listened to again.
    assert [n for n, slices in fake_transcribe if slices == 2] == [
        "GOOD_PLACE_S1_D1_t03",
        "GOOD_PLACE_S1_D1_t04",
    ]
    result = {a.file.path.stem[-3:]: (a.episode_label, a.status) for a in run.match.assignments}
    assert result == {
        "t03": ("s01e05", AssignmentStatus.CONFIDENT),
        "t04": ("s01e06", AssignmentStatus.CONFIDENT),
    }


def test_scattered_answers_mean_a_compilation() -> None:
    # Parks and Recreation S1: a 23-minute stereo title whose excerpts named three episodes.
    out = applied([assignment("t12", [], 0.0)], evidence_for(t12=[(1, 0.6), (2, 0.65), (4, 0.7)]))
    assert out["t12"].episodes == () and out["t12"].confidence == 0.0
    assert any("probably a compilation" in r for r in out["t12"].reasons)


def test_clear_duplicate_suggests_another_cut() -> None:
    # Parks and Recreation S1: broadcast and extended cuts of "Rock Show". t06 is
    # episode length but was left unmatched (t07 holds e06).
    t06 = assignment("t06", [], 0.0).model_copy(update={"file_class": FileClass.EPISODE})
    out = applied(
        [t06, assignment("t07", [6], 0.9, 27)],
        evidence_for(t06=[(6, 0.97)], t07=[(6, 0.97)]),
    )
    assert any("t07.mkv has stronger evidence" in r for r in out["t06"].reasons)
    # Only a second listen that still says "Rock Show" suggests a second cut.
    ev = evidence_for(t06=[(6, 0.97), (6, 0.97), (6, 0.93)], t07=[(6, 0.97)])
    first = ev.by_file[Path("/rip/t06.mkv")]
    ev.by_file[Path("/rip/t06.mkv")] = replace(first, first_answers=first.answers[:1])
    out = applied([t06, assignment("t07", [6], 0.9, 27)], ev)
    assert out["t06"].episodes == () and out["t07"].episodes == (EP[6],)
    assert any("another cut of the same episode" in r for r in out["t06"].reasons)


class ReplayLLM:
    """Replays real answers by file stem: first listen, then (optionally) second."""

    name = "fake/replay"

    def __init__(
        self,
        first: dict[str, tuple[str, float]],
        second: dict[str, list[tuple[str, float]]],
    ) -> None:
        self.first, self.second = first, second

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        stem = re.search(r"FILE (\S+) PART", user)
        assert stem is not None
        relisten = "Excerpt 2 " in user
        items = self.second[stem.group(1)] if relisten else [self.first[stem.group(1)]]
        text = json.dumps(
            {
                "excerpts": [
                    {"excerpt": i + 1, "episode": code, "confidence": c, "reasoning": "plot"}
                    for i, (code, c) in enumerate(items)
                ]
            }
        )
        return LLMResponse(text=text, model="fake", input_tokens=1000, output_tokens=50)


def test_real_parks_disc_with_every_file_checked(
    fake_rip: FakeRipBuilder, fake_transcribe: list[tuple[str, int]], tmp_path: Path
) -> None:
    # What Claude answered in the live run (verify all, no disc hint).
    llm = ReplayLLM(
        {
            "C1_t00": ("s01e01", 0.92),
            "C2_t01": ("s01e02", 0.95),
            "C3_t02": ("s01e03", 0.97),
            "D1_t03": ("s01e04", 0.90),
            "D2_t04": ("s01e05", 0.97),
            "E1_t06": ("s01e06", 0.97),
            "E2_t07": ("s01e06", 0.97),
            "G1_t12": ("s01e02", 0.95),
        },
        {
            "E1_t06": [("s01e06", 0.97), ("s01e06", 0.97), ("s01e06", 0.93)],
            "G1_t12": [("s01e01", 0.60), ("s01e02", 0.65), ("s01e04", 0.70)],
        },
    )
    data = json.loads((FIXTURES / "real" / "parks_s1_d1.json").read_text())
    rip_dir, probe = fake_rip({n: (i["probe"], i["size_bytes"]) for n, i in data.items()})
    run = run_plan(
        PlanRequest(
            rip_dir=rip_dir,
            output_root=tmp_path / "lib",
            tmdb_id=8592,
            season=1,
            verify=VerifyMode.ALL,
        ),
        scanner=lambda path: scan_directory(path, probe=probe),
        client=FakeTMDB(
            {
                "/3/tv/8592": "tv_8592",
                "/3/tv/8592/episode_groups": "tv_8592_episode_groups",
                "/3/tv/8592/season/1": "tv_8592_season_1",
            }
        ).client(),
        verifier=_verifier(llm),
    )
    by_stem = {a.file.path.stem: a for a in run.match.assignments}
    confident = AssignmentStatus.CONFIDENT
    expected = {"C1_t00": 1, "C2_t01": 2, "C3_t02": 3, "D1_t03": 4, "D2_t04": 5, "E2_t07": 6}
    for stem, number in expected.items():
        a = by_stem[stem]
        assert ([e.episode for e in a.episodes], a.status) == ([number], confident)
    for stem, phrase in (("E1_t06", "another cut"), ("G1_t12", "compilation")):
        assert by_stem[stem].status is AssignmentStatus.UNMATCHED
        assert any(phrase in r for r in by_stem[stem].reasons)
    assert sorted(n for n, slices in fake_transcribe if slices == 2) == ["E1_t06", "G1_t12"]


def test_clear_duplicate_with_wrong_length_is_extra_material() -> None:
    # Parks and Recreation S2D1: an 11-minute title that is clearly "Sister City".
    ev = evidence_for(t14=[(5, 0.97), (5, 0.96), (5, 0.93)], t06=[(5, 0.95)])
    first = ev.by_file[Path("/rip/t14.mkv")]
    ev.by_file[Path("/rip/t14.mkv")] = replace(first, first_answers=first.answers[:1])
    out = applied([assignment("t14", [], 0.0, 11), assignment("t06", [5], 0.9)], ev)
    assert any("deleted or extended scenes" in r for r in out["t14"].reasons)


def test_real_parks_s2_disc4_duplicate_cuts_do_not_contest_the_start(
    fake_rip: FakeRipBuilder, fake_transcribe: list[tuple[str, int]], tmp_path: Path
) -> None:
    # Parks and Recreation S2 disc 4: e23 and e24 each appear twice (broadcast cut
    # and the longer cut TMDB lists), F1_t17 is more Telethon material, and E2_t11
    # is a blooper reel. Those duplicates must not count against the inferred start.
    llm = ReplayLLM(
        {
            "C1_t04": ("s02e21", 0.95),
            "C2_t05": ("s02e22", 0.98),
            "D1_t07": ("s02e23", 0.93),
            "D2_t08": ("s02e24", 0.97),
            "D3_t09": ("s02e23", 0.95),
            "D4_t10": ("s02e24", 0.95),
            "F1_t17": ("s02e22", 0.82),
            "E2_t11": ("none", 0.6),
        },
        {
            "C1_t04": [("s02e21", 0.97)] * 3,
            "C2_t05": [("s02e22", 0.97)] * 3,
            "D1_t07": [("s02e23", 0.95), ("s02e23", 0.93), ("s02e23", 0.93)],
            "D2_t08": [("s02e24", 0.97), ("s02e24", 0.97), ("s02e24", 0.95)],
            "D3_t09": [("s02e23", 0.95)] * 3,
            "D4_t10": [("s02e24", 0.97)] * 3,
            "F1_t17": [("s02e22", 0.80), ("s02e22", 0.92), ("s02e22", 0.85)],
            "E2_t11": [("none", 0.6)] * 3,
        },
    )
    data = json.loads((FIXTURES / "real" / "parks_s2_d4.json").read_text())
    rip_dir, probe = fake_rip({n: (i["probe"], i["size_bytes"]) for n, i in data.items()})
    run = run_plan(
        PlanRequest(
            rip_dir=rip_dir,
            output_root=tmp_path / "lib",
            tmdb_id=8592,
            season=2,
            disc=4,
            verify=VerifyMode.LOW,
        ),
        scanner=lambda path: scan_directory(path, probe=probe),
        client=FakeTMDB(
            {
                "/3/tv/8592": "tv_8592",
                "/3/tv/8592/episode_groups": "tv_8592_episode_groups",
                "/3/tv/8592/season/2": "tv_8592_season_2",
            }
        ).client(),
        verifier=_verifier(llm),
    )
    by_stem = {a.file.path.stem: a for a in run.match.assignments}
    for stem, number in {"C1_t04": 21, "C2_t05": 22, "D3_t09": 23, "D4_t10": 24}.items():
        a = by_stem[stem]
        assert ([e.episode for e in a.episodes], a.status) == ([number], AssignmentStatus.CONFIDENT)
    for stem in ("D1_t07", "D2_t08", "F1_t17", "E2_t11"):
        assert by_stem[stem].status is not AssignmentStatus.CONFIDENT
    assert not any("disagree about where the disc starts" in n for n in run.match.notes)


def test_real_parks_s6_disc1_long_premiere_is_asked_about_as_one_episode(
    fake_rip: FakeRipBuilder, fake_transcribe: list[tuple[str, int]], tmp_path: Path
) -> None:
    # What a local Qwen3.6-35B answered in the live run (verify all, no disc hint).
    # C1_t01 is the 43-minute premiere "London": one episode, so one excerpt and no
    # "probably holds two consecutive episodes" prompt. Asked as two halves, its
    # Pawnee scenes were taken for other episodes and it went to review.
    llm = ReplayLLM(
        {
            "C1_t01": ("s06e01", 0.95),
            "C3_t02": ("s06e02", 0.95),
            "C5_t03": ("s06e03", 0.95),
            "F1_t04": ("s06e04", 0.95),
            "F3_t05": ("s06e05", 0.95),
            "F5_t06": ("s06e06", 0.95),
            "G1_t07": ("s06e07", 0.95),
            "G3_t08": ("s06e08", 0.95),
            "H1_t09": ("s06e06", 0.95),
        },
        {"H1_t09": [("s06e06", 0.95), ("s06e06", 0.98), ("s06e06", 0.95)]},
    )
    data = json.loads((FIXTURES / "real" / "parks_s6_d1.json").read_text())
    rip_dir, probe = fake_rip({n: (i["probe"], i["size_bytes"]) for n, i in data.items()})
    run = run_plan(
        PlanRequest(
            rip_dir=rip_dir,
            output_root=tmp_path / "lib",
            tmdb_id=8592,
            season=6,
            verify=VerifyMode.ALL,
        ),
        scanner=lambda path: scan_directory(path, probe=probe),
        client=FakeTMDB(
            {
                "/3/tv/8592": "tv_8592",
                "/3/tv/8592/episode_groups": "tv_8592_episode_groups",
                "/3/tv/8592/season/6": "tv_8592_season_6",
            }
        ).client(),
        verifier=_verifier(llm),
    )
    by_stem = {a.file.path.stem: a for a in run.match.assignments}
    stems = ["C1_t01", "C3_t02", "C5_t03", "F1_t04", "F3_t05", "F5_t06", "G1_t07", "G3_t08"]
    for number, stem in enumerate(stems, start=1):
        a = by_stem[stem]
        assert ([e.episode for e in a.episodes], a.status) == ([number], AssignmentStatus.CONFIDENT)
    assert by_stem["H1_t09"].status is not AssignmentStatus.CONFIDENT
    assert any("another cut" in r for r in by_stem["H1_t09"].reasons)
    assert ("C1_t01", 1) in fake_transcribe
    assert [stem for stem, slices in fake_transcribe if slices == 2] == ["H1_t09"]
