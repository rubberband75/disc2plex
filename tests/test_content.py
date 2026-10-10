"""Tests for content verification: prompts, reply validation, scoring, combining.

The LLM and transcription are faked. The pipeline test replays the answers
Claude actually gave for The Good Place S1 disc 1 (including one wrong answer).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
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
