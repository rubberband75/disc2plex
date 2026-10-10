"""Stage 5b: identify episodes from what is said in them, and merge that evidence.

For each file worth checking, a Whisper transcript of one or two excerpts is
sent to an LLM together with the season's TMDB synopses. The LLM must answer
with JSON naming one listed episode (or "none") per excerpt. That answer is one
signal, never taken as truth:

1. ``ContentVerifier.gather`` collects answers for the files worth checking;
2. the heuristic aligner re-runs with those answers as votes in its pair scores,
   so a few clear identifications can pin down where the disc starts;
3. ``apply_evidence`` boosts agreeing files and demotes or (cautiously)
   reassigns disagreeing ones; ``combine.decide`` still flags any two files
   claiming one episode.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from disc2plex.llm.base import LLMBackend, LLMError
from disc2plex.match.heuristic import MISMATCH_FIT, ContentVotes, HeuristicConfig, runtime_fit
from disc2plex.models import (
    Assignment,
    EpisodeInfo,
    FileClass,
    FileTranscript,
    MatchResult,
    SeasonEpisodes,
)
from disc2plex.report import format_duration
from disc2plex.transcribe import (
    SliceSpec,
    Transcriber,
    TranscriptCache,
    TranscriptionError,
    choose_slices,
    transcribe_file,
)

log = logging.getLogger(__name__)

NONE_CODE = "none"


class VerifyMode(StrEnum):
    NONE = "none"
    LOW = "low"  # only files the heuristic isn't confident about
    ALL = "all"


@dataclass(frozen=True)
class ContentConfig:
    # Agreement: combined = 1 - (1 - heuristic) * (1 - agree_weight * content).
    agree_weight: float = 0.9
    partial_agree_weight: float = 0.5
    # Reassign only when the LLM is at least this sure and the runtime fits.
    reassign_min_confidence: float = 0.85
    reassign_scale: float = 0.95
    # Ceiling for a file the heuristic left UNKNOWN or reassigned against order evidence
    # when the runtime can't corroborate it.
    review_cap: float = 0.84
    # Answers below this confidence are treated as "no signal".
    min_answer_confidence: float = 0.5
    # A disagreement caps the heuristic's confidence here (sends it to review).
    disagree_cap: float = 0.7
    # Transcripts shorter than this carry too little to judge.
    min_transcript_chars: int = 150
    max_llm_calls: int = 30
    confident_threshold: float = 0.85


SYSTEM_PROMPT = """\
You identify which TV episode a transcript excerpt comes from.
Use only the evidence in the excerpt and the episode synopses provided.
The transcript is automatic speech recognition: names may be misspelled or \
phonetically garbled, so match names by sound and match plot details.
For each excerpt, answer with exactly one episode code from the list, or "none" \
if no synopsis clearly fits. Say "none" rather than guess.
Confidence guide: 0.9 or higher only when specific plot details match one \
synopsis and no other; 0.6-0.8 when it fits best but other episodes are possible; \
below 0.5 if unsure. Keep reasoning to one short sentence citing the deciding detail."""


def episode_code(e: EpisodeInfo) -> str:
    return f"s{e.season:02d}e{e.episode:02d}"


def answer_schema(codes: Sequence[str]) -> dict[str, Any]:
    """JSON schema for the reply; the enum restricts answers to listed episodes."""
    return {
        "type": "object",
        "properties": {
            "excerpts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "excerpt": {"type": "integer"},
                        "episode": {"type": "string", "enum": [*codes, NONE_CODE]},
                        "confidence": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["excerpt", "episode", "confidence", "reasoning"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["excerpts"],
        "additionalProperties": False,
    }


def build_user_prompt(
    season: SeasonEpisodes, transcript: FileTranscript, file_duration: float
) -> str:
    """Show, candidate episodes with synopses, then numbered transcript excerpts."""
    show = season.show
    lines = [
        f"Show: {show.name} ({show.year or 'year unknown'})",
        f"Season {season.season}. Candidate episodes:",
    ]
    for e in season.episodes:
        overview = " ".join(e.overview.split()) or "(no synopsis available)"
        lines.append(f'- {episode_code(e)} "{e.title}": {overview}')
    lines.append("")
    lines.append(f"The video file is {format_duration(file_duration)} long.")
    for number, s in enumerate(transcript.slices, start=1):
        end = s.start_seconds + s.duration_seconds
        lines.append(
            f"\nExcerpt {number} (from {format_duration(s.start_seconds)} to "
            f"{format_duration(end)}):\n{s.text.strip() or '(no speech recognised)'}"
        )
    lines.append(
        "\nFor each excerpt, which episode is it from? Reply with JSON matching the schema."
    )
    return "\n".join(lines)


class _ExcerptReply(BaseModel):
    excerpt: int
    episode: str
    confidence: float
    reasoning: str = ""


class _Reply(BaseModel):
    excerpts: list[_ExcerptReply]


@dataclass(frozen=True)
class ExcerptAnswer:
    excerpt: int
    episode: EpisodeInfo | None
    confidence: float
    reasoning: str


def parse_reply(text: str, season: SeasonEpisodes, excerpt_count: int) -> list[ExcerptAnswer]:
    """Validate the LLM's JSON. Raises ValueError if it can't be used at all.

    Unknown episode codes become "none", confidences are clamped to [0, 1], and
    excerpt numbers outside the sent range are dropped.
    """
    try:
        reply = _Reply.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"malformed LLM reply: {exc}") from exc
    by_code = {episode_code(e): e for e in season.episodes}
    answers: dict[int, ExcerptAnswer] = {}
    for item in reply.excerpts:
        if not 1 <= item.excerpt <= excerpt_count or item.excerpt in answers:
            continue
        answers[item.excerpt] = ExcerptAnswer(
            excerpt=item.excerpt,
            episode=by_code.get(item.episode.strip().lower()),
            confidence=min(max(item.confidence, 0.0), 1.0),
            reasoning=item.reasoning.strip(),
        )
    if not answers:
        raise ValueError("LLM reply had no usable excerpt answers")
    return [answers[k] for k in sorted(answers)]


@dataclass
class ContentUsage:
    """Running totals for the run's report."""

    files_checked: int = 0
    llm_calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    notes: list[str] = field(default_factory=list)
    backend: str = ""

    @property
    def summary_line(self) -> str:
        return (
            f"content verification ({self.backend}): {self.files_checked} file(s) checked, "
            f"{self.llm_calls} LLM call(s) ({self.cached_calls} from cache), "
            f"{self.input_tokens:,} input / {self.output_tokens:,} output tokens"
        )


def ask_llm(
    llm: LLMBackend,
    season: SeasonEpisodes,
    transcript: FileTranscript,
    file_duration: float,
    usage: ContentUsage,
) -> list[ExcerptAnswer] | None:
    """One LLM question per file, retried once on malformed JSON. None = no signal."""
    codes = [episode_code(e) for e in season.episodes]
    schema = answer_schema(codes)
    user = build_user_prompt(season, transcript, file_duration)
    for attempt in (1, 2):
        response = llm.complete_json(SYSTEM_PROMPT, user, schema)
        usage.llm_calls += 1
        usage.cached_calls += int(response.cached)
        usage.input_tokens += response.input_tokens
        usage.output_tokens += response.output_tokens
        try:
            return parse_reply(response.text, season, len(transcript.slices))
        except ValueError as exc:
            log.warning("attempt %d: %s", attempt, exc)
    return None


# --- combining ----------------------------------------------------------------


@dataclass(frozen=True)
class FileEvidence:
    """What the transcript said about one file."""

    answers: tuple[ExcerptAnswer, ...]

    def useful(self, min_confidence: float) -> list[ExcerptAnswer]:
        return [a for a in self.answers if a.episode and a.confidence >= min_confidence]


@dataclass
class Evidence:
    """Everything content verification gathered for a run."""

    by_file: dict[Path, FileEvidence] = field(default_factory=dict)
    # Files that were targeted but produced no evidence, with the reason.
    problems: dict[Path, str] = field(default_factory=dict)
    usage: ContentUsage = field(default_factory=ContentUsage)

    def votes(self, min_confidence: float) -> ContentVotes:
        """Answers in the form the heuristic aligner scores."""
        return {
            path: [
                ((a.episode.season, a.episode.episode), a.confidence)
                for a in ev.useful(min_confidence)
                if a.episode is not None
            ]
            for path, ev in self.by_file.items()
        }


def _consecutive(episodes: Sequence[EpisodeInfo]) -> bool:
    seasons = {e.season for e in episodes}
    numbers = sorted(e.episode for e in episodes)
    return len(seasons) == 1 and numbers == list(range(numbers[0], numbers[0] + len(numbers)))


def _describe(answers: Sequence[ExcerptAnswer]) -> str:
    parts = []
    for a in answers:
        what = f'{episode_code(a.episode)} "{a.episode.title}"' if a.episode else "no match"
        detail = f": {a.reasoning}" if a.reasoning else ""
        parts.append(f"excerpt {a.excerpt} -> {what} ({a.confidence:.0%}){detail}")
    return "transcript says " + "; ".join(parts)


def _claimed_with_support(
    episode: EpisodeInfo,
    owner: Assignment,
    assignments: Sequence[Assignment],
    evidence: Evidence,
    cfg: ContentConfig,
) -> Assignment | None:
    """Another file that holds ``episode`` and has its own reason to: agreeing
    transcript evidence, or a confident match with no transcript checked."""
    for other in assignments:
        if other is owner or episode not in other.episodes:
            continue
        ev = evidence.by_file.get(other.file.path)
        if ev is None:
            if other.confidence >= cfg.confident_threshold:
                return other
        elif any(a.episode == episode for a in ev.useful(cfg.reassign_min_confidence)):
            return other
    return None


def apply_evidence(
    match: MatchResult, evidence: Evidence, cfg: ContentConfig | None = None
) -> MatchResult:
    """Fold transcript answers into each assignment's confidence and reasons.

    Run after the aligner has already used the same evidence as votes. Here:
    agreement raises confidence; disagreement either reassigns the file (when
    the LLM is sure, the runtime fits, and no other file has supporting
    evidence for that episode; always capped at "needs review" because it goes
    against title order) or caps confidence so the file goes to review.
    """
    cfg = cfg or ContentConfig()
    heuristic_cfg = replace(HeuristicConfig(), runtime_scale=match.runtime_scale)
    original = list(match.assignments)
    updated: list[Assignment] = []
    for a in original:
        problem = evidence.problems.get(a.file.path)
        ev = evidence.by_file.get(a.file.path)
        if ev is None:
            reasons = (*a.reasons, f"content: {problem}") if problem else a.reasons
            updated.append(a.model_copy(update={"reasons": reasons}))
            continue
        updated.append(_combine_one(a, ev, original, evidence, heuristic_cfg, cfg))
    notes = [*match.notes, evidence.usage.summary_line, *evidence.usage.notes]
    return match.model_copy(update={"assignments": tuple(updated), "notes": tuple(notes)})


def _combine_one(
    a: Assignment,
    ev: FileEvidence,
    assignments: Sequence[Assignment],
    evidence: Evidence,
    heuristic_cfg: HeuristicConfig,
    cfg: ContentConfig,
) -> Assignment:
    reasons = [*a.reasons, _describe(ev.answers)]
    signals = dict(a.signals)
    useful = ev.useful(cfg.min_answer_confidence)
    if not useful:
        signals["content"] = 0.0
        reasons.append("content: no confident identification; order/runtime match kept")
        return a.model_copy(update={"reasons": tuple(reasons), "signals": signals})

    content_eps: list[EpisodeInfo] = []
    for answer in useful:
        assert answer.episode is not None
        if answer.episode not in content_eps:
            content_eps.append(answer.episode)
    content_conf = min(answer.confidence for answer in useful)
    signals["content"] = round(content_conf, 3)
    h = a.confidence

    def update(**changes: object) -> Assignment:
        return a.model_copy(update={"reasons": tuple(reasons), "signals": signals, **changes})

    if a.episodes and set(content_eps) == set(a.episodes):
        reasons.append("content agrees with the order/runtime match")
        return update(confidence=round(1 - (1 - h) * (1 - cfg.agree_weight * content_conf), 3))
    if a.episodes and set(content_eps) < set(a.episodes):
        reasons.append("content partly agrees (identified some of the matched episodes)")
        boost = cfg.partial_agree_weight * content_conf
        return update(confidence=round(1 - (1 - h) * (1 - boost), 3))

    new_eps = sorted(content_eps, key=lambda e: (e.season, e.episode))
    label = ", ".join(episode_code(e) for e in new_eps)
    supported = [
        owner
        for e in new_eps
        if (owner := _claimed_with_support(e, a, assignments, evidence, cfg)) is not None
    ]
    if supported:
        holder = supported[0].file.path.name
        reasons.append(
            f"content suggests {label}, but {holder} has stronger evidence for it; needs review"
        )
        return update(confidence=round(min(h, cfg.disagree_cap), 3))
    fit = runtime_fit(a.file.duration_seconds, new_eps, heuristic_cfg)
    runtime_ok = fit is None or fit >= MISMATCH_FIT
    if content_conf >= cfg.reassign_min_confidence and _consecutive(new_eps) and runtime_ok:
        runtime_factor = 1.0 if fit is None else min(1.0, 0.4 + fit)
        confidence = min(content_conf * cfg.reassign_scale * runtime_factor, cfg.review_cap)
        was = f" (order/runtime suggested {a.episode_label})" if a.episodes else ""
        reasons.append(f"reassigned to {label} from transcript evidence{was}; please confirm")
        return update(
            file_class=FileClass.EPISODE,
            episodes=tuple(new_eps),
            confidence=round(confidence, 3),
        )
    reasons.append(f"content suggests {label}, not strongly enough to reassign; needs review")
    return update(confidence=round(min(h, cfg.disagree_cap), 3))


# --- orchestration ------------------------------------------------------------


def _slices_for(a: Assignment, single_episode_seconds: float | None) -> list[SliceSpec]:
    """One middle slice, or one per half for files that look like two episodes."""
    duration = a.file.duration_seconds
    looks_double = len(a.episodes) >= 2 or (
        single_episode_seconds is not None and duration >= 1.6 * single_episode_seconds
    )
    if not looks_double:
        return choose_slices(duration)
    length = min(180.0, duration / 4)
    return [
        SliceSpec(max(0.0, duration * 0.25 - length / 2), length),
        SliceSpec(max(0.0, duration * 0.75 - length / 2), length),
    ]


def _typical_single_seconds(match: MatchResult) -> float | None:
    singles = [a.file.duration_seconds for a in match.assignments if len(a.episodes) == 1]
    return statistics.median(singles) if singles else None


@dataclass
class ContentVerifier:
    """Bundles the transcriber, LLM, and settings for content verification."""

    transcriber: Transcriber
    llm: LLMBackend
    lang: str = "en"
    transcript_cache: TranscriptCache | None = None
    config: ContentConfig = field(default_factory=ContentConfig)

    def targets(self, match: MatchResult, mode: VerifyMode) -> list[Assignment]:
        """Assignments worth verifying.

        ``low``: episodes below the confident threshold; ``all``: every episode.
        Both include UNKNOWN titles of plausible episode length (0.5-2.6x typical).
        """
        if mode is VerifyMode.NONE:
            return []
        typical = _typical_single_seconds(match)
        chosen = []
        for a in match.assignments:
            wanted_episode = a.file_class is FileClass.EPISODE and (
                mode is VerifyMode.ALL or a.confidence < self.config.confident_threshold
            )
            plausible_unknown = (
                a.file_class is FileClass.UNKNOWN
                and typical is not None
                and 0.5 * typical <= a.file.duration_seconds <= 2.6 * typical
            )
            if wanted_episode or plausible_unknown:
                chosen.append(a)
        return chosen

    def gather(self, match: MatchResult, season: SeasonEpisodes, mode: VerifyMode) -> Evidence:
        """Transcribe and ask the LLM about each target file."""
        evidence = Evidence()
        evidence.usage.backend = self.llm.name
        if not season.episodes:
            return evidence
        typical = _typical_single_seconds(match)
        targets = self.targets(match, mode)
        for position, a in enumerate(targets):
            if evidence.usage.llm_calls >= self.config.max_llm_calls:
                evidence.usage.notes.append(
                    f"LLM call cap ({self.config.max_llm_calls}) reached; "
                    f"{len(targets) - position} file(s) not verified"
                )
                break
            path = a.file.path
            try:
                transcript = transcribe_file(
                    a.file,
                    self.transcriber,
                    lang=self.lang,
                    slices=_slices_for(a, typical),
                    cache=self.transcript_cache,
                )
            except TranscriptionError as exc:
                evidence.problems[path] = f"transcription failed ({exc})"
                continue
            if len(transcript.text) < self.config.min_transcript_chars:
                evidence.problems[path] = "too little speech to judge"
                continue
            evidence.usage.files_checked += 1
            try:
                answers = ask_llm(
                    self.llm, season, transcript, a.file.duration_seconds, evidence.usage
                )
            except LLMError as exc:
                evidence.usage.notes.append(f"content verification stopped: {exc}")
                break
            if answers is None:
                evidence.problems[path] = "LLM reply unusable, no signal"
                continue
            evidence.by_file[path] = FileEvidence(answers=tuple(answers))
        return evidence
