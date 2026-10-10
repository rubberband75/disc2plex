"""Stage 6: turn per-file confidence into a status and flag conflicts.

Also applies user overrides (``--assign``): a file the user identified is
placed as they said, and any other file claiming those episodes goes to review.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from disc2plex.match.heuristic import AMBIGUOUS_START_NOTE
from disc2plex.models import Assignment, AssignmentStatus, EpisodeInfo, FileClass, MatchResult

ASSIGNED_SIGNAL = "assigned"


@dataclass(frozen=True)
class Thresholds:
    confident: float = 0.85
    review: float = 0.5


def status_for(assignment: Assignment, thresholds: Thresholds) -> AssignmentStatus:
    if not assignment.episodes:
        return AssignmentStatus.UNMATCHED
    if assignment.confidence >= thresholds.confident:
        return AssignmentStatus.CONFIDENT
    if assignment.confidence >= thresholds.review:
        return AssignmentStatus.NEEDS_REVIEW
    return AssignmentStatus.UNMATCHED


def apply_overrides(
    result: MatchResult, overrides: Mapping[Path, Sequence[EpisodeInfo]]
) -> MatchResult:
    """Replace the match for each overridden file with the user's answer (confidence 1)."""
    if not overrides:
        return result
    updated = []
    for a in result.assignments:
        episodes = overrides.get(a.file.path)
        if episodes is None:
            updated.append(a)
            continue
        assigned = Assignment(
            file=a.file,
            file_class=FileClass.EPISODE,
            episodes=tuple(episodes),
            confidence=1.0,
            signals={ASSIGNED_SIGNAL: 1.0},
            reasons=(),
        )
        was = f" (matching suggested {a.episode_label})" if a.episode_label else ""
        reason = f"assigned {assigned.episode_label} with --assign{was}"
        updated.append(assigned.model_copy(update={"reasons": (reason,)}))
    notes = result.notes
    if all(ASSIGNED_SIGNAL in a.signals for a in updated if a.episodes):
        # Every matched file was identified by the user; where the disc starts no longer matters.
        notes = tuple(n for n in notes if n != AMBIGUOUS_START_NOTE)
    return result.model_copy(update={"assignments": tuple(updated), "notes": notes})


def decide(result: MatchResult, thresholds: Thresholds | None = None) -> MatchResult:
    """Set each assignment's status; demote files that claim the same episode.

    Conflicts are never resolved by picking a winner, except that a file the
    user assigned (``--assign``) keeps its episodes: every other file involved
    is demoted to NEEDS_REVIEW (or stays UNMATCHED) with the conflict explained.
    """
    th = thresholds or Thresholds()
    claims: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, a in enumerate(result.assignments):
        for e in a.episodes:
            claims[(e.season, e.episode)].append(index)
    pinned = {index for index, a in enumerate(result.assignments) if ASSIGNED_SIGNAL in a.signals}
    conflicted = {
        index: key
        for key, indexes in claims.items()
        if len(indexes) > 1
        for index in indexes
        if index not in pinned
    }

    decided: list[Assignment] = []
    notes = list(result.notes)
    for index, a in enumerate(result.assignments):
        status = status_for(a, th)
        reasons = a.reasons
        if index in conflicted:
            season, episode = conflicted[index]
            reasons = (
                *reasons,
                f"conflict: another file is also matched to s{season:02d}e{episode:02d}",
            )
            if status is AssignmentStatus.CONFIDENT:
                status = AssignmentStatus.NEEDS_REVIEW
        decided.append(a.model_copy(update={"status": status, "reasons": reasons}))
    if conflicted:
        notes.append("some episodes are claimed by more than one file; see review items")
    return result.model_copy(update={"assignments": tuple(decided), "notes": tuple(notes)})
