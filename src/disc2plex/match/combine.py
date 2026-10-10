"""Stage 6: turn per-file confidence into a status and flag conflicts.

Only the heuristic signal exists so far; content verification (milestone 9)
will add a second signal merged here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from disc2plex.models import Assignment, AssignmentStatus, MatchResult


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


def decide(result: MatchResult, thresholds: Thresholds | None = None) -> MatchResult:
    """Set each assignment's status; demote files that claim the same episode.

    Conflicts are never resolved by picking a winner: every file involved is
    demoted to NEEDS_REVIEW (or stays UNMATCHED) with the conflict explained.
    """
    th = thresholds or Thresholds()
    claims: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, a in enumerate(result.assignments):
        for e in a.episodes:
            claims[(e.season, e.episode)].append(index)
    conflicted = {
        index: key for key, indexes in claims.items() if len(indexes) > 1 for index in indexes
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
