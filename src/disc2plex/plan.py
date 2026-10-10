"""Stage 7: turn decided assignments into a rename/move plan.

Pure: no filesystem access except through the injected ``exists`` callback, so
the CLI can mark destinations that are already taken.

Where files go:

- confident episodes -> ``<output>/<Show>/Season NN/<Show> - sNNeNN - Title.ext``
- episodes needing review, unmatched episodes, unknowns -> ``<output>/_review/<disc>/``
  (or left in the rip folder with ``review=leave``)
- extras -> ``<output>/_extras_review/<disc>/`` (default), or skipped
- Play-All and junk -> skipped (left where they are; nothing is ever deleted)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from disc2plex.models import (
    Assignment,
    AssignmentStatus,
    FileClass,
    MatchResult,
    Plan,
    PlanAction,
    PlanItem,
    ShowInfo,
)
from disc2plex.naming import ensure_within, episode_path, sanitize_component

REVIEW_DIR = "_review"
EXTRAS_REVIEW_DIR = "_extras_review"


class ExtrasMode(StrEnum):
    REVIEW = "review"
    SEASON0 = "season0"
    SKIP = "skip"


class ReviewMode(StrEnum):
    MOVE = "move"  # to <output>/_review/<disc>/
    LEAVE = "leave"  # stay in the rip folder


@dataclass(frozen=True)
class PlanOptions:
    action: PlanAction = PlanAction.MOVE
    include_titles: bool = True
    include_tmdb_id: bool = True
    extras: ExtrasMode = ExtrasMode.REVIEW
    review: ReviewMode = ReviewMode.MOVE


def _review_destination(output_root: Path, folder: str, disc_name: str, source: Path) -> Path:
    return output_root / folder / disc_name / sanitize_component(source.name)


def _library_destination(
    a: Assignment, show: ShowInfo, output_root: Path, options: PlanOptions
) -> Path | str:
    """Library path for a confident episode, or a reason string if it can't be named."""
    seasons = {e.season for e in a.episodes}
    numbers = sorted(e.episode for e in a.episodes)
    if len(seasons) > 1 or numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        return (
            f"episodes {a.episode_label} are not consecutive in aired numbering, "
            "so Plex can't name them as one file"
        )
    return episode_path(
        output_root,
        show.name,
        show.year,
        show.tmdb_id,
        seasons.pop(),
        numbers,
        [e.title for e in a.episodes],
        a.file.path.suffix,
        include_tmdb_id=options.include_tmdb_id,
        include_title=options.include_titles,
    )


def _item_for(
    a: Assignment, show: ShowInfo, output_root: Path, disc_name: str, options: PlanOptions
) -> PlanItem:
    def item(destination: Path | None, action: PlanAction, reason: str) -> PlanItem:
        return PlanItem(
            source=a.file.path,
            destination=destination,
            action=action,
            reason=reason,
            file_class=a.file_class,
            status=a.status,
            confidence=a.confidence,
            episode_label=a.episode_label,
        )

    def review(reason: str, folder: str = REVIEW_DIR) -> PlanItem:
        if folder == REVIEW_DIR and options.review is ReviewMode.LEAVE:
            return item(None, PlanAction.SKIP, f"{reason}; left in place (review: leave)")
        dest = _review_destination(output_root, folder, disc_name, a.file.path)
        return item(dest, options.action, reason)

    if a.file_class in (FileClass.PLAY_ALL, FileClass.JUNK):
        return item(None, PlanAction.SKIP, f"{a.file_class.value}: left in place")
    if a.file_class is FileClass.EXTRA:
        if options.extras is ExtrasMode.SKIP:
            return item(None, PlanAction.SKIP, "extra: left in place (--extras skip)")
        if options.extras is ExtrasMode.SEASON0:
            # Mapping extras onto TMDB specials needs content matching (milestone 9).
            return review("extra: Season 00 mapping not available yet", EXTRAS_REVIEW_DIR)
        return review("extra", EXTRAS_REVIEW_DIR)
    if a.file_class is FileClass.UNKNOWN:
        return review("unknown title type")
    if a.status is AssignmentStatus.CONFIDENT:
        destination = _library_destination(a, show, output_root, options)
        if isinstance(destination, str):
            return review(destination)
        return item(destination, options.action, f"matched {a.episode_label}")
    if a.status is AssignmentStatus.NEEDS_REVIEW:
        return review(f"needs review: best guess {a.episode_label} ({a.confidence:.0%})")
    return review("unmatched episode")


def build_plan(
    match: MatchResult,
    show: ShowInfo,
    output_root: Path,
    disc_name: str,
    options: PlanOptions | None = None,
    exists: Callable[[Path], bool] = lambda _p: False,
) -> Plan:
    """Build the plan. Refuses (via ``Plan.errors``) if two files share a destination."""
    opts = options or PlanOptions()
    safe_disc = sanitize_component(disc_name, fallback="disc")
    items = [_item_for(a, show, output_root, safe_disc, opts) for a in match.assignments]
    items.sort(key=lambda i: i.source.name.lower())

    errors: list[str] = []
    warnings = list(match.notes)
    for i in items:
        if i.destination is not None:
            ensure_within(output_root, i.destination)

    counts = Counter(i.destination for i in items if i.destination is not None)
    for destination, count in counts.items():
        if count > 1:
            sources = ", ".join(i.source.name for i in items if i.destination == destination)
            errors.append(f"{count} files would be written to {destination}: {sources}")

    final: list[PlanItem] = []
    for i in items:
        if i.destination is not None and i.destination == i.source:
            # e.g. re-running on <output>/_review/<disc>/ itself
            i = i.model_copy(
                update={"action": PlanAction.SKIP, "reason": f"{i.reason}; already there"}
            )
        elif i.destination is not None and exists(i.destination):
            warnings.append(f"{i.destination} already exists; {i.source.name} will be skipped")
            i = i.model_copy(
                update={"action": PlanAction.SKIP, "reason": f"{i.reason}; destination exists"}
            )
        final.append(i)
    return Plan(
        output_root=output_root,
        items=tuple(final),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
