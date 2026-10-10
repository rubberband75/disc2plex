"""Stage 8: execute a plan safely, record a manifest, and undo it.

Safety rules (CLAUDE.md section 9), enforced here:

- Never overwrite: every placement is no-clobber (hardlink-then-unlink, which
  fails atomically if the destination exists), with a checked rename only as
  a fallback on filesystems without hardlinks.
- Never delete user data: a move removes the source only after the
  destination is complete; undo removes copies/links only when the original
  still exists and the copy is unchanged.
- Never modify file contents.
- Every destination is re-checked against the output root with symlinks resolved.
- The manifest is written before the first operation and after each one.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from disc2plex import __version__
from disc2plex.config import data_dir
from disc2plex.models import EntryStatus, Manifest, ManifestEntry, Plan, PlanAction
from disc2plex.naming import ensure_within, sanitize_component
from disc2plex.plan import EXTRAS_REVIEW_DIR, REVIEW_DIR

log = logging.getLogger(__name__)

PLEXIGNORE_NAME = ".plexignore"
# Patterns without a slash only match files directly in this folder; review
# files live one level down (<review>/<disc>/file.mkv), hence "*/*".
PLEXIGNORE_TEXT = (
    "# Created by disc2plex: keep files awaiting manual review out of Plex.\n*\n*/*\n*/*/*\n"
)
_TEMP_PREFIX = ".disc2plex-partial-"
# Filesystems (some SMB/NFS mounts) that refuse hardlinks report one of these.
_NO_LINK_ERRNOS = {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EMLINK, errno.ENOSYS}


class ApplyError(Exception):
    """The plan cannot be applied (it has errors, or nothing safe to do)."""


def default_manifest_dir() -> Path:
    return data_dir() / "manifests"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- manifest I/O -------------------------------------------------------------


def write_manifest(manifest: Manifest, path: Path) -> None:
    """Write atomically so a crash never leaves a half-written manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def read_manifest(path: Path) -> Manifest:
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ApplyError(f"cannot read manifest {path}: {exc}") from exc


def new_manifest(
    plan: Plan,
    rip_dir: Path,
    args: dict[str, str | int | float | bool | None] | None = None,
    run_id: str | None = None,
) -> Manifest:
    """Build a manifest with every non-skip plan item as a pending entry."""
    return Manifest(
        run_id=run_id or uuid.uuid4().hex[:12],
        created_at=_now(),
        tool_version=__version__,
        rip_dir=rip_dir,
        output_root=plan.output_root,
        args=args or {},
        entries=[
            ManifestEntry(source=i.source, destination=i.destination, action=i.action)
            for i in plan.items
            if i.action is not PlanAction.SKIP and i.destination is not None
        ],
    )


def manifest_path_for(manifest: Manifest, directory: Path) -> Path:
    stamp = manifest.created_at.replace(":", "").replace("-", "")[:15]
    disc = sanitize_component(manifest.rip_dir.name, fallback="disc")
    return directory / f"{stamp}-{disc}-{manifest.run_id}.json"


# --- low-level file operations --------------------------------------------------


def _check_contained(root: Path, destination: Path) -> None:
    """Re-check containment with symlinks resolved (a symlinked folder can't escape)."""
    ensure_within(Path(os.path.realpath(root)), Path(os.path.realpath(destination)))


def _make_parents(directory: Path, root: Path, created: list[Path]) -> None:
    """mkdir -p, recording each directory we create (outermost first)."""
    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for path in reversed(missing):
        _check_contained(root, path / "x")
        path.mkdir()
        created.append(path)


def _place_no_clobber(staged: Path, destination: Path) -> None:
    """Give ``staged`` the name ``destination`` without ever replacing an existing file."""
    try:
        os.link(staged, destination)
    except FileExistsError:
        raise
    except OSError as exc:
        if exc.errno not in _NO_LINK_ERRNOS:
            raise
        # No hardlinks here: check-then-rename. A tiny race window remains, which
        # is acceptable for a single-user tool; the check makes the common case safe.
        if destination.exists():
            raise FileExistsError(errno.EEXIST, "destination exists", str(destination)) from exc
        os.rename(staged, destination)
        return
    staged.unlink()


def _copy_then_place(source: Path, destination: Path) -> None:
    """Copy to a temp file beside the destination, verify size, then place it."""
    temp = destination.with_name(f"{_TEMP_PREFIX}{uuid.uuid4().hex[:8]}-{destination.name}")
    try:
        shutil.copy2(source, temp)
        if temp.stat().st_size != source.stat().st_size:
            raise OSError(f"copy of {source} is incomplete")
        _place_no_clobber(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def _remove_source_or_roll_back(source: Path, destination: Path) -> None:
    """Finish a move by removing the source; if that fails, undo the placement.

    The destination is either a hardlink to the source or a verified copy, so
    removing it on failure loses nothing.
    """
    try:
        source.unlink()
    except OSError:
        destination.unlink()
        raise


def move_file(source: Path, destination: Path) -> None:
    """Move without overwriting. Same filesystem: link + unlink (atomic no-clobber)."""
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            _copy_then_place(source, destination)
            _remove_source_or_roll_back(source, destination)
            return
        if exc.errno not in _NO_LINK_ERRNOS:
            raise
        if destination.exists():
            raise FileExistsError(errno.EEXIST, "destination exists", str(destination)) from exc
        try:
            os.rename(source, destination)
        except OSError as rename_exc:
            if rename_exc.errno != errno.EXDEV:
                raise
            _copy_then_place(source, destination)
            _remove_source_or_roll_back(source, destination)
        return
    _remove_source_or_roll_back(source, destination)


def place_file(source: Path, destination: Path, action: PlanAction) -> None:
    if action is PlanAction.MOVE:
        move_file(source, destination)
    elif action is PlanAction.HARDLINK:
        os.link(source, destination)  # raises FileExistsError if taken
    elif action is PlanAction.COPY:
        _copy_then_place(source, destination)
    else:
        raise ValueError(f"cannot place a file with action {action}")


# --- apply --------------------------------------------------------------------


def _write_plexignore(directory: Path, manifest: Manifest) -> None:
    marker = directory / PLEXIGNORE_NAME
    if marker.exists():
        return
    try:
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(PLEXIGNORE_TEXT)
    except FileExistsError:
        return
    manifest.created_files.append(marker)


def apply_manifest(
    manifest: Manifest,
    manifest_path: Path,
    *,
    on_progress: Callable[[ManifestEntry], None] | None = None,
) -> Manifest:
    """Execute every pending entry, saving the manifest before and after each step."""
    root = manifest.output_root
    write_manifest(manifest, manifest_path)
    for entry in manifest.entries:
        if entry.status is not EntryStatus.PENDING:
            continue
        try:
            _apply_entry(entry, manifest)
        except FileExistsError:
            entry.status = EntryStatus.SKIPPED
            entry.message = "destination already exists; not overwritten"
        except (OSError, ValueError) as exc:
            entry.status = EntryStatus.FAILED
            entry.message = str(exc)
            log.warning("failed %s -> %s: %s", entry.source, entry.destination, exc)
        write_manifest(manifest, manifest_path)
        if on_progress:
            on_progress(entry)
    for review in (REVIEW_DIR, EXTRAS_REVIEW_DIR):
        if (root / review).is_dir() and any(
            e.status is EntryStatus.DONE and (root / review) in e.destination.parents
            for e in manifest.entries
        ):
            _write_plexignore(root / review, manifest)
    manifest.finished = True
    write_manifest(manifest, manifest_path)
    return manifest


def _apply_entry(entry: ManifestEntry, manifest: Manifest) -> None:
    source, destination = entry.source, entry.destination
    if not source.is_file():
        entry.status = EntryStatus.SKIPPED
        entry.message = "source no longer exists (already moved?)"
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(errno.EEXIST, "destination exists", str(destination))
    _check_contained(manifest.output_root, destination)
    _make_parents(destination.parent, manifest.output_root, manifest.created_dirs)
    place_file(source, destination, entry.action)
    stat = destination.stat()
    entry.size_bytes = stat.st_size
    entry.inode = stat.st_ino
    entry.status = EntryStatus.DONE
    entry.message = ""
    log.info("%s %s -> %s", entry.action.value, source, destination)


def apply_plan(
    plan: Plan,
    rip_dir: Path,
    manifest_dir: Path,
    *,
    args: dict[str, str | int | float | bool | None] | None = None,
    on_progress: Callable[[ManifestEntry], None] | None = None,
) -> tuple[Manifest, Path]:
    """Validate, write the manifest, and execute the plan. Returns (manifest, path)."""
    if plan.errors:
        raise ApplyError("plan has errors and cannot be applied:\n" + "\n".join(plan.errors))
    if not plan.output_root.is_dir():
        # Refuse rather than create it: a typo or an unmounted network share
        # would otherwise silently fill the local disk.
        raise ApplyError(
            f"output root {plan.output_root} does not exist (typo, or share not mounted?)"
        )
    manifest = new_manifest(plan, rip_dir, args)
    path = manifest_path_for(manifest, manifest_dir)
    if not manifest.entries:
        raise ApplyError("nothing to do: every item is skipped")
    return apply_manifest(manifest, path, on_progress=on_progress), path


# --- undo ---------------------------------------------------------------------


def _same_file(entry: ManifestEntry, path: Path) -> bool:
    """True if ``path`` still looks like the file this run placed (size + inode)."""
    try:
        stat = path.stat()
    except OSError:
        return False
    return stat.st_size == entry.size_bytes and stat.st_ino == entry.inode


def _undo_entry(entry: ManifestEntry) -> str:
    """Reverse one entry. Returns a message; raises on refusal."""
    source, destination = entry.source, entry.destination
    if entry.action is PlanAction.MOVE:
        if not destination.exists():
            raise OSError("moved file is no longer at its destination")
        if source.exists():
            raise OSError("something now exists at the original location")
        source.parent.mkdir(parents=True, exist_ok=True)
        move_file(destination, source)
        return "moved back"
    # Copies and hardlinks: the original must still be intact before we remove ours.
    if not source.is_file():
        raise OSError("original file is missing; keeping the copy")
    if not destination.exists():
        return "already removed"
    if not _same_file(entry, destination):
        raise OSError("file at destination changed since apply; leaving it")
    if entry.action is PlanAction.HARDLINK and not os.path.samefile(source, destination):
        raise OSError("destination is no longer a link to the original; leaving it")
    if entry.action is PlanAction.COPY and source.stat().st_size != destination.stat().st_size:
        raise OSError("copy no longer matches the original; leaving it")
    destination.unlink()
    return f"removed {entry.action.value}"


def undo_manifest(
    manifest: Manifest,
    manifest_path: Path,
    *,
    on_progress: Callable[[ManifestEntry], None] | None = None,
) -> Manifest:
    """Reverse a previous apply, newest operation first, then tidy what we created."""
    for entry in reversed(manifest.entries):
        if entry.status is not EntryStatus.DONE:
            continue
        try:
            entry.message = _undo_entry(entry)
            entry.status = EntryStatus.UNDONE
        except (OSError, ValueError) as exc:
            entry.message = f"undo refused: {exc}"
            log.warning("undo refused for %s: %s", entry.destination, exc)
        write_manifest(manifest, manifest_path)
        if on_progress:
            on_progress(entry)

    for marker in manifest.created_files:
        if marker.is_file() and marker.read_text(encoding="utf-8") == PLEXIGNORE_TEXT:
            marker.unlink()
    for directory in sorted(manifest.created_dirs, key=lambda p: len(p.parts), reverse=True):
        with contextlib.suppress(OSError):
            directory.rmdir()  # only succeeds if empty
    manifest.undone_at = _now()
    write_manifest(manifest, manifest_path)
    return manifest
