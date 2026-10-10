"""Tests for apply/undo. Everything happens inside pytest's tmp_path with tiny files."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from disc2plex import apply as apply_mod
from disc2plex.apply import (
    PLEXIGNORE_NAME,
    ApplyError,
    apply_manifest,
    apply_plan,
    new_manifest,
    read_manifest,
    undo_manifest,
)
from disc2plex.models import (
    AssignmentStatus,
    EntryStatus,
    FileClass,
    Manifest,
    Plan,
    PlanAction,
    PlanItem,
)


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path]:
    rip = tmp_path / "rip" / "DISC1"
    rip.mkdir(parents=True)
    library = tmp_path / "library"
    library.mkdir()
    for name in ("t00.mkv", "t01.mkv", "t02.mkv"):
        (rip / name).write_bytes(f"video data for {name}".encode())
    return {"rip": rip, "library": library, "manifests": tmp_path / "manifests", "tmp": tmp_path}


def item(source: Path, destination: Path | None, action: PlanAction = PlanAction.MOVE) -> PlanItem:
    return PlanItem(
        source=source,
        destination=destination,
        action=action,
        reason="test",
        file_class=FileClass.EPISODE,
        status=AssignmentStatus.CONFIDENT,
    )


def episode_dest(library: Path, n: int) -> Path:
    return library / "Show (2000) {tmdb-1}" / "Season 01" / f"Show (2000) - s01e{n:02d}.mkv"


def standard_plan(env: dict[str, Path], action: PlanAction = PlanAction.MOVE) -> Plan:
    rip, lib = env["rip"], env["library"]
    return Plan(
        output_root=lib,
        items=(
            item(rip / "t00.mkv", episode_dest(lib, 1), action),
            item(rip / "t01.mkv", episode_dest(lib, 2), action),
            item(rip / "t02.mkv", lib / "_review" / "DISC1" / "t02.mkv", action),
        ),
    )


def run(env: dict[str, Path], plan: Plan) -> tuple[Manifest, Path]:
    return apply_plan(plan, env["rip"], env["manifests"])


def statuses(manifest: Manifest) -> list[EntryStatus]:
    return [e.status for e in manifest.entries]


# --- apply: happy path ----------------------------------------------------------


def test_move_places_files_and_writes_manifest(env: dict[str, Path]) -> None:
    manifest, path = run(env, standard_plan(env))
    lib = env["library"]
    assert statuses(manifest) == [EntryStatus.DONE] * 3
    assert episode_dest(lib, 1).read_bytes() == b"video data for t00.mkv"
    assert not (env["rip"] / "t00.mkv").exists()
    assert (lib / "_review" / "DISC1" / "t02.mkv").exists()

    saved = read_manifest(path)
    assert saved.finished
    assert saved.entries[0].size_bytes == len(b"video data for t00.mkv")
    assert lib / "Show (2000) {tmdb-1}" in saved.created_dirs
    assert path.parent == env["manifests"]


def test_plexignore_written_in_review_folder_only(env: dict[str, Path]) -> None:
    manifest, _ = run(env, standard_plan(env))
    marker = env["library"] / "_review" / PLEXIGNORE_NAME
    assert marker.is_file()
    assert "*/*" in marker.read_text()
    assert marker in manifest.created_files
    assert not (env["library"] / PLEXIGNORE_NAME).exists()


def test_existing_plexignore_is_not_touched(env: dict[str, Path]) -> None:
    (env["library"] / "_review").mkdir()
    (env["library"] / "_review" / PLEXIGNORE_NAME).write_text("mine\n")
    manifest, _ = run(env, standard_plan(env))
    assert (env["library"] / "_review" / PLEXIGNORE_NAME).read_text() == "mine\n"
    assert manifest.created_files == []


def test_skip_items_are_not_in_manifest(env: dict[str, Path]) -> None:
    plan = Plan(
        output_root=env["library"],
        items=(
            item(env["rip"] / "t00.mkv", episode_dest(env["library"], 1)),
            item(env["rip"] / "t01.mkv", None, PlanAction.SKIP),
        ),
    )
    manifest, _ = run(env, plan)
    assert len(manifest.entries) == 1
    assert (env["rip"] / "t01.mkv").exists()


# --- apply: refusals ------------------------------------------------------------


def test_plan_with_errors_is_refused(env: dict[str, Path]) -> None:
    plan = standard_plan(env).model_copy(update={"errors": ("collision",)})
    with pytest.raises(ApplyError, match="collision"):
        run(env, plan)
    assert (env["rip"] / "t00.mkv").exists()
    assert not env["manifests"].exists()


def test_missing_output_root_is_refused(env: dict[str, Path]) -> None:
    plan = standard_plan(env).model_copy(update={"output_root": env["tmp"] / "not-mounted"})
    with pytest.raises(ApplyError, match="does not exist"):
        run(env, plan)
    assert not (env["tmp"] / "not-mounted").exists()


def test_all_skip_plan_is_refused(env: dict[str, Path]) -> None:
    plan = Plan(
        output_root=env["library"], items=(item(env["rip"] / "t00.mkv", None, PlanAction.SKIP),)
    )
    with pytest.raises(ApplyError, match="nothing to do"):
        run(env, plan)


def test_never_overwrites_existing_destination(env: dict[str, Path]) -> None:
    taken = episode_dest(env["library"], 1)
    taken.parent.mkdir(parents=True)
    taken.write_bytes(b"precious existing episode")
    manifest, _ = run(env, standard_plan(env))
    assert manifest.entries[0].status is EntryStatus.SKIPPED
    assert "not overwritten" in manifest.entries[0].message
    assert taken.read_bytes() == b"precious existing episode"
    assert (env["rip"] / "t00.mkv").exists()
    assert manifest.entries[1].status is EntryStatus.DONE


def test_destination_symlink_counts_as_existing(env: dict[str, Path]) -> None:
    taken = episode_dest(env["library"], 1)
    taken.parent.mkdir(parents=True)
    taken.symlink_to(env["tmp"] / "dangling-target")
    manifest, _ = run(env, standard_plan(env))
    assert manifest.entries[0].status is EntryStatus.SKIPPED


def test_missing_source_is_skipped(env: dict[str, Path]) -> None:
    (env["rip"] / "t01.mkv").unlink()
    manifest, _ = run(env, standard_plan(env))
    assert statuses(manifest) == [EntryStatus.DONE, EntryStatus.SKIPPED, EntryStatus.DONE]
    assert "no longer exists" in manifest.entries[1].message


def test_apply_twice_is_idempotent(env: dict[str, Path]) -> None:
    plan = standard_plan(env)
    run(env, plan)
    snapshot = {p: p.read_bytes() for p in env["library"].rglob("*.mkv")}
    second, _ = run(env, plan)
    assert set(statuses(second)) == {EntryStatus.SKIPPED}
    assert {p: p.read_bytes() for p in env["library"].rglob("*.mkv")} == snapshot


def test_destination_outside_root_fails(env: dict[str, Path]) -> None:
    escape = env["tmp"] / "elsewhere" / "evil.mkv"
    plan = Plan(output_root=env["library"], items=(item(env["rip"] / "t00.mkv", escape),))
    manifest, _ = run(env, plan)
    assert manifest.entries[0].status is EntryStatus.FAILED
    assert not escape.parent.exists()
    assert (env["rip"] / "t00.mkv").exists()


def test_symlinked_folder_cannot_escape_root(env: dict[str, Path]) -> None:
    outside = env["tmp"] / "outside"
    outside.mkdir()
    (env["library"] / "Show (2000) {tmdb-1}").symlink_to(outside, target_is_directory=True)
    manifest, _ = run(env, standard_plan(env))
    assert manifest.entries[0].status is EntryStatus.FAILED
    assert list(outside.iterdir()) == []


# --- actions and filesystem fallbacks -----------------------------------------


def test_hardlink_action(env: dict[str, Path]) -> None:
    manifest, _ = run(env, standard_plan(env, PlanAction.HARDLINK))
    src, dst = env["rip"] / "t00.mkv", episode_dest(env["library"], 1)
    assert src.exists() and os.path.samefile(src, dst)
    assert statuses(manifest) == [EntryStatus.DONE] * 3


def test_copy_action(env: dict[str, Path]) -> None:
    run(env, standard_plan(env, PlanAction.COPY))
    src, dst = env["rip"] / "t00.mkv", episode_dest(env["library"], 1)
    assert src.read_bytes() == dst.read_bytes()
    assert not os.path.samefile(src, dst)
    assert not list(dst.parent.glob(".disc2plex-partial-*"))


def _fail_link(code: int):  # type: ignore[no-untyped-def]
    def fake_link(src: object, dst: object) -> None:
        raise OSError(code, os.strerror(code))

    return fake_link


def test_move_without_hardlink_support_falls_back_to_rename(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(apply_mod.os, "link", _fail_link(errno.EPERM))
    manifest, _ = run(env, standard_plan(env))
    assert statuses(manifest) == [EntryStatus.DONE] * 3
    assert episode_dest(env["library"], 1).read_bytes() == b"video data for t00.mkv"


def test_rename_fallback_still_never_overwrites(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(apply_mod.os, "link", _fail_link(errno.EPERM))
    taken = episode_dest(env["library"], 1)
    taken.parent.mkdir(parents=True)
    taken.write_bytes(b"keep me")
    # Simulate the destination appearing between the pre-check and the move.
    original = apply_mod._apply_entry

    def racing(entry, manifest):  # type: ignore[no-untyped-def]
        apply_mod.place_file(entry.source, entry.destination, entry.action)

    monkeypatch.setattr(apply_mod, "_apply_entry", racing)
    manifest, _ = run(env, standard_plan(env))
    assert manifest.entries[0].status is EntryStatus.SKIPPED
    assert taken.read_bytes() == b"keep me"
    monkeypatch.setattr(apply_mod, "_apply_entry", original)


def test_cross_device_move_copies_then_removes_source(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_link = os.link

    def link(src: str | Path, dst: str | Path) -> None:
        # Linking from the rip folder "crosses devices"; staging links work.
        if Path(src).parent == env["rip"]:
            raise OSError(errno.EXDEV, "cross-device link")
        real_link(src, dst)

    monkeypatch.setattr(apply_mod.os, "link", link)
    manifest, _ = run(env, standard_plan(env))
    assert statuses(manifest) == [EntryStatus.DONE] * 3
    assert episode_dest(env["library"], 1).read_bytes() == b"video data for t00.mkv"
    assert not (env["rip"] / "t00.mkv").exists()


def test_move_rolls_back_if_source_cannot_be_removed(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_unlink = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.parent == env["rip"]:
            raise PermissionError(errno.EACCES, "read-only rip folder")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    manifest, _ = run(env, standard_plan(env))
    assert set(statuses(manifest)) == {EntryStatus.FAILED}
    assert (env["rip"] / "t00.mkv").read_bytes() == b"video data for t00.mkv"
    assert not episode_dest(env["library"], 1).exists()


# --- manifest durability ------------------------------------------------------


def test_manifest_saved_before_and_during_run(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = standard_plan(env)
    manifest = new_manifest(plan, env["rip"])
    path = env["manifests"] / "m.json"
    seen: list[list[EntryStatus]] = []
    real_place = apply_mod.place_file

    def place(source: Path, destination: Path, action: PlanAction) -> None:
        seen.append([e.status for e in read_manifest(path).entries])
        real_place(source, destination, action)

    monkeypatch.setattr(apply_mod, "place_file", place)
    apply_manifest(manifest, path)
    assert seen[0] == [EntryStatus.PENDING] * 3
    assert seen[2] == [EntryStatus.DONE, EntryStatus.DONE, EntryStatus.PENDING]


def test_interrupted_run_can_be_undone(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = standard_plan(env)
    manifest = new_manifest(plan, env["rip"])
    path = env["manifests"] / "m.json"
    real_place = apply_mod.place_file
    calls = {"n": 0}

    def place(source: Path, destination: Path, action: PlanAction) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        real_place(source, destination, action)

    monkeypatch.setattr(apply_mod, "place_file", place)
    with pytest.raises(KeyboardInterrupt):
        apply_manifest(manifest, path)

    saved = read_manifest(path)
    assert statuses(saved) == [EntryStatus.DONE, EntryStatus.PENDING, EntryStatus.PENDING]
    undo_manifest(saved, path)
    assert (env["rip"] / "t00.mkv").read_bytes() == b"video data for t00.mkv"
    assert not (env["library"] / "Show (2000) {tmdb-1}").exists()


# --- undo -----------------------------------------------------------------------


def test_undo_restores_everything(env: dict[str, Path]) -> None:
    before = {p.name: p.read_bytes() for p in env["rip"].iterdir()}
    _, path = run(env, standard_plan(env))
    undone = undo_manifest(read_manifest(path), path)
    assert {p.name: p.read_bytes() for p in env["rip"].iterdir()} == before
    assert list(env["library"].iterdir()) == []  # created dirs and .plexignore removed
    assert set(statuses(undone)) == {EntryStatus.UNDONE}
    assert read_manifest(path).undone_at is not None


def test_undo_twice_is_harmless(env: dict[str, Path]) -> None:
    _, path = run(env, standard_plan(env))
    undo_manifest(read_manifest(path), path)
    second = undo_manifest(read_manifest(path), path)
    assert set(statuses(second)) == {EntryStatus.UNDONE}
    assert len(list(env["rip"].iterdir())) == 3


def test_undo_never_overwrites_original_location(env: dict[str, Path]) -> None:
    _, path = run(env, standard_plan(env))
    (env["rip"] / "t00.mkv").write_bytes(b"new rip of the same title")
    undone = undo_manifest(read_manifest(path), path)
    entry = undone.entries[0]
    assert entry.status is EntryStatus.DONE
    assert "undo refused" in entry.message
    assert (env["rip"] / "t00.mkv").read_bytes() == b"new rip of the same title"
    assert episode_dest(env["library"], 1).exists()  # left in place, nothing lost


def test_undo_keeps_directories_that_gained_other_files(env: dict[str, Path]) -> None:
    _, path = run(env, standard_plan(env))
    season = episode_dest(env["library"], 1).parent
    (season / "someone else's file.mkv").write_bytes(b"x")
    undo_manifest(read_manifest(path), path)
    assert (season / "someone else's file.mkv").exists()


def test_undo_hardlink_removes_only_the_link(env: dict[str, Path]) -> None:
    _, path = run(env, standard_plan(env, PlanAction.HARDLINK))
    undo_manifest(read_manifest(path), path)
    assert (env["rip"] / "t00.mkv").read_bytes() == b"video data for t00.mkv"
    assert not episode_dest(env["library"], 1).exists()


def test_undo_copy_refuses_if_copy_changed(env: dict[str, Path]) -> None:
    _, path = run(env, standard_plan(env, PlanAction.COPY))
    dst = episode_dest(env["library"], 1)
    dst.write_bytes(b"edited after apply, different size")
    undone = undo_manifest(read_manifest(path), path)
    assert "undo refused" in undone.entries[0].message
    assert dst.exists()


def test_undo_copy_refuses_if_original_missing(env: dict[str, Path]) -> None:
    _, path = run(env, standard_plan(env, PlanAction.COPY))
    (env["rip"] / "t00.mkv").unlink()
    undone = undo_manifest(read_manifest(path), path)
    assert "original file is missing" in undone.entries[0].message
    assert episode_dest(env["library"], 1).exists()


def test_read_manifest_rejects_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ApplyError, match="cannot read manifest"):
        read_manifest(bad)
