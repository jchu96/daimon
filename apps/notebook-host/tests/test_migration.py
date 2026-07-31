"""Tests for notebook_host.migration — the crash-safe flat-to-nested boot migration.

Every interrupted intermediate filesystem state is constructed directly in
``tmp_path`` rather than by actually killing a process, so these tests are
deterministic and don't race a real SIGKILL. Each proves the migration
converges to the fully-migrated shape from that state, per the plan's
known-risk note that the algorithm was reasoned, not reproduced, at planning
time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from notebook_host.jail import UidPoolExhaustedError
from notebook_host.migration import migrate_flat_layout

_UID_START = 100000
_UID_END = 100999


def _uids_file(tmp_path: Path) -> Path:
    return tmp_path / "uids.json"


def _migrate(tmp_path: Path, *, uid_start: int = _UID_START, uid_end: int = _UID_END) -> list[str]:
    return migrate_flat_layout(
        tmp_path,
        uids_file=_uids_file(tmp_path),
        uid_start=uid_start,
        uid_end=uid_end,
        allow_unjailed=True,
    )


def _assert_no_leftovers(tmp_path: Path) -> None:
    assert list(tmp_path.glob(".migrating-*")) == [], (
        "no staging directory should survive a converged migration"
    )
    assert list(tmp_path.glob("*.py")) == [], "no flat source should survive a converged migration"
    assert list(tmp_path.glob("*_workspace")) == [], (
        "no flat disposable workspace should survive a converged migration"
    )
    assert list(tmp_path.glob("*.marimo.log")) == [], (
        "no flat disposable log should survive a converged migration"
    )


# ─── state 1: untouched flat tree ───────────────────────────────────────────


def test_migrate_untouched_flat_tree_converges_and_second_call_is_noop(tmp_path: Path) -> None:
    """The ordinary case: a full flat layout moves onto the nested shape in one call."""
    (tmp_path / "a.py").write_bytes(b"import marimo as mo\napp = mo.App()")
    (tmp_path / "a.data").mkdir()
    (tmp_path / "a.data" / "x.csv").write_bytes(b"col1,col2\n1,2\n")
    (tmp_path / "a_workspace").mkdir()
    (tmp_path / "a_workspace" / "junk").write_bytes(b"disposable")
    (tmp_path / "a.marimo.log").write_bytes(b"old log")
    blogs_payload = {"a": {"slug": "a", "created_at": 1.0}}
    pids_payload: dict[str, object] = {}
    (tmp_path / "blogs.json").write_text(json.dumps(blogs_payload))
    (tmp_path / "pids.json").write_text(json.dumps(pids_payload))

    migrated = _migrate(tmp_path)

    assert migrated == ["a"], "the one legacy slug must be reported as migrated"
    assert (
        tmp_path / "a" / "notebook.py"
    ).read_bytes() == b"import marimo as mo\napp = mo.App()", (
        "the notebook source must survive the move with identical bytes"
    )
    assert (tmp_path / "a" / "data" / "x.csv").read_bytes() == b"col1,col2\n1,2\n", (
        "the attachment must survive the move with identical bytes"
    )
    assert (tmp_path / "a" / "workspace").is_dir(), "workspace must exist in the nested tree"
    assert (tmp_path / "a" / "home").is_dir(), "home must exist in the nested tree"
    assert not (tmp_path / "a.py").exists(), "the flat source must be gone"
    assert not (tmp_path / "a.data").exists(), "the flat attachment dir must be gone"
    assert not (tmp_path / "a_workspace").exists(), "the flat disposable workspace must be gone"
    assert not (tmp_path / "a.marimo.log").exists(), "the flat disposable log must be gone"
    assert json.loads((tmp_path / "blogs.json").read_text()) == blogs_payload, (
        "blogs.json must never be treated as a slug and must be left byte-identical"
    )
    assert json.loads((tmp_path / "pids.json").read_text()) == pids_payload, (
        "pids.json must never be treated as a slug and must be left byte-identical"
    )
    assert tmp_path.stat().st_mode & 0o777 == 0o700, "data_dir itself must be locked to 0700"
    for d in (
        tmp_path / "a",
        tmp_path / "a" / "data",
        tmp_path / "a" / "workspace",
        tmp_path / "a" / "home",
    ):
        assert d.stat().st_mode & 0o777 == 0o700, f"{d} must be mode 0700"
    _assert_no_leftovers(tmp_path)

    second = _migrate(tmp_path)
    assert second == [], "a second call on an already-migrated tree must be a no-op"
    _assert_no_leftovers(tmp_path)


# ─── state 2: killed after staging.mkdir, before anything moved ────────────


def test_migrate_resumes_staging_created_but_empty(tmp_path: Path) -> None:
    """staging/ exists and is empty; the flat source and attachments are untouched.

    Pass 1 must leave the empty staging dir alone; pass 2 must complete it.
    """
    (tmp_path / ".migrating-a").mkdir()
    (tmp_path / "a.py").write_bytes(b"source bytes")
    (tmp_path / "a.data").mkdir()
    (tmp_path / "a.data" / "x.csv").write_bytes(b"attachment bytes")

    migrated = _migrate(tmp_path)

    assert migrated == ["a"], "the slug must converge and be reported as migrated"
    assert (tmp_path / "a" / "notebook.py").read_bytes() == b"source bytes", (
        "the source must end up in the final tree with identical bytes"
    )
    assert (tmp_path / "a" / "data" / "x.csv").read_bytes() == b"attachment bytes", (
        "the attachment must end up in the final tree with identical bytes"
    )
    _assert_no_leftovers(tmp_path)


# ─── state 3: killed after data moved in, before the source moved ──────────


def test_migrate_resumes_staging_with_data_moved_but_source_still_flat(tmp_path: Path) -> None:
    """staging/data/ already holds the attachment; a.py is still flat; a.data is gone.

    This is the state that would lose the attachment if pass 1 deleted a
    staging directory lacking notebook.py.
    """
    staging = tmp_path / ".migrating-a"
    staging.mkdir()
    (staging / "data").mkdir()
    (staging / "data" / "x.csv").write_bytes(b"attachment bytes")
    (tmp_path / "a.py").write_bytes(b"source bytes")
    # a.data does NOT exist — it already moved into staging/data.

    migrated = _migrate(tmp_path)

    assert migrated == ["a"], "the slug must converge and be reported as migrated"
    assert (tmp_path / "a" / "data" / "x.csv").read_bytes() == b"attachment bytes", (
        "the attachment must survive at a/data/x.csv — this is the case that loses data "
        "if pass 1 ever deletes a staging dir lacking notebook.py"
    )
    assert (tmp_path / "a" / "notebook.py").read_bytes() == b"source bytes", (
        "the source must be moved in and end up in the final tree"
    )
    _assert_no_leftovers(tmp_path)


# ─── state 4: killed after the source moved in, before the promote rename ──


def test_migrate_promotes_staging_with_source_already_moved_in(tmp_path: Path) -> None:
    """staging/{notebook.py,data/,workspace/,home/} all exist; nothing flat remains.

    Pass 1 must promote this staging directory by rename, without pass 2
    ever seeing a.py again.
    """
    staging = tmp_path / ".migrating-a"
    staging.mkdir()
    (staging / "notebook.py").write_bytes(b"source bytes")
    (staging / "data").mkdir()
    (staging / "workspace").mkdir()
    (staging / "home").mkdir()

    migrated = _migrate(tmp_path)

    assert migrated == ["a"], "the slug must be reported as migrated via the pass-1 promote path"
    assert (tmp_path / "a" / "notebook.py").read_bytes() == b"source bytes", (
        "the promoted notebook must have the original bytes"
    )
    _assert_no_leftovers(tmp_path)


# ─── state 5: killed after the promote, before pass 3 ──────────────────────


def test_migrate_cleans_disposable_leftovers_after_final_tree_already_promoted(
    tmp_path: Path,
) -> None:
    """a/ is already fully promoted; a_workspace/ and a.marimo.log are still flat.

    Both flat leftovers must be removed and the already-promoted tree left
    untouched.
    """
    final = tmp_path / "a"
    final.mkdir()
    (final / "notebook.py").write_bytes(b"already promoted")
    (final / "data").mkdir()
    (final / "workspace").mkdir()
    (final / "home").mkdir()
    (tmp_path / "a_workspace").mkdir()
    (tmp_path / "a_workspace" / "junk").write_bytes(b"disposable")
    (tmp_path / "a.marimo.log").write_bytes(b"old log")

    migrated = _migrate(tmp_path)

    assert migrated == [], "an already-promoted slug is not re-reported as migrated"
    assert (final / "notebook.py").read_bytes() == b"already promoted", (
        "the already-promoted tree must be untouched"
    )
    assert not (tmp_path / "a_workspace").exists(), "the flat disposable workspace must be removed"
    assert not (tmp_path / "a.marimo.log").exists(), "the flat disposable log must be removed"
    _assert_no_leftovers(tmp_path)


# ─── state 6: both a staging dir and a completed final dir exist ───────────


def test_migrate_discards_staging_copy_when_final_tree_already_exists(tmp_path: Path) -> None:
    """Both .migrating-a/ and a/ exist. Pass 1 discards the staging copy.

    The final tree's notebook.py bytes must be the ones that were already in
    a/, not the (different) ones in the staging copy.
    """
    final = tmp_path / "a"
    final.mkdir()
    (final / "notebook.py").write_bytes(b"winner bytes")
    (final / "data").mkdir()
    (final / "workspace").mkdir()
    (final / "home").mkdir()

    staging = tmp_path / ".migrating-a"
    staging.mkdir()
    (staging / "notebook.py").write_bytes(b"stale garbage bytes")

    migrated = _migrate(tmp_path)

    assert migrated == [], "the already-final slug must not be re-reported as migrated"
    assert (final / "notebook.py").read_bytes() == b"winner bytes", (
        "the final tree must win the race; the staging copy is garbage"
    )
    _assert_no_leftovers(tmp_path)


# ─── state 7: two slugs at different stages simultaneously ─────────────────


def test_migrate_converges_two_slugs_at_different_stages_in_one_call(tmp_path: Path) -> None:
    """slug 'a' is untouched flat; slug 'b' is mid-staging. Both converge in one call."""
    (tmp_path / "a.py").write_bytes(b"a source")

    staging_b = tmp_path / ".migrating-b"
    staging_b.mkdir()
    (staging_b / "notebook.py").write_bytes(b"b source")
    (staging_b / "data").mkdir()
    (staging_b / "workspace").mkdir()
    (staging_b / "home").mkdir()

    migrated = _migrate(tmp_path)

    assert set(migrated) == {"a", "b"}, "both slugs must converge and be reported as migrated"
    assert (tmp_path / "a" / "notebook.py").read_bytes() == b"a source"
    assert (tmp_path / "b" / "notebook.py").read_bytes() == b"b source"
    _assert_no_leftovers(tmp_path)


# ─── state 8: uid pool exhaustion ───────────────────────────────────────────


def test_migrate_raises_uid_pool_exhausted_instead_of_skipping_the_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full uid pool must abort the migration loudly, not silently skip the slug.

    This is the propagation test that stops a future reader from "fixing" a
    boot failure here with a swallowing try/except.
    """
    import notebook_host.jail as jail_mod

    monkeypatch.setattr(jail_mod, "can_apply_jail", lambda: True)
    _uids_file(tmp_path).write_text(json.dumps({"other-slug": _UID_START}))
    (tmp_path / "b.py").write_bytes(b"fresh slug source")

    with pytest.raises(UidPoolExhaustedError):
        _migrate(tmp_path, uid_start=_UID_START, uid_end=_UID_START)


# ─── additional coverage from the task-1 acceptance criteria ───────────────


def test_migrate_slug_with_no_legacy_data_dir_gets_an_empty_data_dir(tmp_path: Path) -> None:
    """A slug with no .data/ directory migrates successfully with an empty data/."""
    (tmp_path / "a.py").write_bytes(b"no attachments here")

    migrated = _migrate(tmp_path)

    assert migrated == ["a"]
    assert (tmp_path / "a" / "data").is_dir(), "data/ must be created even with no legacy .data/"
    assert list((tmp_path / "a" / "data").iterdir()) == [], "data/ must be empty"


def test_migrate_skips_unparseable_slug_without_aborting_the_run(tmp_path: Path) -> None:
    """A file that doesn't pass safe_slug is skipped with a warning, not fatal."""
    (tmp_path / "not a slug!.py").write_bytes(b"junk")
    (tmp_path / "good.py").write_bytes(b"good source")

    migrated = _migrate(tmp_path)

    assert migrated == ["good"], "the valid slug must still migrate despite the invalid sibling"
    assert (tmp_path / "not a slug!.py").exists(), (
        "the unparseable file must be left in place for an operator to inspect"
    )


def test_migrate_never_touches_registry_files(tmp_path: Path) -> None:
    """blogs.json, pids.json and uids.json are never treated as slugs."""
    blogs = {"z": {"slug": "z", "created_at": 1.0}}
    pids: dict[str, object] = {"z": {"slug": "z", "pid": 1, "port": 8100, "started_at": 1.0}}
    (tmp_path / "blogs.json").write_text(json.dumps(blogs))
    (tmp_path / "pids.json").write_text(json.dumps(pids))
    (tmp_path / "z.py").write_bytes(b"z source")

    _migrate(tmp_path)

    assert json.loads((tmp_path / "blogs.json").read_text()) == blogs
    assert json.loads((tmp_path / "pids.json").read_text()) == pids
