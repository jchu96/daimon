"""Tests for notebook_host.jail: SlugPaths contract, tree lifecycle, uid pool."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from notebook_host.jail import get_slug_paths


def test_get_slug_paths_computes_the_documented_layout() -> None:
    """get_slug_paths maps a slug onto the fixed six-path layout."""
    paths = get_slug_paths(Path("/d"), "abc")
    assert paths.root == Path("/d/abc"), "root should be data_dir / slug"
    assert paths.notebook == Path("/d/abc/notebook.py"), (
        "notebook basename is fixed regardless of slug"
    )
    assert paths.data == Path("/d/abc/data"), "data should be root / 'data'"
    assert paths.workspace == Path("/d/abc/workspace"), "workspace should be root / 'workspace'"
    assert paths.home == Path("/d/abc/home"), "home should be root / 'home'"
    assert paths.log == Path("/d/abc/marimo.log"), "log should be root / 'marimo.log'"


def test_get_slug_paths_performs_no_filesystem_access(tmp_path: Path) -> None:
    """get_slug_paths is pure: calling it against a missing data_dir creates nothing."""
    missing = tmp_path / "does-not-exist"
    get_slug_paths(missing, "abc")
    assert not missing.exists(), "get_slug_paths must not touch the filesystem"


def test_ensure_slug_jail_creates_all_four_dirs_at_mode_0700(tmp_path: Path) -> None:
    """ensure_slug_jail creates root/data/workspace/home, each mode 0700."""
    from notebook_host.jail import ensure_slug_jail

    paths = ensure_slug_jail(tmp_path, "abc")
    for d in (paths.root, paths.data, paths.workspace, paths.home):
        assert d.is_dir(), f"{d} should exist and be a directory"
        assert d.stat().st_mode & 0o777 == 0o700, f"{d} should be mode 0700"


def test_ensure_slug_jail_is_idempotent_and_preserves_existing_data(tmp_path: Path) -> None:
    """Calling ensure_slug_jail twice succeeds and leaves data/ files untouched."""
    from notebook_host.jail import ensure_slug_jail

    paths = ensure_slug_jail(tmp_path, "abc")
    marker = paths.data / "marker.txt"
    marker.write_text("hello")

    ensure_slug_jail(tmp_path, "abc")

    assert marker.read_text() == "hello", (
        "a pre-existing file under data/ must survive a repeat ensure_slug_jail call"
    )


def test_ensure_slug_jail_chowns_to_self_uid_without_root(tmp_path: Path) -> None:
    """ensure_slug_jail(..., uid=os.getuid()) succeeds via self-chown, no root needed."""
    from notebook_host.jail import ensure_slug_jail

    paths = ensure_slug_jail(tmp_path, "abc", uid=os.getuid())
    for d in (paths.root, paths.data, paths.workspace, paths.home):
        assert d.stat().st_uid == os.getuid(), f"{d} should be owned by the caller's uid"


def test_remove_slug_tree_deletes_everything(tmp_path: Path) -> None:
    """remove_slug_tree removes the whole per-slug root in one call."""
    from notebook_host.jail import ensure_slug_jail, remove_slug_tree

    ensure_slug_jail(tmp_path, "abc")
    remove_slug_tree(tmp_path, "abc")
    assert not (tmp_path / "abc").exists(), "the slug's root should be gone"


def test_remove_slug_tree_on_absent_tree_does_not_raise(tmp_path: Path) -> None:
    """remove_slug_tree is a no-op (not an error) when the tree never existed."""
    from notebook_host.jail import remove_slug_tree

    remove_slug_tree(tmp_path, "never-existed")  # must not raise


# ─── uid pool ────────────────────────────────────────────────────────────────


def test_allocate_uid_returns_lowest_free_value_in_range() -> None:
    """A fresh slug gets the lowest unused uid in [start, end]."""
    from notebook_host.jail import allocate_uid

    assert allocate_uid({}, "a", start=100000, end=100002) == 100000, (
        "first allocation in an empty registry should be start"
    )
    assert allocate_uid({"a": 100000}, "b", start=100000, end=100002) == 100001, (
        "next distinct slug should get the next free uid"
    )


def test_allocate_uid_is_idempotent_for_an_already_registered_slug() -> None:
    """A slug already in the registry keeps its uid even if a lower one is free."""
    from notebook_host.jail import allocate_uid

    registry = {"a": 100005}
    assert allocate_uid(registry, "a", start=100000, end=100010) == 100005, (
        "an already-registered slug must return its existing uid, not the lowest free one"
    )


def test_allocate_uid_raises_when_pool_exhausted() -> None:
    """Asking for a uid when every value in range is taken raises, naming the range."""
    from notebook_host.jail import UidPoolExhaustedError, allocate_uid

    registry = {"a": 100000, "b": 100001}
    try:
        allocate_uid(registry, "c", start=100000, end=100001)
    except UidPoolExhaustedError as exc:
        message = str(exc)
        assert "100000" in message, "exhaustion error should name the range start"
        assert "100001" in message, "exhaustion error should name the range end"
    else:
        raise AssertionError("expected UidPoolExhaustedError when the pool is full")


def test_allocate_uid_does_not_mutate_its_input() -> None:
    """allocate_uid is pure: the caller's registry dict is unchanged after the call."""
    from notebook_host.jail import allocate_uid

    registry = {"a": 100000}
    before = dict(registry)
    allocate_uid(registry, "b", start=100000, end=100005)
    assert registry == before, "allocate_uid must not mutate the registry it was given"


def test_load_uid_registry_missing_file_returns_empty(tmp_path: Path) -> None:
    """A missing registry file loads as an empty dict, same posture as load_pids."""
    from notebook_host.jail import load_uid_registry

    assert load_uid_registry(tmp_path / "missing.json") == {}, (
        "a missing uids.json should load as {}"
    )


def test_load_uid_registry_malformed_file_returns_empty(tmp_path: Path) -> None:
    """A file that isn't valid JSON loads as an empty dict rather than raising."""
    from notebook_host.jail import load_uid_registry

    path = tmp_path / "uids.json"
    path.write_text("not json{")
    assert load_uid_registry(path) == {}, "malformed JSON should load as {}"


def test_load_uid_registry_skips_invalid_entries(tmp_path: Path) -> None:
    """Non-str keys, non-int values, and bool values (an int subclass) are skipped."""
    from notebook_host.jail import load_uid_registry

    path = tmp_path / "uids.json"
    path.write_text('{"a": 1, "b": "x", "c": true, "12": 3}')
    # JSON object keys are always strings, so the plan's "non-str key" case is
    # expressed here as the numeral-shaped string "12"; only the
    # type-mismatched values ("b": str, "c": bool) get filtered.
    assert load_uid_registry(path) == {"a": 1, "12": 3}, (
        "only entries with a str key and a genuine int (non-bool) value should survive"
    )


def test_get_or_create_slug_uid_is_stable_across_calls(tmp_path: Path) -> None:
    """Calling get_or_create_slug_uid twice for the same slug returns the same uid."""
    from notebook_host.jail import get_or_create_slug_uid

    path = tmp_path / "uids.json"
    first = get_or_create_slug_uid(path, "a", start=100000, end=100999)
    contents_after_first = path.read_text()
    second = get_or_create_slug_uid(path, "a", start=100000, end=100999)

    assert first == second, "the same slug must always get the same uid"
    assert path.read_text() == contents_after_first, (
        "a hit (no new allocation) must not rewrite the registry file"
    )


def test_release_slug_uid_frees_the_slot_for_reuse(tmp_path: Path) -> None:
    """After release, the slug is gone from the registry and its uid may be reused."""
    from notebook_host.jail import get_or_create_slug_uid, load_uid_registry, release_slug_uid

    path = tmp_path / "uids.json"
    freed_uid = get_or_create_slug_uid(path, "a", start=100000, end=100000)

    release_slug_uid(path, "a")

    assert "a" not in load_uid_registry(path), "released slug must be gone from the registry"
    reused = get_or_create_slug_uid(path, "b", start=100000, end=100000)
    assert reused == freed_uid, "a released uid must become available to a new slug"


def test_release_slug_uid_on_unknown_slug_is_a_noop(tmp_path: Path) -> None:
    """release_slug_uid for a slug never registered does not raise or create a file."""
    from notebook_host.jail import release_slug_uid

    path = tmp_path / "uids.json"
    release_slug_uid(path, "never-existed")  # must not raise
    assert not path.exists(), "releasing an unknown slug must not create the registry file"


def test_save_uid_registry_writes_atomically_via_tmp_and_replace(tmp_path: Path) -> None:
    """save_uid_registry writes the exact records, readable back via load_uid_registry."""
    from notebook_host.jail import load_uid_registry, save_uid_registry

    path = tmp_path / "uids.json"
    save_uid_registry(path, {"a": 100000, "b": 100001})
    assert load_uid_registry(path) == {"a": 100000, "b": 100001}, (
        "save then load should round-trip the registry"
    )
    on_disk = json.loads(path.read_text())
    assert on_disk == {"a": 100000, "b": 100001}, "the file's raw JSON should match the records"


# ─── remove_slug_tree uid release ─────────────────────────────────────────────


def test_remove_slug_tree_with_uids_file_releases_the_slug(tmp_path: Path) -> None:
    """remove_slug_tree(..., uids_file=p) drops the slug from the uid registry too."""
    from notebook_host.jail import (
        ensure_slug_jail,
        get_or_create_slug_uid,
        load_uid_registry,
        remove_slug_tree,
    )

    uids_file = tmp_path / "uids.json"
    ensure_slug_jail(tmp_path, "abc")
    get_or_create_slug_uid(uids_file, "abc", start=100000, end=100005)

    remove_slug_tree(tmp_path, "abc", uids_file=uids_file)

    assert "abc" not in load_uid_registry(uids_file), (
        "remove_slug_tree with uids_file must release the slug's uid"
    )


def test_remove_slug_tree_without_uids_file_leaves_registry_untouched(tmp_path: Path) -> None:
    """remove_slug_tree with no uids_file kwarg does not touch the uid registry."""
    from notebook_host.jail import (
        ensure_slug_jail,
        get_or_create_slug_uid,
        load_uid_registry,
        remove_slug_tree,
    )

    uids_file = tmp_path / "uids.json"
    ensure_slug_jail(tmp_path, "abc")
    get_or_create_slug_uid(uids_file, "abc", start=100000, end=100005)

    remove_slug_tree(tmp_path, "abc")

    assert "abc" in load_uid_registry(uids_file), (
        "remove_slug_tree without uids_file must leave the uid registry unchanged"
    )


# ─── privilege drop: can_apply_jail / resolve_jail_uid / build_jailed_preexec ─


def test_resolve_jail_uid_raises_when_jail_unavailable_and_not_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """can_apply_jail() False + allow_unjailed=False raises JailUnavailableError."""
    from notebook_host import jail

    monkeypatch.setattr(jail, "can_apply_jail", lambda: False)
    with pytest.raises(jail.JailUnavailableError) as exc_info:
        jail.resolve_jail_uid(
            tmp_path / "uids.json", "abc", start=100000, end=100005, allow_unjailed=False
        )
    assert "allow_unjailed_spawn" in str(exc_info.value), (
        "the fail-closed error must name the deliberate opt-out setting"
    )


def test_resolve_jail_uid_returns_none_when_jail_unavailable_and_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """can_apply_jail() False + allow_unjailed=True returns None (spawn unjailed)."""
    from notebook_host import jail

    monkeypatch.setattr(jail, "can_apply_jail", lambda: False)
    result = jail.resolve_jail_uid(
        tmp_path / "uids.json", "abc", start=100000, end=100005, allow_unjailed=True
    )
    assert result is None, "an explicit opt-out on an unjailable host must return None, not raise"


def test_resolve_jail_uid_returns_a_stable_int_when_jail_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """can_apply_jail() True resolves (and persists) a real uid from the pool."""
    from notebook_host import jail

    monkeypatch.setattr(jail, "can_apply_jail", lambda: True)
    uids_file = tmp_path / "uids.json"
    first = jail.resolve_jail_uid(uids_file, "abc", start=100000, end=100005, allow_unjailed=False)
    second = jail.resolve_jail_uid(uids_file, "abc", start=100000, end=100005, allow_unjailed=False)
    assert isinstance(first, int), "a jailable host must resolve to a real uid, not None"
    assert first == second, "the same slug must resolve to the same uid across calls"


def test_resolve_jail_uid_propagates_pool_exhaustion_even_when_unjailed_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A full uid pool raises UidPoolExhaustedError, never degrading to None."""
    from notebook_host import jail

    monkeypatch.setattr(jail, "can_apply_jail", lambda: True)
    uids_file = tmp_path / "uids.json"
    jail.get_or_create_slug_uid(uids_file, "a", start=100000, end=100000)

    with pytest.raises(jail.UidPoolExhaustedError):
        jail.resolve_jail_uid(uids_file, "b", start=100000, end=100000, allow_unjailed=True)


def test_build_jailed_preexec_returns_a_callable_for_any_uid() -> None:
    """build_jailed_preexec never returns None — build-time only, not invoked here."""
    from notebook_host.jail import build_jailed_preexec

    preexec = build_jailed_preexec(1000, rlimit_as_bytes=None, rlimit_cpu_seconds=None)
    assert callable(preexec), "build_jailed_preexec must always return a callable"
