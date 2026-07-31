"""Tests for notebook_host.jail: SlugPaths contract, tree lifecycle, uid pool."""

from __future__ import annotations

import os
from pathlib import Path

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
