"""Per-slug on-disk layout and directory-tree lifecycle.

This module owns two things: the shape of a slug's directory tree
(``SlugPaths``) and creating/removing that tree on disk. ``data_dir/<slug>/``
is the isolation boundary a jailed marimo process is confined to; every
host-owned registry file (``blogs.json``, ``pids.json``, ``uids.json``)
deliberately sits outside it, as a sibling of every slug root rather than
inside any of them.

Nothing here imports ``Settings`` — callers read config and pass paths / ints
in explicitly, so this module has no dependency on ``notebook_host.config``.

Pure functions only for path computation; the tree create/remove pair is the
only filesystem I/O (mkdir/chmod/chown/rmtree — no clock, no process control,
no network).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

_TREE_MODE = 0o700


@dataclass(frozen=True)
class SlugPaths:
    """Every on-disk path a slug owns. The single source of truth for the layout.

    ``notebook``'s basename is always ``notebook.py`` regardless of slug —
    deliberate, because ``spawn_marimo`` passes ``file_path.name`` as marimo's
    positional argument, so a fixed basename keeps that argv slug-independent.
    """

    root: Path
    notebook: Path
    data: Path
    workspace: Path
    home: Path
    log: Path


def get_slug_paths(data_dir: Path, slug: str) -> SlugPaths:
    """Compute every path a slug owns. Pure — no filesystem access.

    ``slug`` must already have passed ``lifecycle.safe_slug``; this function
    does not re-validate it (importing ``lifecycle`` from here would create a
    cycle, since plan-wiring makes ``lifecycle`` import ``jail``).
    """
    root = data_dir / slug
    return SlugPaths(
        root=root,
        notebook=root / "notebook.py",
        data=root / "data",
        workspace=root / "workspace",
        home=root / "home",
        log=root / "marimo.log",
    )


def ensure_slug_jail(data_dir: Path, slug: str, *, uid: int | None = None) -> SlugPaths:
    """Create (or repair) a slug's whole directory tree at mode 0700.

    Creates ``root``, ``data``, ``workspace``, ``home`` and chmods each to
    0700 unconditionally — ``mkdir``'s ``mode`` argument is masked by umask,
    so an explicit ``chmod`` is required. When ``uid`` is given, each of those
    four directories is also chowned to ``(uid, uid)``.

    Idempotent: safe to call repeatedly on an existing, already-owned tree.
    Does NOT touch ``notebook.py``, ``marimo.log``, or anything already inside
    ``data/`` — file ownership is the write site's job, and files created by
    the jailed marimo process are already owned correctly by construction.

    ``slug`` must already have passed ``lifecycle.safe_slug``.
    """
    paths = get_slug_paths(data_dir, slug)
    dirs = (paths.root, paths.data, paths.workspace, paths.home)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, _TREE_MODE)
        if uid is not None:
            os.chown(d, uid, uid)
    return paths


def remove_slug_tree(data_dir: Path, slug: str) -> None:
    """Remove a slug's whole directory tree in one call.

    Replaces the unlink-plus-two-rmtrees pattern the flat layout required
    across three call sites — exactly the duplication that let the
    background sweep delete only one of the three on-disk pieces. A no-op
    (does not raise) if the tree doesn't exist.

    ``slug`` must already have passed ``lifecycle.safe_slug``.
    """
    shutil.rmtree(get_slug_paths(data_dir, slug).root, ignore_errors=True)
