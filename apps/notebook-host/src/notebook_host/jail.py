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

import json
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


# ─── uid pool ────────────────────────────────────────────────────────────────
#
# A persisted slug -> uid registry, not a deterministic hash of the slug.
# Blogs are permanent and accumulate for the life of the deployment, so a
# hash into any convenient-sized range carries real birthday-collision risk
# — and a collision silently defeats isolation for both colliding slugs. The
# registry lives at ``data_dir / "uids.json"``, alongside ``blogs.json`` and
# ``pids.json``, outside every slug root. A plain ``dict[str, int]`` is
# enough here — unlike ``PidRecord``/``BlogRecord``, a uid row carries no
# other fields, so a Pydantic model would buy nothing.


class UidPoolExhaustedError(RuntimeError):
    """Raised when every uid in the configured range is already allocated."""


def load_uid_registry(path: Path) -> dict[str, int]:
    """Read the uid registry. Returns an empty dict if missing or malformed.

    Same posture as ``pids_store.load_pids``: a malformed file means a
    previous instance died mid-write, so we forget it rather than trust
    partial state. Entries whose key is not a ``str`` or whose value is not
    a genuine ``int`` (``bool`` is an ``int`` subclass and is rejected too)
    are skipped rather than raising.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(key, str):
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        out[key] = value
    return out


def save_uid_registry(path: Path, records: dict[str, int]) -> None:
    """Atomically rewrite the uid registry (tmp + rename), same idiom as save_pids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(records, indent=2))
    os.replace(tmp, path)


def allocate_uid(registry: dict[str, int], slug: str, *, start: int, end: int) -> int:
    """Return the uid for ``slug``, allocating a new one if needed. Pure.

    If ``slug`` is already in ``registry``, its existing uid is returned
    unchanged — idempotency the boot migration depends on. Otherwise the
    lowest value in ``range(start, end + 1)`` not present in
    ``registry.values()`` is returned. Does not touch disk and does not
    mutate ``registry``. Raises ``UidPoolExhaustedError`` if every value in
    the range is already taken.
    """
    if slug in registry:
        return registry[slug]
    used = set(registry.values())
    for uid in range(start, end + 1):
        if uid not in used:
            return uid
    raise UidPoolExhaustedError(f"uid pool exhausted ({start}-{end}, {len(registry)} allocated)")


def get_or_create_slug_uid(path: Path, slug: str, *, start: int, end: int) -> int:
    """Load the registry, allocate (or reuse) a uid for ``slug``, save if changed.

    No write on the hit path — an already-registered slug returns without
    rewriting the file.
    """
    registry = load_uid_registry(path)
    if slug in registry:
        return registry[slug]
    uid = allocate_uid(registry, slug, start=start, end=end)
    registry[slug] = uid
    save_uid_registry(path, registry)
    return uid


def release_slug_uid(path: Path, slug: str) -> None:
    """Drop a slug's uid from the registry, making it immediately reusable.

    A no-op for a slug that isn't registered — does not raise and does not
    create the file if it didn't already exist.

    The caller MUST have already killed the slug's process before calling
    this: the uid becomes reusable the instant this returns, and a
    surviving process still holding the old uid would otherwise be able to
    reach whichever slug gets allocated it next. Releasing is required
    rather than optional — edit-mode notebooks are TTL-reaped every
    ``subprocess_ttl_seconds`` (default 24h), so a never-releasing pool
    would burn through its whole range on ordinary churn.
    """
    registry = load_uid_registry(path)
    if registry.pop(slug, None) is not None:
        save_uid_registry(path, registry)
