"""Root-gated proof that the per-notebook uid jail actually holds.

These tests do not run in default CI and are not expected to. The
``pytest-notebook-host`` job runs on an unprivileged GitHub runner, and the
root ``addopts`` in ``pyproject.toml`` already deselect the ``slow`` marker
every test in this module carries. Dropping into another uid via
``os.setuid`` requires ``CAP_SETUID``, which an unprivileged process never
has — so unlike the rest of the suite, these cannot be faked with a
monkeypatched ``preexec_fn``; they exist specifically to prove the property
the monkeypatched tests elsewhere in this tree cannot.

Run them explicitly, as root:

    sudo -E uv run pytest apps/notebook-host/tests/test_jail_privilege.py -m slow

or inside a container built from ``apps/notebook-host/Dockerfile``, which
runs as root by default.

An automated home for these belongs with the already-anticipated
notebook-host ``slow`` tier on a Linux runner (tracked as a future
requirement) — this plan does not add that CI job.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from notebook_host.jail import build_jailed_preexec, ensure_slug_jail
from notebook_host.lifecycle import scrub_env

_UID_A = 100001
_UID_B = 100002


def _run_as_uid(
    uid: int, code: str, *, cwd: str | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a child process dropped into ``uid`` via preexec_fn."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        preexec_fn=build_jailed_preexec(uid, rlimit_as_bytes=None, rlimit_cpu_seconds=None),
        capture_output=True,
        text=True,
        timeout=30,
    )


_READ_CODE = """
import json
try:
    with open({path!r}) as f:
        data = f.read()
except PermissionError:
    print(json.dumps({{"error": "PermissionError"}}))
except OSError as e:
    print(json.dumps({{"error": "OSError:" + type(e).__name__}}))
else:
    print(json.dumps({{"data": data}}))
"""


@pytest.mark.slow
@pytest.mark.skipif(os.geteuid() != 0, reason="uid drop requires root")
def test_child_drops_to_uid_gid_and_clears_supplementary_groups(tmp_path: Path) -> None:
    """The child reports getuid()==uid, getgid()==uid, and an empty supplementary group list.

    The groups assertion is the one that catches a missing or misordered
    ``setgroups([])`` — without it the child keeps 0 (root) in its
    supplementary list even after setuid/setgid drop the primary identity.
    """
    paths = ensure_slug_jail(tmp_path, "slug-a", uid=_UID_A)
    code = "import json, os; print(json.dumps({'uid': os.getuid(), 'gid': os.getgid(), 'groups': os.getgroups()}))"
    result = _run_as_uid(_UID_A, code, cwd=str(paths.workspace))
    assert result.returncode == 0, f"child should exit cleanly; stderr={result.stderr}"
    identity = json.loads(result.stdout)
    assert identity["uid"] == _UID_A, "child should report the dropped uid"
    assert identity["gid"] == _UID_A, (
        "child should report the dropped gid (== uid, no separate allocation)"
    )
    assert identity["groups"] == [], (
        "supplementary groups must be empty — a missing setgroups([]) would leave root's groups intact"
    )


@pytest.mark.slow
@pytest.mark.skipif(os.geteuid() != 0, reason="uid drop requires root")
def test_own_subtree_reachable_via_relative_path(tmp_path: Path) -> None:
    """With cwd inside its own workspace, a jailed process can read its own data/ via a relative path."""
    paths = ensure_slug_jail(tmp_path, "slug-a", uid=_UID_A)
    seeded = paths.data / "x.txt"
    seeded.write_bytes(b"hello-from-a")
    os.chown(seeded, _UID_A, _UID_A)
    os.chmod(seeded, 0o600)

    code = _READ_CODE.format(path="../data/x.txt")
    result = _run_as_uid(_UID_A, code, cwd=str(paths.workspace))
    assert result.returncode == 0, f"child should exit cleanly; stderr={result.stderr}"
    body = json.loads(result.stdout)
    assert body.get("data") == "hello-from-a", (
        f"own-subtree read should succeed and return the seeded bytes, got {body}"
    )


@pytest.mark.slow
@pytest.mark.skipif(os.geteuid() != 0, reason="uid drop requires root")
def test_cross_slug_relative_escape_denied(tmp_path: Path) -> None:
    """A relative '..' traversal into a sibling slug's data is denied."""
    paths_a = ensure_slug_jail(tmp_path, "slug-a", uid=_UID_A)
    paths_b = ensure_slug_jail(tmp_path, "slug-b", uid=_UID_B)
    secret = paths_b.data / "secret.txt"
    secret.write_bytes(b"top-secret")
    os.chown(secret, _UID_B, _UID_B)
    os.chmod(secret, 0o600)
    # data_dir itself is the 0700 root:root boundary every slug root sits under.
    os.chown(tmp_path, 0, 0)
    os.chmod(tmp_path, 0o700)

    code = _READ_CODE.format(path="../../slug-b/data/secret.txt")
    result = _run_as_uid(_UID_A, code, cwd=str(paths_a.workspace))
    assert result.returncode == 0, f"child should exit cleanly; stderr={result.stderr}"
    body = json.loads(result.stdout)
    assert body.get("error") == "PermissionError", (
        f"a relative cross-slug escape must raise PermissionError, got {body}"
    )


@pytest.mark.slow
@pytest.mark.skipif(os.geteuid() != 0, reason="uid drop requires root")
def test_cross_slug_absolute_escape_denied(tmp_path: Path) -> None:
    """An absolute-path read into a sibling slug's data is denied — same wall as the relative case."""
    paths_a = ensure_slug_jail(tmp_path, "slug-a", uid=_UID_A)
    paths_b = ensure_slug_jail(tmp_path, "slug-b", uid=_UID_B)
    secret = paths_b.data / "secret.txt"
    secret.write_bytes(b"top-secret")
    os.chown(secret, _UID_B, _UID_B)
    os.chmod(secret, 0o600)
    os.chown(tmp_path, 0, 0)
    os.chmod(tmp_path, 0o700)

    code = _READ_CODE.format(path=str(secret))
    result = _run_as_uid(_UID_A, code, cwd=str(paths_a.workspace))
    assert result.returncode == 0, f"child should exit cleanly; stderr={result.stderr}"
    body = json.loads(result.stdout)
    assert body.get("error") == "PermissionError", (
        f"an absolute cross-slug escape must raise PermissionError, got {body}"
    )


@pytest.mark.slow
@pytest.mark.skipif(os.geteuid() != 0, reason="uid drop requires root")
def test_registry_files_unreachable(tmp_path: Path) -> None:
    """blogs.json and uids.json — host-owned state outside every jail — are unreadable by a jailed uid."""
    paths_a = ensure_slug_jail(tmp_path, "slug-a", uid=_UID_A)
    blogs = tmp_path / "blogs.json"
    uids = tmp_path / "uids.json"
    blogs.write_text("{}", encoding="utf-8")
    uids.write_text("{}", encoding="utf-8")
    os.chown(blogs, 0, 0)
    os.chown(uids, 0, 0)
    os.chmod(blogs, 0o600)
    os.chmod(uids, 0o600)
    os.chown(tmp_path, 0, 0)
    os.chmod(tmp_path, 0o700)

    for registry_path in (blogs, uids):
        code = _READ_CODE.format(path=str(registry_path))
        result = _run_as_uid(_UID_A, code, cwd=str(paths_a.workspace))
        assert result.returncode == 0, f"child should exit cleanly; stderr={result.stderr}"
        body = json.loads(result.stdout)
        assert body.get("error") == "PermissionError", (
            f"{registry_path.name} must be unreadable by a jailed uid, got {body}"
        )


@pytest.mark.slow
@pytest.mark.skipif(os.geteuid() != 0, reason="uid drop requires root")
def test_proc_environ_closed_across_uids(tmp_path: Path) -> None:
    """A jailed uid cannot read another uid's (or root's) /proc/<pid>/environ.

    Closes scouting finding 5: same-uid env-scrub alone would not stop this;
    the uid split is what actually denies it, verified here directly rather
    than assumed.
    """
    long_lived = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", "import time; time.sleep(20)"],
        preexec_fn=build_jailed_preexec(_UID_A, rlimit_as_bytes=None, rlimit_cpu_seconds=None),
    )
    try:
        time.sleep(0.5)  # give the child a moment to actually be running
        for target_pid in (long_lived.pid, os.getpid()):
            code = _READ_CODE.format(path=f"/proc/{target_pid}/environ")
            result = _run_as_uid(_UID_B, code)
            assert result.returncode == 0, f"child should exit cleanly; stderr={result.stderr}"
            body = json.loads(result.stdout)
            assert body.get("error") == "PermissionError", (
                f"/proc/{target_pid}/environ must be unreadable by a different uid, got {body}"
            )
    finally:
        long_lived.kill()
        long_lived.wait(timeout=5)


@pytest.mark.slow
@pytest.mark.skipif(os.geteuid() != 0, reason="uid drop requires root")
def test_uv_and_marimo_run_under_the_dropped_uid(tmp_path: Path) -> None:
    """uv run --with marimo actually works under a dropped uid, given a per-slug HOME.

    This is the single most important test in the phase — it is the thing
    that was predicted to sink the whole mechanism (CONTEXT.md D-02).
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    paths = ensure_slug_jail(tmp_path, "slug-a", uid=_UID_A)
    env = scrub_env(dict(os.environ))
    env["HOME"] = str(paths.home)

    result = subprocess.run(  # noqa: S603
        [uv, "run", "--with", "marimo", "marimo", "--version"],
        cwd=str(paths.workspace),
        env=env,
        preexec_fn=build_jailed_preexec(_UID_A, rlimit_as_bytes=None, rlimit_cpu_seconds=None),
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"uv run --with marimo should succeed under the dropped uid: {combined}"
    )
    assert "Permission denied" not in combined, (
        f"the dropped-uid run must not hit a permission error: {combined}"
    )
