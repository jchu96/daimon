"""The checkpoint turn's prompt, and the readers for its reply.

When a task's workspace has to be replaced, daimon spends one bounded,
billed turn on the OLD session asking it to write down what it was doing and
tar its own files into the session's outputs directory. That archive is the
only way working files cross into the successor: the Files API lists
``/mnt/session/outputs/*`` and mounted uploads and nothing else, so the
working directory, ``/tmp`` and the repo checkout are invisible to daimon
unless the session itself packs them (capability matrix P4.b).

Everything here is pure string work. The module deliberately imports nothing
from the rest of ``daimon.core``: it is the one piece of the transfer path
that has to be readable on its own, because a wrong word in the prompt costs
a real turn and can lose real work.

Sandbox facts the prompt is built around (all observed live, capability
matrix §A/§D):

- A bash turn starts with cwd ``/`` and ``HOME=/root``, so every path in the
  prompt is absolute or explicitly relative to ``/``.
- The outputs listing is basename-only, so the archive is written flat and
  recognised later by its name prefix (:func:`is_handoff_filename`).
- ``/mnt/session/uploads`` (credential and bundle mounts), ``/mnt/memory``
  (agent memory) and ``/mnt/skills`` are mounts belonging to the destination,
  never to the task; they must not travel in the bundle.
"""

from __future__ import annotations

import re
import uuid

HANDOFF_FILENAME_PREFIX = "daimon-handoff-"

# Daimon's own cap on a transfer bundle, equal to
# ``output_delivery.MAX_BYTES_PER_FILE`` by design: a bundle is downloaded
# through the same path as any other session output. Not an MA limit — 25 MiB
# uploads were accepted live (P4.g). Deliberately NOT imported from
# ``output_delivery``; a test asserts the two stay equal.
HANDOFF_MAX_BYTES = 20 * 1024 * 1024

# Absolute mount points that must never enter a bundle: they hold the
# destination's own credentials, the agent's memory store, the platform's
# skills, and the outputs directory the archive itself is written into.
CHECKPOINT_EXCLUDED_PATHS: tuple[str, ...] = (
    "/mnt/session/uploads",
    "/mnt/memory",
    "/mnt/skills",
    "/mnt/session/outputs",
)

# Directory names that are large, reproducible, or both. Matched at any depth.
CHECKPOINT_EXCLUDED_GLOBS: tuple[str, ...] = (
    ".git/objects",
    "node_modules",
    ".venv",
    "__pycache__",
    ".cache",
)

# Written inside the sandbox, outside every archived root, so the list of
# oversized files never ends up inside the archive it filters.
_EXCLUDE_LIST_PATH = "/tmp/daimon-handoff-excludes.txt"

_SHA1_LINE = re.compile(r"^\s*([0-9a-f]{40})\s*$", re.MULTILINE)

# Lines that merely echo a command from the prompt, rather than being its
# output, when :func:`checkpoint_archive_listed` looks for the listing.
_COMMAND_ECHO_PREFIXES = ("tar ", "ls ", "find ", "cd ", "git ", "$ ", "# ")


def handoff_filename(transfer_id: uuid.UUID) -> str:
    """The flat outputs filename for one transfer's bundle."""

    return f"{HANDOFF_FILENAME_PREFIX}{transfer_id}.tar.gz"


def is_handoff_filename(filename: str) -> bool:
    """True for a transfer bundle's basename.

    Prefix-only by design: the outputs listing gives basenames, and the
    output sweep uses this to leave bundles in place instead of posting and
    deleting them.
    """

    return filename.startswith(HANDOFF_FILENAME_PREFIX)


def _relative_to_root(path: str) -> str:
    """``/root`` -> ``root``: a tar member prefix under ``-C /``."""

    return path.strip().strip("/")


def build_checkpoint_prompt(
    *,
    transfer_id: uuid.UUID,
    repo_mount_path: str | None,
    max_bundle_mib: int,
    home_dir: str = "/root",
) -> str:
    """The single user message sent to the old session's checkpoint turn.

    Deterministic: same inputs, same bytes. ``repo_mount_path`` adds the
    git-state step and the prohibition on changing the repository; without a
    repo neither appears.
    """

    archive_path = f"/mnt/session/outputs/{handoff_filename(transfer_id)}"
    roots = [_relative_to_root(home_dir)]
    if repo_mount_path is not None:
        roots.append(_relative_to_root(repo_mount_path))
    roots_argument = " ".join(roots)

    excludes = [f"--exclude-from={_EXCLUDE_LIST_PATH}"]
    excludes += [f"--exclude='{_relative_to_root(path)}'" for path in CHECKPOINT_EXCLUDED_PATHS]
    excludes += [f"--exclude='*/{glob}'" for glob in CHECKPOINT_EXCLUDED_GLOBS]
    excludes.append("--exclude='*.env'")
    tar_command = "\n".join(
        [f"  tar czf {archive_path} \\"]
        + [f"    {exclude} \\" for exclude in excludes]
        + [f"    -C / {roots_argument}"]
    )

    steps: list[str] = []

    def step(header: str) -> str:
        """Number the steps as they are appended, so a skipped one leaves no gap."""

        return f"Step {len(steps) + 1} - {header}"

    sections: list[str] = [
        "This is a checkpoint turn. The workspace you are working in is being retired and "
        "your work is moving to a new one. Do exactly the steps below, in order, and nothing "
        "else. Do not open or read any image file."
    ]
    steps.append(
        "\n".join(
            [
                step("write the handoff note."),
                f"Write {home_dir}/HANDOFF.md with these five sections, in this order:",
                "- Task: what this thread is trying to achieve, in one paragraph.",
                "- Decisions: the decisions already made, and why.",
                "- In progress: the step you were in the middle of when this turn started.",
                "- Open questions: what is unresolved or waiting on someone else.",
                "- Files: every working file that matters, absolute path plus a one-line "
                "description of what it holds.",
            ]
        )
    )

    if repo_mount_path is not None:
        steps.append(
            "\n".join(
                [
                    f"{step('record the repository state.')} Run these four commands, in order:",
                    f"  git -C {repo_mount_path} rev-parse HEAD",
                    f"  git -C {repo_mount_path} status --porcelain",
                    f"  git -C {repo_mount_path} diff HEAD > {home_dir}/uncommitted.patch",
                    f"  git -C {repo_mount_path} ls-files --others --exclude-standard"
                    f" > {home_dir}/untracked.txt",
                    "NEVER RUN GIT COMMIT, GIT PUSH, GIT STASH, OR ANY OTHER COMMAND THAT "
                    "CHANGES THIS REPOSITORY'S HISTORY, INDEX, WORKING TREE, OR REMOTE. The "
                    "uncommitted work is captured as a patch on purpose. Leave the tree exactly "
                    "as you found it.",
                ]
            )
        )

    steps.append(
        "\n".join(
            [
                f"{step('build the archive.')} The shell starts in /, and the archive must be "
                "written flat into the outputs directory. Run these three commands, in order, "
                "exactly as written:",
                "  cd /",
                f"  find {roots_argument} -type f -size +{max_bundle_mib}M > {_EXCLUDE_LIST_PATH}",
                tar_command,
                "The find step lists files too large to carry so tar skips them; do not edit "
                "either command.",
            ]
        )
    )

    show: list[str] = [f"{step('show the result.')} Run:", f"  ls -l {archive_path}"]
    if repo_mount_path is not None:
        show.append(f"  git -C {repo_mount_path} rev-parse HEAD")
    steps.append("\n".join(show))

    steps.append(
        f"{step('reply')} with the output of the commands above and nothing else: no summary, "
        "no commentary, no explanation of what you did."
    )
    return "\n\n".join(sections + steps)


def checkpoint_head_lines(reply: str) -> tuple[str | None, str | None]:
    """The first and last commit hashes echoed by a checkpoint reply.

    The prompt asks for ``rev-parse HEAD`` before and after the archive step;
    two different hashes mean the session committed despite the prohibition,
    which the caller records rather than prevents. Returns ``(None, None)``
    when no hash was echoed and ``(hash, None)`` when only one was, so
    "unknown" is never mistaken for "unchanged".
    """

    hashes = _SHA1_LINE.findall(reply)
    if not hashes:
        return (None, None)
    if len(hashes) == 1:
        return (hashes[0], None)
    return (hashes[0], hashes[-1])


def checkpoint_archive_listed(reply: str, filename: str) -> bool:
    """True when the reply shows the archive in an ``ls`` listing.

    A weak confirmation, and only that: it proves the session named the file
    outside the commands it was handed. The authoritative check is the Files
    API listing, which the transfer polls next.
    """

    for line in reply.splitlines():
        stripped = line.strip()
        if filename not in stripped:
            continue
        if stripped.startswith(_COMMAND_ECHO_PREFIXES):
            continue
        return True
    return False
