"""Tests for the checkpoint turn's prompt and the readers for its reply."""

from __future__ import annotations

import re
import uuid

from daimon.core.checkpoint_prompt import (
    CHECKPOINT_EXCLUDED_GLOBS,
    CHECKPOINT_EXCLUDED_PATHS,
    HANDOFF_FILENAME_PREFIX,
    HANDOFF_MAX_BYTES,
    build_checkpoint_prompt,
    checkpoint_archive_listed,
    checkpoint_head_lines,
    handoff_filename,
    is_handoff_filename,
)
from daimon.core.output_delivery import MAX_BYTES_PER_FILE

TRANSFER_ID = uuid.UUID("0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0")
REPO = "/mnt/repo/analytics"


def test_handoff_max_bytes_matches_the_output_delivery_per_file_cap() -> None:
    assert HANDOFF_MAX_BYTES == MAX_BYTES_PER_FILE, (
        "a transfer bundle is downloaded through the output-delivery path, so its cap "
        "must equal that path's per-file cap"
    )


def test_handoff_filename_is_the_prefix_plus_the_transfer_id() -> None:
    assert handoff_filename(TRANSFER_ID) == f"{HANDOFF_FILENAME_PREFIX}{TRANSFER_ID}.tar.gz", (
        "the outputs listing is basename-only, so the name is the whole identity"
    )


def test_is_handoff_filename_accepts_a_bundle_and_rejects_an_ordinary_output() -> None:
    assert is_handoff_filename(handoff_filename(TRANSFER_ID)), (
        "the sweep must recognise a bundle by its basename prefix"
    )
    assert not is_handoff_filename("report.png"), "an ordinary output is not a bundle"


def test_prompt_names_the_exact_archive_path_and_filename() -> None:
    prompt = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=REPO, max_bundle_mib=20
    )
    filename = handoff_filename(TRANSFER_ID)
    assert f"tar czf /mnt/session/outputs/{filename}" in prompt, (
        "the archive must be written flat into the outputs directory"
    )
    assert f"ls -l /mnt/session/outputs/{filename}" in prompt, (
        "the reply has to show the archive so the transfer can read the size back"
    )


def test_prompt_excludes_every_configured_path_and_glob() -> None:
    prompt = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=REPO, max_bundle_mib=20
    )
    for path in CHECKPOINT_EXCLUDED_PATHS:
        assert f"--exclude='{path.lstrip('/')}'" in prompt, (
            f"{path} is a destination-owned mount and must never enter the bundle"
        )
    for glob in CHECKPOINT_EXCLUDED_GLOBS:
        assert f"--exclude='*/{glob}'" in prompt, f"{glob} must be excluded at any depth"
    assert "--exclude='*.env'" in prompt, "credential files must never enter the bundle"


def test_prompt_writes_an_oversize_exclude_list_before_tarring() -> None:
    prompt = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=REPO, max_bundle_mib=7
    )
    find_index = prompt.index("find root mnt/repo/analytics -type f -size +7M")
    tar_index = prompt.index("tar czf")
    assert find_index < tar_index, "the exclude list must exist before tar reads it"
    exclude_list = prompt[find_index:].split(">", 1)[1].split("\n", 1)[0].strip()
    assert f"--exclude-from={exclude_list}" in prompt, (
        "tar must read back the same exclude list the find step wrote"
    )


def test_prompt_forbids_history_changing_git_commands_and_never_asks_for_one() -> None:
    prompt = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=REPO, max_bundle_mib=20
    )
    lowered = prompt.lower()
    assert lowered.count("git commit") == 1, "'git commit' may appear only in the prohibition"
    assert lowered.count("git push") == 1, "'git push' may appear only in the prohibition"
    assert lowered.count("git stash") == 1, "'git stash' may appear only in the prohibition"
    prohibition = next(
        line for line in prompt.splitlines() if line.startswith("NEVER RUN GIT COMMIT")
    )
    assert "GIT PUSH" in prohibition and "GIT STASH" in prohibition, (
        "the one sentence naming these commands is the sentence banning them"
    )
    assert "git commit" not in prompt, "the prohibition is stated in capitals, never as a command"


def test_prompt_captures_repo_state_without_a_repo_omitting_the_git_steps() -> None:
    with_repo = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=REPO, max_bundle_mib=20
    )
    for command in (
        f"git -C {REPO} rev-parse HEAD",
        f"git -C {REPO} status --porcelain",
        f"git -C {REPO} diff HEAD > /root/uncommitted.patch",
        f"git -C {REPO} ls-files --others --exclude-standard > /root/untracked.txt",
    ):
        assert command in with_repo, f"{command} is part of the repo-state step"
    assert with_repo.count(f"git -C {REPO} rev-parse HEAD") == 2, (
        "HEAD is echoed before and after the archive so a commit is detectable"
    )
    assert " root mnt/repo/analytics" in with_repo, "the repo mount is archived alongside $HOME"

    without_repo = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=None, max_bundle_mib=20
    )
    assert "git -C" not in without_repo, "no repo means no git commands"
    assert "NEVER RUN GIT" not in without_repo, "and no repo prohibition to state"
    assert "-C / root" in without_repo, "the home directory is archived either way"


def test_prompt_numbers_steps_consecutively_when_the_repo_step_is_skipped() -> None:
    without_repo = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=None, max_bundle_mib=20
    )
    headers = re.findall(r"^Step (\d+) - ", without_repo, re.MULTILINE)
    assert headers == ["1", "2", "3", "4"], "steps are renumbered, not left with a hole"


def test_prompt_honours_a_non_default_home_dir() -> None:
    prompt = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID,
        repo_mount_path=REPO,
        max_bundle_mib=20,
        home_dir="/home/claude",
    )
    assert "/home/claude/HANDOFF.md" in prompt, "the note goes in the given home directory"
    assert "/home/claude/uncommitted.patch" in prompt, "so does the patch"
    assert "-C / home/claude mnt/repo/analytics" in prompt, (
        "the tar roots are the given home directory and the repo, relative to /"
    )
    assert "/root/" not in prompt, "the default home directory must not leak in"


def test_prompt_asks_for_output_only_and_forbids_reading_images() -> None:
    prompt = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=REPO, max_bundle_mib=20
    )
    assert "Do not open or read any image file." in prompt, (
        "reading an image is what poisons a session's history"
    )
    assert "no summary, no commentary" in prompt, "the reply is parsed, not read"


def test_prompt_is_deterministic_and_under_the_word_budget() -> None:
    first = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=REPO, max_bundle_mib=20
    )
    second = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=REPO, max_bundle_mib=20
    )
    assert first == second, "the same transfer must produce byte-identical prompts"
    assert len(first.split()) < 600, "a long checkpoint prompt costs tokens on a billed turn"


def test_checkpoint_head_lines_reads_the_two_echoed_hashes() -> None:
    reply = (
        "9f3c2b1a7d5e4f6081920a3b4c5d6e7f80912a3b\n"
        " M packages/core/x.py\n"
        "-rw-r--r-- 1 claude claude 1049089 Sep 13 10:00 bundle.tar.gz\n"
        "9f3c2b1a7d5e4f6081920a3b4c5d6e7f80912a3b\n"
    )
    assert checkpoint_head_lines(reply) == (
        "9f3c2b1a7d5e4f6081920a3b4c5d6e7f80912a3b",
        "9f3c2b1a7d5e4f6081920a3b4c5d6e7f80912a3b",
    ), "both HEAD echoes should be read back"


def test_checkpoint_head_lines_reports_a_changed_head_when_the_session_committed() -> None:
    before, after = checkpoint_head_lines(
        "1111111111111111111111111111111111111111\nok\n2222222222222222222222222222222222222222\n"
    )
    assert before != after, "a differing pair is how a forbidden commit is detected"


def test_checkpoint_head_lines_returns_none_rather_than_guessing() -> None:
    assert checkpoint_head_lines("no hashes here") == (None, None), (
        "no echo means no answer, not an empty string"
    )
    assert checkpoint_head_lines("3333333333333333333333333333333333333333")[1] is None, (
        "one hash cannot prove HEAD was unchanged"
    )


def test_checkpoint_archive_listed_accepts_a_listing_and_rejects_a_command_echo() -> None:
    filename = handoff_filename(TRANSFER_ID)
    listing = f"-rw-r--r-- 1 claude claude 1049089 Sep 13 10:00 /mnt/session/outputs/{filename}"
    assert checkpoint_archive_listed(listing, filename), "an ls line confirms the archive"
    echo = f"  tar czf /mnt/session/outputs/{filename} --exclude='*.env' -C / root"
    assert not checkpoint_archive_listed(echo, filename), (
        "echoing the command back is not evidence the archive exists"
    )
    assert not checkpoint_archive_listed("tar: exiting with failure status", filename), (
        "a failed run must not read as a listed archive"
    )


def test_prompt_captures_uncommitted_changes_when_the_answer_is_copy_or_absent() -> None:
    for unsaved_work in ("copy", None):
        prompt = build_checkpoint_prompt(
            transfer_id=TRANSFER_ID,
            repo_mount_path=REPO,
            max_bundle_mib=20,
            unsaved_work=unsaved_work,
        )
        assert f"git -C {REPO} diff HEAD > /root/uncommitted.patch" in prompt, (
            f"unsaved_work={unsaved_work!r} means capture the work, so the patch step stays"
        )
        assert f"git -C {REPO} ls-files --others --exclude-standard" in prompt, (
            "untracked files are part of the work being captured"
        )
        assert " root mnt/repo/analytics" in prompt, "and the checkout itself travels"


def test_prompt_leaves_uncommitted_changes_behind_when_the_answer_is_leave() -> None:
    prompt = build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=REPO, max_bundle_mib=20, unsaved_work="leave"
    )
    assert "diff HEAD" not in prompt, (
        "the person chose to leave the changes, so nothing captures them as a patch"
    )
    assert "ls-files --others" not in prompt, "nor lists the untracked files to carry"
    assert "-C / root\n" in f"{prompt}\n" and "-C / root mnt" not in prompt, (
        "the checkout must not be tarred either, or the changes would come across anyway"
    )
    assert "find root -type f" in prompt, "and the oversize scan covers only what is archived"
    assert "deliberately being left behind" in prompt, (
        "the prompt has to say the omission is the person's decision, not a failure"
    )
    assert f"git -C {REPO} rev-parse HEAD" in prompt, "HEAD is still recorded"
    assert f"git -C {REPO} status --porcelain" in prompt, "and so is what was left dirty"
    assert "NEVER RUN GIT COMMIT" in prompt, "leaving the work still means changing nothing"


def test_prompt_ignores_the_unsaved_work_answer_when_no_repo_is_mounted() -> None:
    assert build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=None, max_bundle_mib=20, unsaved_work="leave"
    ) == build_checkpoint_prompt(
        transfer_id=TRANSFER_ID, repo_mount_path=None, max_bundle_mib=20
    ), "with no checkout there is nothing to leave in it, so the prompt is unchanged"
