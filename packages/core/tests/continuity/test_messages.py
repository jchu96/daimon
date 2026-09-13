"""Exact-string tests for `daimon.core.continuity.messages` and `tool_messages`.

Every expected string is a literal, not re-derived from the module's own
`"\n".join` templates — otherwise a bug in the template would pass its own
test. Values (`"Ada"`, `"acme/widgets"`, ...) are arbitrary stand-ins for the
placeholders documented in the setup copy conventions.
"""

from __future__ import annotations

import itertools

import pytest
from daimon.core.continuity.messages import (
    FORBIDDEN_TOKENS,
    ChangeAvailability,
    ChangeKind,
    ConfigurationChange,
    render_change_confirmation,
    render_current_work_must_finish,
    render_fresh_start,
    render_handoff_acknowledged,
    render_preparation_failed,
    render_replacement_summary,
    render_responder_changed_without_handoff,
    render_unexpected_loss,
    render_unsaved_work_question,
)
from daimon.core.continuity.tool_messages import (
    render_tool_refusal_setup_thread,
    render_tool_refusal_unreachable,
    render_tool_unsaved_work_question,
)

# --- A. key / keys_bulk / key_removed ---------------------------------------


def test_render_change_confirmation_returns_two_lines_when_key_saved() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="key", availability="saved", detail="TOGGL_TOKEN"
    )
    assert render_change_confirmation(change) == (
        "TOGGL_TOKEN saved for Ada.\nAnyone who talks to Ada can use it."
    ), "saved availability should render exactly two lines"


def test_render_change_confirmation_returns_three_lines_when_key_ready_now() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="key", availability="ready_now", detail="TOGGL_TOKEN"
    )
    assert render_change_confirmation(change) == (
        "TOGGL_TOKEN saved for Ada.\nAda can use it now.\nAnyone who talks to Ada can use it."
    ), "ready_now availability should insert the 'can use it now' line"


def test_render_change_confirmation_returns_three_lines_when_key_next_message() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="key", availability="next_message", detail="TOGGL_TOKEN"
    )
    assert render_change_confirmation(change) == (
        "TOGGL_TOKEN saved for Ada.\n"
        "Ada can use it from your next message here.\n"
        "Anyone who talks to Ada can use it."
    ), "next_message availability should insert the 'from your next message' line"


def test_render_change_confirmation_renders_saved_only_lines_when_key_preparation_failed() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="key", availability="preparation_failed", detail="TOGGL_TOKEN"
    )
    assert render_change_confirmation(change) == (
        "TOGGL_TOKEN saved for Ada.\nAnyone who talks to Ada can use it."
    ), "preparation_failed for kind='key' should render the same two lines as 'saved'"


def test_render_change_confirmation_pluralizes_bulk_keys_when_count_greater_than_one() -> None:
    change = ConfigurationChange(target_name="Ada", kind="keys_bulk", availability="saved", count=3)
    assert render_change_confirmation(change) == (
        "3 keys saved for Ada.\nAnyone who talks to Ada can use it."
    ), "count > 1 should use the plural 'keys' noun"


def test_render_change_confirmation_singularizes_bulk_keys_when_count_is_one() -> None:
    change = ConfigurationChange(target_name="Ada", kind="keys_bulk", availability="saved", count=1)
    assert render_change_confirmation(change) == (
        "1 key saved for Ada.\nAnyone who talks to Ada can use it."
    ), "count == 1 should use the singular 'key' noun, not '1 keys'"


def test_render_change_confirmation_renders_bulk_keys_ready_now_as_three_lines() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="keys_bulk", availability="ready_now", count=3
    )
    assert render_change_confirmation(change) == (
        "3 keys saved for Ada.\nAda can use it now.\nAnyone who talks to Ada can use it."
    ), "keys_bulk should follow the same availability lines 2-3 as a single key"


def test_render_change_confirmation_renders_bulk_keys_next_message_as_three_lines() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="keys_bulk", availability="next_message", count=3
    )
    assert render_change_confirmation(change) == (
        "3 keys saved for Ada.\n"
        "Ada can use it from your next message here.\n"
        "Anyone who talks to Ada can use it."
    ), "keys_bulk should follow the same availability lines 2-3 as a single key"


def test_render_change_confirmation_renders_key_removed_regardless_of_availability() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="key_removed", availability="saved", detail="TOGGL_TOKEN"
    )
    assert render_change_confirmation(change) == (
        "TOGGL_TOKEN removed from Ada.\n"
        "It stops being supplied from your next message here.\n"
        "Work already running with it is not stopped, and it is not cancelled at the service."
    ), "key_removed copy is fixed and does not vary by availability"


# --- B. model ----------------------------------------------------------------


def test_render_change_confirmation_renders_model_changed() -> None:
    change = ConfigurationChange(target_name="Ada", kind="model", availability="next_message")
    assert render_change_confirmation(change) == (
        "Ada will use the new model starting with your next message.\n"
        "Your task, decisions and working files stay as they are."
    ), "model change copy should not depend on availability or detail"


# --- C. instructions / skill / mcp -------------------------------------------


def test_render_change_confirmation_renders_instructions_updated() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="instructions", availability="next_message"
    )
    assert render_change_confirmation(change) == (
        "Ada's instructions are updated.\nIt uses them from your next message here."
    ), "instructions copy should not depend on availability"


def test_render_change_confirmation_renders_skill_added() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="skill", availability="next_message", detail="pdf-tools"
    )
    assert render_change_confirmation(change) == (
        "Ada has the pdf-tools skill.\nIt can use it from your next message here."
    ), "skill copy should name the skill from detail"


def test_render_change_confirmation_renders_skill_removed() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="skill_removed", availability="next_message", detail="pdf-tools"
    )
    assert render_change_confirmation(change) == (
        "Ada no longer has the pdf-tools skill.\nThe change applies from your next message here."
    ), "skill_removed copy should name the skill from detail"


def test_render_change_confirmation_renders_mcp_connected() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="mcp", availability="next_message", detail="Toggl"
    )
    assert render_change_confirmation(change) == (
        "Ada is connected to Toggl.\nIts tools are available from your next message here."
    ), "mcp copy should name the service from detail"


def test_render_change_confirmation_renders_mcp_removed() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="mcp_removed", availability="next_message", detail="Toggl"
    )
    assert render_change_confirmation(change) == (
        "Ada is no longer connected to Toggl.\nThe change applies from your next message here."
    ), "mcp_removed copy should name the service from detail"


def test_render_change_confirmation_renders_mcp_preparation_failed_as_partial_connection() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="mcp", availability="preparation_failed", detail="Toggl"
    )
    assert render_change_confirmation(change) == (
        "Toggl token saved for Ada.\n"
        "The connection did not finish, so its tools are not available yet.\n"
        "Ask me to connect Toggl again to retry."
    ), "mcp preparation_failed is the one kind with its own partial-connection text"


# --- D. repo / branch ---------------------------------------------------------


def test_render_change_confirmation_renders_repo_token_only_when_repo_and_branch_are_none() -> None:
    change = ConfigurationChange(target_name="Ada", kind="repo", availability="next_message")
    assert render_change_confirmation(change) == (
        "Your GitHub token is saved for Ada.\n"
        "No repo is pinned yet.\n"
        "Ada's GitHub connection uses it from your next message here."
    ), "repo kind with no repo/branch should render the token-only text"


def test_render_change_confirmation_renders_repo_switch_without_unsaved_work() -> None:
    change = ConfigurationChange(
        target_name="Ada",
        kind="repo",
        availability="next_message",
        repo="acme/widgets",
        branch="main",
    )
    assert render_change_confirmation(change) == (
        "Ada now works in acme/widgets on main.\n"
        "It switches to that checkout from your next message here.\n"
        "Your conversation and working files come with you."
    ), "a plain repo switch should not mention uncommitted changes"


def test_render_change_confirmation_renders_repo_switch_after_copy() -> None:
    change = ConfigurationChange(
        target_name="Ada",
        kind="repo",
        availability="next_message",
        repo="acme/widgets",
        branch="main",
        unsaved_work="copy",
        copied_file_count=5,
    )
    assert render_change_confirmation(change) == (
        "Ada now works in acme/widgets on main.\n"
        "I copied 5 changed files into your working files first; nothing was committed or pushed.\n"
        "It switches to that checkout from your next message here."
    ), "unsaved_work='copy' should report the copied file count"


def test_render_change_confirmation_renders_repo_switch_after_leave() -> None:
    change = ConfigurationChange(
        target_name="Ada",
        kind="repo",
        availability="next_message",
        repo="acme/widgets",
        branch="main",
        unsaved_work="leave",
    )
    assert render_change_confirmation(change) == (
        "Ada now works in acme/widgets on main.\n"
        "The uncommitted changes stay in the old checkout and do not come across.\n"
        "It switches to that checkout from your next message here."
    ), "unsaved_work='leave' should say the changes stay behind"


def test_render_unsaved_work_question_renders_four_lines() -> None:
    assert render_unsaved_work_question("acme/widgets") == (
        "There are uncommitted changes in acme/widgets.\n"
        "I can copy them into your working files before switching, or leave them where they are.\n"
        "Nothing is committed or pushed either way.\n"
        "Which would you like?"
    ), "the unsaved-work question is always exactly these four lines"


# --- E. environment ------------------------------------------------------------


def test_render_change_confirmation_renders_environment_changed() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="environment", availability="next_message", detail="prod"
    )
    assert render_change_confirmation(change) == (
        "Ada runs in the prod environment from your next message here.\n"
        "Your conversation, decisions and working files come with you.\n"
        "Anything still running stops. I cannot carry a running process or notebook kernel across."
    ), "environment copy should warn that running work stops"


# --- F. handoff acknowledged -----------------------------------------------------


def test_render_handoff_acknowledged_without_requested_work() -> None:
    assert render_handoff_acknowledged(
        target_name="Ada", from_name="Rex", channel="#data", requested_work=None
    ) == (
        "Ada takes over this task from your next message here.\n"
        "Your conversation, decisions and working files come with it.\n"
        "Ada uses its own keys, connections and memory, not Rex's.\n"
        "Who answers in #data is unchanged."
    ), "with no requested work there should be exactly four lines"


def test_render_handoff_acknowledged_appends_requested_work_line() -> None:
    assert render_handoff_acknowledged(
        target_name="Ada", from_name="Rex", channel="#data", requested_work="finish the report"
    ) == (
        "Ada takes over this task from your next message here.\n"
        "Your conversation, decisions and working files come with it.\n"
        "Ada uses its own keys, connections and memory, not Rex's.\n"
        "Who answers in #data is unchanged.\n"
        "It will pick up with: finish the report."
    ), "requested_work should append a fifth line verbatim"


# --- H. fresh start --------------------------------------------------------------


def test_render_fresh_start() -> None:
    assert render_fresh_start("Ada") == (
        "Starting fresh from your next message here.\n"
        "Leaves behind: this task's working files and unfinished work.\n"
        "Keeps: everything already posted in this thread, and Ada's saved memory, "
        "keys and connections.\n"
        "Nothing is removed until the new workspace is ready."
    ), "fresh start copy is fixed four lines naming the target"


# --- I. preparation failure -------------------------------------------------------


def test_render_preparation_failed() -> None:
    assert render_preparation_failed("Ada") == (
        "I could not get Ada ready with the latest setup, so I have not started this message.\n"
        "What was saved is still saved.\n"
        "Your task, decisions and working files are unchanged.\n"
        "Mention me again to retry."
    ), "preparation failure copy states the turn did not run"


# --- J. unexpected session loss ----------------------------------------------------


def test_render_unexpected_loss_transcript() -> None:
    assert render_unexpected_loss("transcript") == (
        "I lost the workspace this task was running in and started a new one.\n"
        "I have this thread's conversation and the files that were saved to your task.\n"
        "Anything unsaved in the old workspace is gone, and nothing that was running "
        "came across.\n"
        "Tell me what to re-check and I'll go from there."
    ), "transcript recovery should mention the saved task files"


def test_render_unexpected_loss_history() -> None:
    assert render_unexpected_loss("history") == (
        "I lost the workspace this task was running in and started a new one.\n"
        "I have what was posted in this thread, but not the earlier conversation.\n"
        "Anything unsaved in the old workspace is gone, and nothing that was running "
        "came across.\n"
        "Tell me what to re-check and I'll go from there."
    ), "history-only recovery should say the earlier conversation is missing"


# --- K. current work must finish first ----------------------------------------------


def test_render_current_work_must_finish_for_a_plain_config_change() -> None:
    assert render_current_work_must_finish("Ada", handoff=False) == (
        "Ada is still working on the previous message here.\n"
        "Your change is saved and it picks it up on the next message, not that one."
    ), "a plain config change should say it applies on the next message"


def test_render_current_work_must_finish_for_a_handoff() -> None:
    assert render_current_work_must_finish("Ada", handoff=True) == (
        "Ada takes over from your next message here.\n"
        "The message I'm working on now finishes with me."
    ), "a handoff should say the in-flight message finishes with the current responder"


# --- L. responder changed without handoff -------------------------------------------


def test_render_responder_changed_without_handoff() -> None:
    assert render_responder_changed_without_handoff(
        new_responder="Nova", owner="Ada", channel="#data"
    ) == (
        "Nova now answers in #data, but this conversation's work belongs to Ada.\n"
        'Say "have Nova take over this task" and I\'ll move the conversation and '
        "working files across.\n"
        "Or start a new thread to begin fresh with Nova."
    ), "the quoted trigger phrase must name the new responder verbatim"


# --- M. planned replacement summary --------------------------------------------------


def test_render_replacement_summary_full_with_nothing_lost() -> None:
    assert render_replacement_summary("full", []) == (
        "Your conversation, decisions and working files came across."
    ), "full transfer with nothing lost is a single line"


def test_render_replacement_summary_transcript_with_nothing_lost() -> None:
    assert render_replacement_summary("transcript", []) == (
        "Your conversation and decisions came across; the working files could not be "
        "saved from the old workspace."
    ), "transcript transfer explains why the working files are missing"


def test_render_replacement_summary_history_with_nothing_lost() -> None:
    assert render_replacement_summary("history", []) == (
        "Only what was posted in this thread came across."
    ), "history transfer is the narrowest summary"


def test_render_replacement_summary_appends_not_carried_line_when_lost_is_nonempty() -> None:
    assert render_replacement_summary("full", ["memory"]) == (
        "Your conversation, decisions and working files came across.\nNot carried: memory."
    ), "a single lost item should still get its own 'Not carried' line"


def test_render_replacement_summary_joins_multiple_lost_items_with_commas() -> None:
    assert render_replacement_summary("history", ["memory", "keys"]) == (
        "Only what was posted in this thread came across.\nNot carried: memory, keys."
    ), "multiple lost items should be comma-joined on the 'Not carried' line"


# --- Tool-facing (ToolError) copy -----------------------------------------------------


def test_render_tool_refusal_unreachable() -> None:
    assert render_tool_refusal_unreachable("Ada", "#data") == (
        "'Ada' does not answer anywhere in this workspace, so it cannot be handed a task.\n"
        "Tell the caller an admin can say: make Ada answer in #data. Then the handoff "
        "will work.\n"
        "Nothing was changed. Do not retry."
    ), "the unreachable refusal should name the admin fix and forbid retrying"


def test_render_tool_refusal_setup_thread() -> None:
    assert render_tool_refusal_setup_thread("Ada") == (
        "Setup conversations always answer as Daimon, so a task cannot be handed over here.\n"
        "Tell the caller to ask Ada in a channel where it answers, or to start a thread there.\n"
        "Nothing was changed. Do not retry."
    ), "the setup-thread refusal should redirect to a channel where the target answers"


def test_render_tool_unsaved_work_question() -> None:
    assert render_tool_unsaved_work_question("acme/widgets") == (
        "Switching would change the checkout and there are uncommitted changes in "
        "acme/widgets.\n"
        "Ask the caller this question and nothing else, then call hand_off_task again "
        "with unsaved_work:\n"
        '"There are uncommitted changes in acme/widgets. I can copy them into your '
        "working files before switching, or leave them where they are. Nothing is "
        'committed or pushed either way. Which would you like?"\n'
        "Nothing was changed."
    ), "the tool-facing question must embed the D. question verbatim on one line"


def test_render_unsaved_work_question_and_tool_version_share_the_same_question_text() -> None:
    person_question = render_unsaved_work_question("acme/widgets")
    tool_text = render_tool_unsaved_work_question("acme/widgets")
    one_line_person_question = " ".join(person_question.split("\n"))
    assert f'"{one_line_person_question}"' in tool_text, (
        "the tool-facing question must quote the exact person-facing question, "
        "joined onto a single line"
    )


# --- ValueError on caller misuse ------------------------------------------------------


def test_render_change_confirmation_raises_when_key_missing_detail() -> None:
    change = ConfigurationChange(target_name="Ada", kind="key", availability="saved")
    with pytest.raises(ValueError, match="requires detail"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_keys_bulk_missing_count() -> None:
    change = ConfigurationChange(target_name="Ada", kind="keys_bulk", availability="saved")
    with pytest.raises(ValueError, match="requires count"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_keys_bulk_count_is_zero() -> None:
    change = ConfigurationChange(target_name="Ada", kind="keys_bulk", availability="saved", count=0)
    with pytest.raises(ValueError, match=">= 1"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_keys_bulk_has_detail() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="keys_bulk", availability="saved", count=1, detail="TOGGL_TOKEN"
    )
    with pytest.raises(ValueError, match="does not use detail"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_key_removed_missing_detail() -> None:
    change = ConfigurationChange(target_name="Ada", kind="key_removed", availability="saved")
    with pytest.raises(ValueError, match="requires detail"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_model_has_detail() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="model", availability="saved", detail="unused"
    )
    with pytest.raises(ValueError, match="does not use detail"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_instructions_has_detail() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="instructions", availability="saved", detail="unused"
    )
    with pytest.raises(ValueError, match="does not use detail"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_skill_missing_detail() -> None:
    change = ConfigurationChange(target_name="Ada", kind="skill", availability="saved")
    with pytest.raises(ValueError, match="requires detail"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_skill_removed_missing_detail() -> None:
    change = ConfigurationChange(target_name="Ada", kind="skill_removed", availability="saved")
    with pytest.raises(ValueError, match="requires detail"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_mcp_missing_detail() -> None:
    change = ConfigurationChange(target_name="Ada", kind="mcp", availability="saved")
    with pytest.raises(ValueError, match="requires detail"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_mcp_removed_missing_detail() -> None:
    change = ConfigurationChange(target_name="Ada", kind="mcp_removed", availability="saved")
    with pytest.raises(ValueError, match="requires detail"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_environment_missing_detail() -> None:
    change = ConfigurationChange(target_name="Ada", kind="environment", availability="saved")
    with pytest.raises(ValueError, match="requires detail"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_repo_has_detail() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="repo", availability="saved", detail="unused"
    )
    with pytest.raises(ValueError, match="does not use detail"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_repo_branch_set_without_repo() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="repo", availability="saved", branch="main"
    )
    with pytest.raises(ValueError, match="repo and branch together"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_token_only_repo_sets_unsaved_work() -> None:
    change = ConfigurationChange(
        target_name="Ada", kind="repo", availability="saved", unsaved_work="leave"
    )
    with pytest.raises(ValueError, match="token-only"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_copy_missing_copied_file_count() -> None:
    change = ConfigurationChange(
        target_name="Ada",
        kind="repo",
        availability="saved",
        repo="acme/widgets",
        branch="main",
        unsaved_work="copy",
    )
    with pytest.raises(ValueError, match="requires copied_file_count"):
        render_change_confirmation(change)


def test_render_change_confirmation_raises_when_copied_file_count_set_without_copy() -> None:
    change = ConfigurationChange(
        target_name="Ada",
        kind="repo",
        availability="saved",
        repo="acme/widgets",
        branch="main",
        copied_file_count=3,
    )
    with pytest.raises(ValueError, match="only valid with unsaved_work='copy'"):
        render_change_confirmation(change)


# --- Sweep: no forbidden tokens, no trailing newline, no blank lines ------------------

_ALL_AVAILABILITIES: tuple[ChangeAvailability, ...] = (
    "saved",
    "ready_now",
    "next_message",
    "preparation_failed",
)


def _valid_matrix() -> list[ConfigurationChange]:
    """Every kind x availability combination that is valid input, one instance each."""
    changes: list[ConfigurationChange] = []
    kinds_and_detail: dict[ChangeKind, str | None] = {
        "key": "TOGGL_TOKEN",
        "key_removed": "TOGGL_TOKEN",
        "model": None,
        "instructions": None,
        "skill": "pdf-tools",
        "skill_removed": "pdf-tools",
        "mcp": "Toggl",
        "mcp_removed": "Toggl",
        "environment": "prod",
    }
    for (kind, detail), availability in itertools.product(
        kinds_and_detail.items(), _ALL_AVAILABILITIES
    ):
        changes.append(
            ConfigurationChange(
                target_name="Ada", kind=kind, availability=availability, detail=detail
            )
        )
    for availability in _ALL_AVAILABILITIES:
        changes.append(
            ConfigurationChange(
                target_name="Ada", kind="keys_bulk", availability=availability, count=3
            )
        )
        changes.append(
            ConfigurationChange(
                target_name="Ada", kind="keys_bulk", availability=availability, count=1
            )
        )
        changes.append(
            ConfigurationChange(target_name="Ada", kind="repo", availability=availability)
        )
        changes.append(
            ConfigurationChange(
                target_name="Ada",
                kind="repo",
                availability=availability,
                repo="acme/widgets",
                branch="main",
            )
        )
        changes.append(
            ConfigurationChange(
                target_name="Ada",
                kind="repo",
                availability=availability,
                repo="acme/widgets",
                branch="main",
                unsaved_work="copy",
                copied_file_count=5,
            )
        )
        changes.append(
            ConfigurationChange(
                target_name="Ada",
                kind="repo",
                availability=availability,
                repo="acme/widgets",
                branch="main",
                unsaved_work="leave",
            )
        )
    return changes


def _assert_clean(label: str, rendered: str) -> None:
    assert not rendered.endswith("\n"), f"{label} must not end in a trailing newline"
    lines = rendered.split("\n")
    for line in lines:
        assert line != "", f"{label} must not contain an empty line: {rendered!r}"
    lowered = rendered.lower()
    for token in FORBIDDEN_TOKENS:
        assert token not in lowered, (
            f"{label} must not contain forbidden token {token!r}: {rendered!r}"
        )


def test_render_change_confirmation_matrix_never_contains_forbidden_tokens_or_blank_lines() -> None:
    for change in _valid_matrix():
        rendered = render_change_confirmation(change)
        _assert_clean(f"render_change_confirmation({change!r})", rendered)


def test_other_render_functions_never_contain_forbidden_tokens_or_blank_lines() -> None:
    other_rendered: dict[str, str] = {
        "render_unsaved_work_question": render_unsaved_work_question("acme/widgets"),
        "render_handoff_acknowledged (no work)": render_handoff_acknowledged(
            target_name="Ada", from_name="Rex", channel="#data", requested_work=None
        ),
        "render_handoff_acknowledged (with work)": render_handoff_acknowledged(
            target_name="Ada", from_name="Rex", channel="#data", requested_work="finish the report"
        ),
        "render_fresh_start": render_fresh_start("Ada"),
        "render_preparation_failed": render_preparation_failed("Ada"),
        "render_unexpected_loss (transcript)": render_unexpected_loss("transcript"),
        "render_unexpected_loss (history)": render_unexpected_loss("history"),
        "render_current_work_must_finish (config)": render_current_work_must_finish(
            "Ada", handoff=False
        ),
        "render_current_work_must_finish (handoff)": render_current_work_must_finish(
            "Ada", handoff=True
        ),
        "render_responder_changed_without_handoff": render_responder_changed_without_handoff(
            new_responder="Nova", owner="Ada", channel="#data"
        ),
        "render_replacement_summary (full)": render_replacement_summary("full", []),
        "render_replacement_summary (transcript)": render_replacement_summary("transcript", []),
        "render_replacement_summary (history, lost)": render_replacement_summary(
            "history", ["memory", "keys"]
        ),
        "render_tool_refusal_unreachable": render_tool_refusal_unreachable("Ada", "#data"),
        "render_tool_refusal_setup_thread": render_tool_refusal_setup_thread("Ada"),
        "render_tool_unsaved_work_question": render_tool_unsaved_work_question("acme/widgets"),
    }
    for label, rendered in other_rendered.items():
        _assert_clean(label, rendered)
