"""Tests for the credential-guidance system preamble.

The preamble tells every agent WHERE its credentials live (env-secret file vs
MA-vault-bound MCP auth) so it stops hallucinating "no key" / hunting for
non-existent MCP keys. It is sentinel-delimited so re-applying replaces the
block instead of stacking — reconcile re-runs and panel edits must be
idempotent or the spec hash never stabilises.
"""

from __future__ import annotations

from daimon.core.agent_guidance import (
    CREDENTIAL_GUIDANCE_BLOCK,
    apply_credential_guidance,
)


def test_prepends_block_when_absent() -> None:
    out = apply_credential_guidance("You are daimon. Be concise.")
    assert CREDENTIAL_GUIDANCE_BLOCK in out, "guidance block must be present after applying"
    assert out.endswith("You are daimon. Be concise."), (
        "original system must be preserved below the block"
    )
    assert "/mnt/session/uploads/.env" in out, "must tell the agent where env secrets are mounted"
    assert "MCP" in out, "must explain MCP vault-bound auth"


def test_idempotent_applied_twice_equals_once() -> None:
    once = apply_credential_guidance("base prompt")
    twice = apply_credential_guidance(once)
    assert twice == once, "re-applying must replace the block, never stack it"


def test_empty_system_yields_only_block() -> None:
    out = apply_credential_guidance("")
    assert CREDENTIAL_GUIDANCE_BLOCK in out, "empty system still gets the guidance block"


def test_carves_out_redacted_self_inspection() -> None:
    # Regression for the QA trace where an agent refused a values-redacted
    # request to confirm whether a key was present, calling it
    # "credential harvesting". The boundary the block must draw is on the
    # secret VALUE leaking, not on reading config at all.
    block = CREDENTIAL_GUIDANCE_BLOCK
    assert "VALUE" in block, "block must distinguish a secret's value from its existence"
    assert "credential harvesting" in block, (
        "block must explicitly defuse the 'credential harvesting' over-refusal"
    )
    assert "REDACTED" in block, "block must endorse value-redacted inspection as safe"


def test_states_the_interactive_delivery_contract_not_only_the_headless_one() -> None:
    # Regression for the staging trace where an agent answered one mention
    # twice: it called send_message into the thread it was already running in,
    # and its ordinary reply was delivered too. The block previously described
    # only the headless-routine path ("a routine must call send_message"), so
    # that was the sole statement about how output reaches Discord and a
    # weakly-primed agent generalised it to interactive turns.
    block = CREDENTIAL_GUIDANCE_BLOCK
    assert "send_message" in block, "block must name the tool the failure mode misuses"
    assert "ROUTINES RUN HEADLESS" in block, "the headless-routine path must still be stated"
    assert block.index("A CHAT REPLY DELIVERS ITSELF") < block.index("ROUTINES RUN HEADLESS"), (
        "the interactive contract must precede the routine exception, so the "
        "default case is read first and the exception reads as an exception"
    )
    assert "Never" in block and "invoked from" in block, (
        "block must forbid send_message into the thread the turn was invoked from"
    )
    assert "interactive exception is Discord file delivery" in block, (
        "the Discord file path must be explicit about overriding the general send_message rule"
    )
    slack = block[block.index("On Slack") : block.index("On Discord")]
    discord = block[block.index("On Discord") : block.index("Calling `read`")]
    assert "/mnt/session/outputs IS the delivery path" in slack
    assert "do NOT also" in slack and "send_message" in slack, (
        "Slack guidance must prevent output-directory files from being sent twice"
    )
    assert "/mnt/session/outputs is NOT a delivery path" in discord
    assert "create_file_upload_url" in discord and "send_message" in discord, (
        "Discord guidance must preserve its explicit file-upload sequence"
    )
    assert "a FILE never does" not in block
    assert "There is no way to attach a file to your reply" not in block


def test_replaces_stale_block_preserving_user_body() -> None:
    seeded = apply_credential_guidance("ORIGINAL BODY")
    # Simulate a user editing only their body underneath the (now stale) block.
    edited = seeded.replace("ORIGINAL BODY", "EDITED BODY")
    reapplied = apply_credential_guidance(edited)
    assert reapplied.count("/mnt/session/uploads/.env") == seeded.count(
        "/mnt/session/uploads/.env"
    ), "must not duplicate the block when one already exists"
    assert reapplied.endswith("EDITED BODY"), "user's edited body must be preserved"
