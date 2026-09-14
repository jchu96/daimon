from __future__ import annotations

from pathlib import Path

from daimon.core.defaults.loader import load_agent_specs, load_skill_paths, load_skill_spec

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULTS = REPO_ROOT / "defaults"

# The posted card (both Discord and Slack) already states the target, the
# requester restriction, the expiry, and who can use the result afterwards.
# The conversational reply must not restate any of these card facts.
FORBIDDEN_CARD_FACTS = (
    "expires in 30 minutes",
    "Only the requester",
    "only you can open",
    "Anyone who talks to",
)


def _daimon_system() -> str:
    specs = load_agent_specs(DEFAULTS / "agents")
    daimon = next(s for s in specs if s.name == "daimon")
    system = daimon.system or ""
    assert system, "seeded daimon agent must have a system prompt"
    return system


def _workspace_setup_body() -> str:
    for skill_dir in load_skill_paths(DEFAULTS / "skills"):
        spec, body = load_skill_spec(skill_dir)
        if spec.name == "workspace-setup":
            return body
    raise AssertionError("defaults/skills/workspace-setup must exist")


def _section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def test_daimon_key_only_reply_shape_does_not_duplicate_card_facts() -> None:
    """The key-only reply shape must point at the card, not restate it."""
    system = _daimon_system()
    reply_shape = _section(system, "Key-only request:", "\n\n")
    for phrase in FORBIDDEN_CARD_FACTS:
        assert phrase not in reply_shape, (
            f"daimon.yaml key-only reply shape restates card fact {phrase!r}; "
            "the posted card already carries it"
        )


def test_workspace_setup_keys_reply_shape_does_not_duplicate_card_facts() -> None:
    """The Keys step's key-only guidance must point at the card, not restate it."""
    body = _workspace_setup_body()
    keys_section = _section(body, "2. **Keys.**", "3. **Skills.**")
    for phrase in FORBIDDEN_CARD_FACTS:
        assert phrase not in keys_section, (
            f"workspace-setup Keys section restates card fact {phrase!r}; "
            "the posted card already carries it"
        )


def test_guidance_names_pending_task_and_bind_public_repo() -> None:
    """The model must be told to pass pending_task and to use bind_public_repo."""
    combined = _daimon_system() + "\n" + _workspace_setup_body()
    assert "pending_task" in combined, (
        "guidance must tell the model to pass pending_task so a waiting task resumes "
        "on its own after the value is saved"
    )
    assert "bind_public_repo" in combined, (
        "guidance must name bind_public_repo for a named agent's public GitHub repo"
    )


def test_skill_repo_guidance_no_longer_claims_a_working_repo_change() -> None:
    """Skill-repo separation landed: importing skills must not claim to touch the working repo."""
    body = _workspace_setup_body()
    skills_section = _section(body, "3. **Skills.**", "4. **MCP servers.**")
    for phrase in ("also binds the target's working repo", "working-repo binding"):
        assert phrase not in skills_section, (
            f"skill-repo guidance still claims {phrase!r}; a skill-repo token no longer "
            "rewrites the working repo binding"
        )
    assert "never changes the working repo" in skills_section, (
        "skill-repo guidance must say importing skills never changes the working repo"
    )


def test_after_confirmed_save_notes_next_message_availability_and_card_result() -> None:
    """The after-save paragraph must not claim a saved key works immediately or need paraphrasing."""
    body = _workspace_setup_body()
    save_section = _section(body, "After a confirmed save,", "## Which agent answers where")
    assert "next message" in save_section, (
        "guidance must say a saved key becomes usable from the next message, not this one"
    )
    assert "paraphrase" in save_section, (
        "guidance must say not to paraphrase the card's result line"
    )
