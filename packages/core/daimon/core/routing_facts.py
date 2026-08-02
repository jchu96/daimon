"""The routing truth: mention gating and one-bot-not-one-per-agent.

These sentences exist here, once, because both the chat tool that sets a
default (`set_agent_default` / `clear_agent_default`) and the setup panel that
sets one must say the same thing about how routing actually works — and a
claim about routing that lives in prose somewhere else drifts from the code
that makes it true. The deliberate design is for the product to answer the
routing question at the moment the belief forms (right after a default is
set or cleared), rather than for a system prompt to assert it up front.

Both functions are pure: no I/O, no clock, same inputs always produce the
same output.
"""

from __future__ import annotations


def build_set_default_note(*, agent_name: str, scope_label: str) -> str:
    """Routing truth shown after a default is set at ``scope_label``.

    States both facts: members reach ``agent_name`` only by @mentioning the
    bot in the channel, and there is one bot user for the whole workspace —
    not one per agent — so setting a default did not add a new bot member.
    """
    return (
        f"{agent_name} is now the default for {scope_label}. Members still "
        f"reach it only by @mentioning the bot in that channel — there is one "
        f"bot for the whole workspace, not one bot per agent, so this did not "
        f"add a new member to the server."
    )


def build_clear_default_note(*, scope_label: str, cleared: bool) -> str:
    """Routing truth shown after a default is cleared at ``scope_label``.

    When ``cleared`` is False, ``scope_label`` had no default and nothing
    changed. Otherwise states that ``scope_label`` no longer has a default,
    that resolution now falls through to the next tier, and repeats the
    mention requirement.
    """
    if not cleared:
        return f"{scope_label} had no default agent to clear — nothing changed."
    return (
        f"{scope_label} no longer has a default agent; resolution now falls "
        f"through to the next tier. Whichever agent answers is still reached "
        f"only by @mentioning the bot in that channel — there is one bot for "
        f"the whole workspace, not one bot per agent."
    )
