"""Re-render a posted control card in place, in the bot process.

The card the MCP process posted is the only durable record of a private-value
request: the ephemeral reply a submitter gets is gone on refresh and was never
visible to anyone else. So every state the request reaches — received, then
applied, partial, refused, superseded or expired — is written back onto that
same message.

Two platform facts shape this module. A components-v2 message can never be
edited back to a classic `View` (Discord answers 50035), so every edit here
passes a `LayoutView`, no exceptions. And the initial post is the one and only
ping in the lifecycle: the footer mention on the `requested` card. Every edit
therefore goes out with `AllowedMentions.none()` — re-rendering must not
re-ping the requester each time the state moves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import structlog
from daimon.adapters.discord.posted_controls.view import build_card_view
from daimon.core.continuity.messages import ConfigurationChange
from daimon.core.credential_requests import split_skill_repo_target
from daimon.core.github_repo_auth import normalize_owner_repo
from daimon.core.posted_controls import (
    CardKind,
    CardState,
    PostedCard,
    RefusalReason,
    build_posted_card,
)
from daimon.core.stores.domain import CredentialRequestRow

import discord

__all__ = ["edit_posted_card"]

_log = structlog.get_logger()

#: Fallbacks for the two display names a row may not carry. A row minted
#: before the agent was resolved names no target; one minted outside a turn
#: names no responder.
_UNNAMED_AGENT = "the agent"
_UNNAMED_RESPONDER = "Daimon"


def _card_for_row(
    row: CredentialRequestRow,
    *,
    state: CardState,
    outcome: ConfigurationChange | None,
    refusal: RefusalReason | None,
    refusal_lines: Sequence[str],
) -> PostedCard:
    """Rebuild this request's card in `state` from the row alone.

    The row is the whole input on purpose: the edit runs in a different
    process from the post and long after it, so anything the card needs has
    to be recoverable from the durable request. `target` packs the branch for
    the two repo kinds, which is why it is unpacked rather than read off a
    separate column.
    """
    kind = cast("CardKind", row.kind)
    repo: str | None = None
    branch: str | None = None
    if kind in ("repo", "skill_repo"):
        url, branch, _path = split_skill_repo_target(row.target)
        repo = normalize_owner_repo(url)
    return build_posted_card(
        kind=kind,
        state=state,
        agent_name=row.target_name or _UNNAMED_AGENT,
        responder_name=row.responder_name or _UNNAMED_RESPONDER,
        target=row.target,
        requester_platform_user_id=row.requester_platform_user_id,
        expires_at=row.expires_at,
        token=row.token,
        mcp_server_url=row.mcp_server_url,
        repo=repo,
        branch=branch,
        outcome=outcome,
        refusal=refusal,
        refusal_lines=refusal_lines,
    )


async def edit_posted_card(
    client: discord.Client,
    *,
    row: CredentialRequestRow,
    state: CardState,
    outcome: ConfigurationChange | None = None,
    refusal: RefusalReason | None = None,
    refusal_lines: Sequence[str] = (),
) -> None:
    """Re-render the request's own message into `state`.

    Never raises. By the time this runs the state it is announcing is already
    durable — the row is consumed, the write has landed or been refused — so
    an edit that cannot be delivered (message deleted, thread archived,
    permissions lost) is a downgrade in feedback, not in correctness.

    Returns silently for a row with no posted message to edit; the ids are
    recorded right after the post, so a row without them never had a card.
    """
    if row.origin_thread_id is None or row.posted_message_id is None:
        return
    view = build_card_view(
        _card_for_row(
            row, state=state, outcome=outcome, refusal=refusal, refusal_lines=refusal_lines
        )
    )
    message = client.get_partial_messageable(int(row.origin_thread_id)).get_partial_message(
        int(row.posted_message_id)
    )
    try:
        await message.edit(view=view, allowed_mentions=discord.AllowedMentions.none())
    except discord.HTTPException as err:
        _log.warning(
            "posted_card.edit_failed", err_type=type(err).__name__, state=state, kind=row.kind
        )
