"""Edit a posted control card in place, from the Slack bot process.

The MCP process posts the `requested` card
(`daimon.adapters.mcp.tools.slack._credential_button`); this process owns
every later state of that same message. Both build the card in core and
render it with `build_card_blocks`, so the edit lands the same four slots in
the same order as the initial post rather than collapsing the card to a
one-line marker.

The card's identity is durable: the request row records the channel the card
was posted in and its `ts`, so an edit targets the original card even when
the submission arrives from somewhere else (a different channel's modal, a
restarted bot). A row with no recorded `ts` predates its post or never got
one; there is nothing to edit and nothing to report.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import structlog
from daimon.core.continuity.messages import ConfigurationChange
from daimon.core.credential_requests import split_skill_repo_target
from daimon.core.github_repo_auth import normalize_owner_repo
from daimon.core.posted_controls import (
    CardKind,
    CardState,
    RefusalReason,
    build_card_blocks,
    build_posted_card,
    card_notification_text,
)
from daimon.core.stores.domain import CredentialRequestRow
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

__all__ = ["edit_posted_card"]

log = structlog.get_logger()

_REPO_KINDS: frozenset[str] = frozenset({"repo", "skill_repo"})


def _repo_and_branch(row: CredentialRequestRow) -> tuple[str | None, str | None]:
    """`(owner/repo, branch)` for the two repo kinds, `(None, None)` otherwise.

    The repo kinds pack `repo_url@branch#path` into the row's `target`, which
    is the only record of the branch the card was posted for.
    """
    if row.kind not in _REPO_KINDS:
        return None, None
    repo_url, branch, _ = split_skill_repo_target(row.target)
    return normalize_owner_repo(repo_url), branch


async def edit_posted_card(
    client: AsyncWebClient,
    *,
    row: CredentialRequestRow,
    state: CardState,
    outcome: ConfigurationChange | None = None,
    refusal: RefusalReason | None = None,
    refusal_lines: Sequence[str] = (),
) -> None:
    """Re-render the request's own message into `state`.

    Same four slots in the same order as the initial post. Only the
    `requested` state carries the button, so only that state hands the token
    to the renderer.

    A Slack refusal (`SlackApiError`) is logged rather than raised: a failed
    edit is a downgrade in feedback, not in correctness — the request row is
    already spent, and the lifecycle the card announces has already happened.
    """
    if row.posted_message_id is None:
        return
    repo, branch = _repo_and_branch(row)
    card = build_posted_card(
        kind=cast("CardKind", row.kind),
        state=state,
        agent_name=row.target_name or "the agent",
        responder_name=row.responder_name or "Daimon",
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
    try:
        await client.chat_update(  # pyright: ignore[reportUnknownMemberType]
            channel=row.parent_channel_id or row.channel_id,
            ts=row.posted_message_id,
            text=card_notification_text(card),
            blocks=build_card_blocks(card, token=row.token if state == "requested" else None),
        )
    except SlackApiError as err:
        log.warning(
            "posted_card.edit_failed",
            state=state,
            kind=row.kind,
            error=str(err.response.get("error", "slack_api_error")),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # slack_sdk response is dict-like
        )
