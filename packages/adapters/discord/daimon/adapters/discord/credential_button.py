"""CredentialRequestButton -- persistent, per-request credential button.

The agent posts this button from the MCP process (Cloud Run); the click is
dispatched to the Discord bot process (worker VM) — a different Python
process entirely. Discord routes interactions by bot application id, not by
which process sent the message, so both processes only need to authenticate
with the same bot token; no other coordination is required.

A per-request button cannot be served by a single long-lived View instance
registered with `bot.add_view()` — that call only registers one already-built
instance's fixed custom_id, and this button's custom_id is minted fresh per
request. `discord.ui.DynamicItem` is the primitive that matches a *template*
regex against an arbitrary future custom_id and reconstructs per-click state
via `from_custom_id`, so this is the one place in the Discord adapter that
carries a persistent, dispatchable custom_id — everywhere else stays
non-persistent (see `views.py`'s module docstring for the documented
exception).

Authorization is requester-only: `interaction_check` compares the clicking
user's id against the id the request was minted for. There is deliberately
NO admin gate here — the setup panel's mutation buttons are admin-only, but
this flow intentionally widens that to "whoever the agent was talking to"
for one-click UX. That makes the requester comparison in `interaction_check`
the ONLY authorization on this write path. Do not "restore" an admin check
here without a deliberate decision to change that.

discord.py's own dispatcher swallows every exception raised inside
`from_custom_id`, `interaction_check`, and `callback` (logs internally and
returns) — verified against the installed `discord/ui/view.py`. That makes
those three methods a genuine adapter boundary: the ordinary
let-failures-propagate rule does not apply here, because propagating would
mean the exception vanishes into a dispatcher we don't control and the user
sees nothing but a stuck interaction. Each method below catches narrowly (or,
for the check/callback bodies, catches broadly with a structlog record and an
ephemeral reply) specifically because this is the last point before that
swallowing dispatcher.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Self, cast

import structlog
from daimon.adapters.discord.bot import DaimonBot
from daimon.adapters.discord.credential_modals import EnvCredentialModal, McpCredentialModal
from daimon.core.credential_requests import (
    CUSTOM_ID_TEMPLATE,
    CredentialRequestKind,
    build_button_label,
    build_custom_id,
)
from daimon.core.stores.credential_requests import peek_credential_request
from daimon.core.stores.domain import CredentialRequestRow
from sqlalchemy.exc import SQLAlchemyError

import discord
from discord.ext import commands

_log = structlog.get_logger()

# Shown when the lookup in from_custom_id found nothing (unknown token or a
# DB failure) — the real label (naming the exact target) only exists once a
# row is found, so this is the best available fallback for a dead button.
_FALLBACK_LABEL = "Add credential"

_NO_LONGER_VALID = "This request is no longer valid — ask again."
_WRONG_REQUESTER = "This request was for someone else — ask again in your own thread."
_EXPIRED = "This request expired — ask again."
_ALREADY_USED = "This request was already used — ask again."
_CHECK_FAILED = "Something went wrong checking this request — please try again."
_CALLBACK_FAILED = "Something went wrong opening this form — please try again."


class CredentialRequestButton(
    discord.ui.DynamicItem[discord.ui.Button[discord.ui.View]], template=CUSTOM_ID_TEMPLATE
):
    """Persistent, per-request credential button reconstructed from its custom_id.

    Registered ONCE as a class (not per-button) via `client.add_dynamic_items`
    in `bot.py`'s `setup_hook`. See the module docstring for why authorization
    here is requester-only with no admin gate, and why the three dispatched
    methods below catch exceptions themselves rather than letting them
    propagate.
    """

    def __init__(self, *, token: str, label: str, request_row: CredentialRequestRow | None) -> None:
        button: discord.ui.Button[discord.ui.View] = discord.ui.Button(
            style=discord.ButtonStyle.primary,
            label=label,
            custom_id=build_custom_id(token),
        )
        super().__init__(button)
        self.token = token
        self.request_row = request_row

    @classmethod
    async def from_custom_id(  # type: ignore[override]  # discord.py's ClientT is a free TypeVar; this adapter only ever runs DaimonBot
        cls,
        interaction: discord.Interaction[commands.Bot],
        item: discord.ui.Item[Any],
        match: re.Match[str],
        /,
    ) -> Self:
        """Rebuild the item from its token. Runs before the 3s ack budget starts.

        A single indexed primary-key read — no MA calls, no extra queries.
        Never raises: discord.py logs and discards any exception from this
        method, so a raised DB error here would leave the user with a
        permanently stuck interaction and nothing in our own logs.
        """
        token = match["token"]
        bot = cast(DaimonBot, interaction.client)
        try:
            async with bot.runtime.sessionmaker() as session:
                request_row = await peek_credential_request(session, token=token)
        except SQLAlchemyError:
            _log.exception("credential_button.lookup_failed", token_tail=token[-4:])
            request_row = None
        if request_row is None:
            return cls(token=token, label=_FALLBACK_LABEL, request_row=None)
        # `CredentialRequestRow.kind` is a bare `str` at the store boundary (ORM ->
        # Pydantic mapping); every row is inserted with a validated
        # `CredentialRequestKind` (daimon.core.credential_requests.create_*), so this
        # narrowing reflects a real invariant rather than papering over one.
        kind = cast(CredentialRequestKind, request_row.kind)
        label = build_button_label(kind, request_row.target)
        return cls(token=token, label=label, request_row=request_row)

    async def interaction_check(  # type: ignore[override]  # see from_custom_id
        self, interaction: discord.Interaction[commands.Bot], /
    ) -> bool:
        """Requester-only authorization plus fast expiry/single-use feedback.

        This is the ONLY authorization gate on the credential write — there is
        no admin check. The authoritative single-use gate is the atomic
        consume inside the modal's `on_submit`; rejecting an already-used or
        expired row here is fast user feedback, not the security boundary.
        """
        try:
            request_row = self.request_row
            if request_row is None:
                await interaction.response.send_message(_NO_LONGER_VALID, ephemeral=True)
                return False
            if str(interaction.user.id) != request_row.requester_platform_user_id:
                await interaction.response.send_message(_WRONG_REQUESTER, ephemeral=True)
                return False
            if request_row.expires_at < datetime.now(UTC):
                await interaction.response.send_message(_EXPIRED, ephemeral=True)
                return False
            if request_row.used_at is not None:
                await interaction.response.send_message(_ALREADY_USED, ephemeral=True)
                return False
            return True
        except Exception as err:  # noqa: BLE001 -- dynamic-item dispatch is an adapter boundary (see module docstring); discord.py's own dispatcher swallows anything raised here
            _log.exception(
                "credential_button.interaction_check_failed", err_type=type(err).__name__
            )
            await interaction.response.send_message(_CHECK_FAILED, ephemeral=True)
            return False

    async def callback(  # type: ignore[override]  # see from_custom_id
        self, interaction: discord.Interaction[commands.Bot]
    ) -> None:
        """Open the env or MCP modal matching this request's kind."""
        try:
            request_row = self.request_row
            if request_row is None:
                return
            bot = cast(DaimonBot, interaction.client)
            if request_row.kind == "env":
                await interaction.response.send_modal(
                    EnvCredentialModal(runtime=bot.runtime, request_row=request_row)
                )
            else:
                await interaction.response.send_modal(
                    McpCredentialModal(runtime=bot.runtime, request_row=request_row)
                )
        except Exception as err:  # noqa: BLE001 -- dynamic-item dispatch is an adapter boundary (see module docstring); discord.py's own dispatcher swallows anything raised here
            _log.exception("credential_button.callback_failed", err_type=type(err).__name__)
            await interaction.response.send_message(_CALLBACK_FAILED, ephemeral=True)
