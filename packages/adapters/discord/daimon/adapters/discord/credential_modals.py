"""EnvCredentialModal / McpCredentialModal — the two credential-button modals.

Two separate modals, not one type-dispatching modal: an env secret and an MCP
auth token are different resources with different write paths
(`put_agent_file` vs `add_external_mcp_credential`), mirroring the two modals
that already exist for the same writes (`agent_setup/credentials.py`'s
`PasteSecretModal`, `agent_setup/modals_mcp.py`'s `AddMcpModal`). Each modal
here collects exactly ONE field — the secret value itself — because every
routing field (agent, key/server name) is already fixed by the consumed
`credential_requests` row; the user never retypes it.

Secret hygiene, matching the structural guarantees `PasteSecretModal` already
documents:
- the value never reaches a log record (env logs the key name only; MCP logs
  a masked tail only),
- the value never reaches a `custom_id` (the button's custom_id carries only
  the opaque request token, minted before either modal exists),
- the value never reaches a container/embed (a Modal TextInput has no
  render surface other than the ephemeral confirmation, which never echoes
  the value back),
- there is no URL-fetch path and no attachment path.

The atomic single-use consume runs BEFORE either write, so a request can
only ever produce one write no matter how many times its modal is
(re)submitted — the loser of a race, or any resubmission, gets `None` back
and writes nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from daimon.adapters.discord.agent_setup.credentials import (
    _MAX_SECRET_VALUE_BYTES,  # pyright: ignore[reportPrivateUsage]  # reusing PasteSecretModal's byte cap rather than inventing a second number
)
from daimon.adapters.discord.agent_setup.write import mask_tail
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.mcp_vault import add_external_mcp_credential
from daimon.core.stores import credential_requests
from daimon.core.stores.agent_files import put_agent_file
from daimon.core.stores.domain import CredentialRequestRow

import discord

_log = structlog.get_logger()

_NO_LONGER_VALID = "This request is no longer valid — ask again."


class EnvCredentialModal(discord.ui.Modal, title="Add secret"):
    """Add secrets modal: one value, atomic consume, existing agent_files write."""

    def __init__(self, *, runtime: DiscordRuntime, request_row: CredentialRequestRow) -> None:
        super().__init__()
        self._runtime = runtime
        self._row = request_row
        self.value_input: discord.ui.TextInput[EnvCredentialModal] = discord.ui.TextInput(
            label="Secret value",
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=4000,
            placeholder="usable by everyone who talks to this agent",
        )
        self.add_item(self.value_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        raw_value = str(self.value_input.value or "")

        if not raw_value.strip():
            await interaction.followup.send(
                "Secret value cannot be empty — try again.", ephemeral=True
            )
            return
        if len(raw_value.encode()) > _MAX_SECRET_VALUE_BYTES:
            await interaction.followup.send(
                f"Secret value is too large. Max {_MAX_SECRET_VALUE_BYTES} bytes.",
                ephemeral=True,
            )
            return

        now = datetime.now(UTC)
        try:
            async with self._runtime.sessionmaker() as session, session.begin():
                consumed_row = await credential_requests.consume_credential_request(
                    session, token=self._row.token, now=now
                )
                if consumed_row is None:
                    await interaction.followup.send(_NO_LONGER_VALID, ephemeral=True)
                    return
                await put_agent_file(
                    session,
                    tenant_id=consumed_row.tenant_id,
                    agent_id=consumed_row.agent_id,
                    key=consumed_row.target,
                    content=raw_value,
                )
        except Exception:
            _log.exception("credential_modal.env_write_failed", key=self._row.target)
            await interaction.followup.send(
                "Something went wrong — please try again.", ephemeral=True
            )
            return

        # Log the key NAME only — never the value.
        _log.info("credential_modal.env.submit", key=consumed_row.target)
        await interaction.followup.send(
            f"Added `{consumed_row.target}`. Takes effect on the next session — "
            "anyone who talks to this agent can use it.",
            ephemeral=True,
        )


class McpCredentialModal(discord.ui.Modal, title="Add MCP credential"):
    """Add MCP credential modal: one token, atomic consume, existing vault write."""

    def __init__(self, *, runtime: DiscordRuntime, request_row: CredentialRequestRow) -> None:
        super().__init__()
        self._runtime = runtime
        self._row = request_row
        self.token_input: discord.ui.TextInput[McpCredentialModal] = discord.ui.TextInput(
            label="Auth token",
            required=True,
            max_length=255,
            placeholder="usable by everyone who talks to this agent",
        )
        self.add_item(self.token_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        token_value = str(self.token_input.value or "")

        if not token_value.strip():
            await interaction.followup.send(
                "Auth token cannot be empty — try again.", ephemeral=True
            )
            return

        public_url_setting = self._runtime.settings.mcp.public_url
        jwt_secret_setting = self._runtime.settings.mcp.jwt_secret
        if public_url_setting is None or jwt_secret_setting is None:
            await interaction.followup.send(
                "daimon-mcp is not configured (public_url / jwt_secret missing) — "
                "credential not written.",
                ephemeral=True,
            )
            return

        now = datetime.now(UTC)
        async with self._runtime.sessionmaker() as session, session.begin():
            consumed_row = await credential_requests.consume_credential_request(
                session, token=self._row.token, now=now
            )
        if consumed_row is None:
            await interaction.followup.send(_NO_LONGER_VALID, ephemeral=True)
            return

        mcp_server_url = consumed_row.mcp_server_url
        if mcp_server_url is None:
            _log.error("credential_modal.mcp_missing_server_url", token_tail=self._row.token[-4:])
            await interaction.followup.send(
                "This request is missing its server URL — please ask again.", ephemeral=True
            )
            return

        _log.info(
            "credential_modal.mcp.submit",
            mcp_server_url=mcp_server_url,
            token_masked=mask_tail(token_value),
        )
        try:
            await add_external_mcp_credential(
                self._runtime.anthropic,
                account_id=consumed_row.account_id,
                agent_id=consumed_row.agent_id,
                jwt_secret=jwt_secret_setting.get_secret_value().encode(),
                public_url=str(public_url_setting),
                mcp_server_url=mcp_server_url,
                token=token_value,
                now=now,
                session_factory=self._runtime.sessionmaker,
            )
        except Exception as err:
            _log.exception(
                "credential_modal.mcp_write_failed",
                mcp_server_url=mcp_server_url,
                err_type=type(err).__name__,
            )
            # Surface only the exception class name — never the stringified
            # exception, which for SDK/network errors can carry the request
            # envelope (a token-leak surface). Full traceback goes to structlog.
            await interaction.followup.send(
                "Credential request consumed, but storing the auth token failed "
                f"(`{type(err).__name__}`). Ask for a new request to retry.",
                ephemeral=True,
            )
            return

        await interaction.followup.send(
            f"MCP credential added for `{mcp_server_url}`. Anyone who talks to "
            "this agent can use it.",
            ephemeral=True,
        )
