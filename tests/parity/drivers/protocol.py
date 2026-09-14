"""PlatformDriver -- the typed entry-point surface shared by DiscordDriver
and SlackDriver.

Every method builds its own platform Runtime from the given `sessionmaker`
+ `router` (an MA transport fake); callers never construct a
DiscordRuntime/SlackRuntime directly. Turn dispatch and blocked-copy are
implemented and exercised by this plan's three scenarios (a/b/c). The
resource-lifecycle methods (`delete_agent`, `fork_agent`, `purge_account`,
`uninstall`) are declared here so both drivers type-check against one
Protocol now; their scenarios (d/e/f/g) land in plan 05-09.

The posted-control block at the end (`post_credential_card`,
`click_private_input`, `submit_private_input` and the two capture readers)
is the same idea one process boundary further out: the card is posted by the
MCP server and edited by the bot, so each driver fakes both platforms' APIs
and hands the scenarios a platform-neutral `CapturedCard` per post or edit.

Signature normalization (D-03): the two adapters' `fork_agent` diverge only
in how the source agent is identified (Discord: `source_spec: AgentSpec`,
Slack: `source_name: str`) -- `AgentSpec.name` is the only field the
production code actually reads off `source_spec`, so the Protocol takes
`source_name: str` and DiscordDriver builds a minimal `AgentSpec` from it
internally. `uninstall` normalizes Discord's `guild` vs Slack's `team_id`
to a single `workspace_id: str`.
"""

from __future__ import annotations

import uuid
from typing import Literal, Protocol

from daimon.core.posted_controls import CardKind
from daimon.core.purge import AccountPurgeResult
from daimon.testing.ma import MARouter
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .cards import CapturedCard

#: Namespace for the account id a driver acts as. `credential_requests`,
#: `agent_files` and `task_continuations` all carry an account id with no FK
#: to `accounts.id` (each is erased by the platform-user-scoped helper
#: instead), so a scenario needs no seeded account row -- only a stable id
#: per (tenant, platform user), which is what a real install would have.
_PARITY_ACCOUNT_NAMESPACE = uuid.UUID("6f9d4a5e-0b21-4f3c-9a84-1c7b2e5d8f10")


def parity_account_id(tenant_id: uuid.UUID, user_id: str) -> uuid.UUID:
    """The account id both drivers act as for this (tenant, platform user)."""
    return uuid.uuid5(_PARITY_ACCOUNT_NAMESPACE, f"{tenant_id}:{user_id}")


class PlatformDriver(Protocol):
    """Drives a turn (and, eventually, agent lifecycle/purge/uninstall)
    through a platform's real entry point."""

    param_id: str

    async def dispatch_turn(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        workspace_id: str,
        channel_id: str,
        user_id: str,
        text: str,
        billing_config: object | None = None,
    ) -> list[str]:
        """Drive one turn through the real platform entry point.

        `on_message` for Discord, `_handle_app_mention` for Slack -- never a
        mid-chain `_orchestrate` call (D-02). Returns every user-facing text
        body posted back (blocked-copy or final agent reply), in dispatch
        order, so scenario tests can assert on it without touching
        platform-specific rendering (SUITE-04's job, not this one).
        """
        ...

    def expected_blocked_text(self, kind: Literal["balance", "cap"]) -> str:
        """The exact per-driver copy a blocked turn must have posted.

        Deliberately NOT shared across platforms -- the divergence principle
        (02-CONTEXT D-10) means Discord and Slack copy is allowed to differ;
        each driver returns its own platform's exact string.
        """
        ...

    async def delete_agent(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        name: str,
    ) -> None:
        """Archive the MA agent matching `name` under the given tenant."""
        ...

    async def fork_agent(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        source_name: str,
        new_name: str,
        account_id: uuid.UUID,
    ) -> None:
        """Fork `source_name` into a new agent `new_name`, copying its
        credential + repo binding."""
        ...

    async def purge_account(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        account_id: uuid.UUID,
    ) -> AccountPurgeResult:
        """Purge every principal + row under `account_id`."""
        ...

    async def uninstall(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        workspace_id: str,
    ) -> None:
        """Tear down a workspace install: soft-archive the tenant (Slack also
        deletes its workspace-scoped ephemeral rows -- an intentional
        per-platform divergence, not asserted identically)."""
        ...

    # -- posted-control lifecycle -------------------------------------------
    #
    # Three entry points, one per process boundary the real flow crosses: the
    # MCP server mints the row and posts the card, the bot process dispatches
    # the click, and the bot process runs the submission. Each drives the
    # platform's own production entry point against a platform fake, so a
    # scenario written once reads the same card on both.

    async def post_credential_card(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        workspace_id: str,
        channel_id: str,
        user_id: str,
        kind: CardKind,
        target: str,
        agent_name: str,
        mcp_server_url: str | None = None,
        branch: str | None = None,
        pending_task: str | None = None,
    ) -> str:
        """Mint and post one credential request; return its token.

        Runs the REAL MCP tool implementation
        (`tools/credential_requests._request_*_impl`) against a fake of the
        platform's own API, so the row, the card and the button's token all
        come from production code. A turn origin is created first, because
        the tools refuse to mint without one.
        """
        ...

    async def click_private_input(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        workspace_id: str,
        channel_id: str,
        user_id: str,
        token: str,
    ) -> str | None:
        """Click the card's button as `user_id`; return the private refusal.

        `None` means the form opened. Discord runs
        `CredentialRequestButton.from_custom_id` -> `interaction_check` ->
        `callback`; Slack runs `handle_credential_request_click`. Both edit
        the card in place on the expiry branch, which lands in
        `captured_cards`.
        """
        ...

    async def submit_private_input(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        workspace_id: str,
        channel_id: str,
        user_id: str,
        token: str,
        value: str = "",
        file_bytes: bytes | None = None,
    ) -> None:
        """Submit the private form: a typed value, or an uploaded `.env`.

        Discord fills the modal and awaits its `on_submit`; Slack evaluates
        the `view_submission` payload and runs the matching post-ack runner.
        Every card edit the submission makes lands in `captured_cards`.
        """
        ...

    def captured_cards(self) -> list[CapturedCard]:
        """Every post and edit of the card this driver made, in order."""
        ...

    def captured_card_states(self) -> list[str]:
        """The state of each captured card, in order."""
        ...
