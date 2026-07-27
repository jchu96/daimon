"""PlatformDriver -- the typed entry-point surface shared by DiscordDriver
and SlackDriver.

Every method builds its own platform Runtime from the given `sessionmaker`
+ `router` (an MA transport fake); callers never construct a
DiscordRuntime/SlackRuntime directly. Turn dispatch and blocked-copy are
implemented and exercised by this plan's three scenarios (a/b/c). The
resource-lifecycle methods (`delete_agent`, `fork_agent`, `purge_account`,
`uninstall`) are declared here so both drivers type-check against one
Protocol now; their scenarios (d/e/f/g) land in plan 05-09.

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

from daimon.core.purge import AccountPurgeResult
from daimon.testing.ma import MARouter
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


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
