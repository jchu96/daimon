"""Stage one of the two-stage turn chokepoint (D-01): `admit()`.

A single core call performs identity resolution, the channel -> tenant ->
deployment config cascade, MA resolve + SDK retrieve, the per-tenant balance
gate, and the per-user monthly cap gate -- returning a frozen `Admission` or
raising a typed error. No boolean gate result crosses this boundary.

Gate ORDER is load-bearing and must not be reordered: identity -> cascade ->
missing-config -> resolve/retrieve -> balance -> cap. A tenant that is both
over-balance and mis-configured must see the config error (matches both
adapters' inline sequences today).

Ported verbatim from `bot.py`'s inline pre-turn sequence (the reference
implementation); nothing calls `admit()` yet -- the adapters still run their
own inline sequences until the cutover plans (06-04/06-05).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from anthropic.types.beta import BetaEnvironment, BetaManagedAgentsAgent
from daimon.core.billing import is_over_cap
from daimon.core.defaults.provisioning import reconcile_tenant_defaults
from daimon.core.ma_resolver import resolve_agent, resolve_environment
from daimon.core.scope import ResolvedConfig, ScopeContext
from daimon.core.stores.identity import get_or_create_platform_principal
from daimon.core.stores.scoped_config_read import resolve as resolve_config
from daimon.core.tenant_balance import is_over_balance
from daimon.core.turn.deps import TurnDeps
from daimon.core.turn.errors import AdmissionDenied, MissingTurnConfigError

__all__ = ["Admission", "AdmissionDenied", "MissingTurnConfigError", "admit"]


@dataclass(frozen=True)
class Admission:
    """Everything `run_prepared_turn` needs to start a turn, resolved once."""

    account_id: uuid.UUID
    agent: BetaManagedAgentsAgent
    environment: BetaEnvironment
    config: ResolvedConfig


async def admit(
    deps: TurnDeps,
    *,
    tenant_id: uuid.UUID,
    platform: str,
    external_user_id: str,
    channel_id: str,
    now: datetime,
) -> Admission:
    """Run the full pre-turn gate sequence; raise instead of returning bool."""
    # --- Identity resolution ---
    async with deps.sessionmaker() as session:
        principal = await get_or_create_platform_principal(
            session,
            tenant_id=tenant_id,
            platform=platform,
            external_id=external_user_id,
        )
        await session.commit()

    # --- Config resolution (per turn) ---
    scope = ScopeContext(
        account_id=principal.account_id,
        tenant_id=tenant_id,
        channel_id=channel_id,
    )
    async with deps.sessionmaker() as session:
        config = await resolve_config(session, context=scope, default=deps.deployment_default)

    # --- Missing config check (before any MA call) ---
    if config.agent_name is None or config.environment_name is None:
        missing: tuple[Literal["agent", "environment"], ...]
        if config.agent_name is None and config.environment_name is None:
            missing = ("agent", "environment")
        elif config.agent_name is None:
            missing = ("agent",)
        else:
            missing = ("environment",)
        raise MissingTurnConfigError(
            missing=missing,
            agent_name_tier=config.agent_name_tier,
            environment_name_tier=config.environment_name_tier,
        )

    async def _apply() -> object:
        return await reconcile_tenant_defaults(
            deps.anthropic,
            deps.defaults_root,
            tenant_id=tenant_id,
            public_url=deps.public_url,
        )

    # --- Resolve agent/environment via ma_resolver (self-heals on
    # archive/recreate); MAResolverMissError propagates unwrapped. ---
    agent_id = await resolve_agent(
        deps.anthropic,
        tenant_id=tenant_id,
        daimon_tag=config.agent_name,
        cached_id=None,
        apply_callable=_apply,
        cache=deps.resolver_cache,
    )
    env_id = await resolve_environment(
        deps.anthropic,
        tenant_id=tenant_id,
        daimon_tag=config.environment_name,
        cached_id=None,
        apply_callable=_apply,
        cache=deps.resolver_cache,
    )

    agent = await deps.anthropic.beta.agents.retrieve(agent_id)
    environment = await deps.anthropic.beta.environments.retrieve(env_id)

    # --- Admission gate: per-tenant balance -- independent of Stripe config ---
    if await is_over_balance(sessionmaker=deps.sessionmaker, tenant_id=tenant_id):
        raise AdmissionDenied(reason="balance_depleted")

    # --- Admission gate: monthly usage cap ---
    if await is_over_cap(
        billing_config=deps.billing_config,
        sessionmaker=deps.sessionmaker,
        tenant_id=tenant_id,
        user_id=external_user_id,
        now=now,
    ):
        raise AdmissionDenied(reason="cap_exceeded")

    return Admission(
        account_id=principal.account_id,
        agent=agent,
        environment=environment,
        config=config,
    )
