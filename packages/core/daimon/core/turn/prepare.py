"""Stage two of the two-stage turn chokepoint (D-01): `bind_session()`.

A single core call performs thread-session find-or-create, the full
`create_session` kwarg assembly, the `thread_sessions` mapping write, and
usage-recorder binding — returning a frozen `PreparedTurn` whose recorder is
a non-public field. Adapters never see or construct billing wiring.

`fernet=deps.fernet` is unconditional here: this is the fix for SPEC Req
7(a), the historical Slack gap where `create_session` was called without a
`fernet` argument.
"""

from __future__ import annotations

import functools
import uuid
from dataclasses import dataclass

from daimon.core.ma_identity import derive_agent_uuid
from daimon.core.pricing import MODEL_PRICING
from daimon.core.sessions import create_session
from daimon.core.stores.thread_sessions import create_thread_session, get_live_thread_session
from daimon.core.turn.admission import Admission
from daimon.core.turn.deps import TurnDeps
from daimon.core.turn.posture import UsageRecorder
from daimon.core.usage_recording import record_turn_usage

__all__ = ["PreparedTurn", "bind_session"]


@dataclass(frozen=True)
class PreparedTurn:
    """Everything `run_prepared_turn` (06-05) needs to drive a turn.

    `_record` is intentionally underscore-prefixed and excluded from the
    public contract adapters consume — the recorder is reachable only
    through `run_prepared_turn`.
    """

    admission: Admission
    ma_session_id: str
    mapping_id: uuid.UUID | None
    watermark: str | None
    reused: bool
    session_account_id: uuid.UUID
    _record: UsageRecorder


def _bind_recorder(
    deps: TurnDeps,
    admission: Admission,
    *,
    tenant_id: uuid.UUID,
    external_user_id: str,
    ma_session_id: str,
) -> UsageRecorder:
    """Build the usage recorder bound to a specific session id.

    Factored as a module-level helper (not inlined in `bind_session`) so
    06-05's dead-session recovery cycle can re-invoke it against the NEW
    session id after a recreate, rather than reusing a stale binding.
    """
    return functools.partial(
        record_turn_usage,
        sessionmaker=deps.sessionmaker,
        platform_user_id=external_user_id,
        managed_session_id=ma_session_id,
        model_id=admission.agent.model.id,
        tenant_id=tenant_id,
        markup=deps.markup,
        pricing=MODEL_PRICING.get(admission.agent.model.id),
    )


async def bind_session(
    deps: TurnDeps,
    admission: Admission,
    *,
    tenant_id: uuid.UUID,
    platform: str,
    external_user_id: str,
    thread_id: str,
    session_account_id: uuid.UUID,
    reuse_existing: bool,
) -> PreparedTurn:
    """Find-or-create the MA session for this turn and bind its recorder.

    When `reuse_existing` is True, a live `thread_sessions` row for
    (tenant_id, platform, thread_id, session_account_id) is reused verbatim
    (no `create_session` call). Otherwise (no live row, or
    `reuse_existing=False` for Discord's channel-mention path) a fresh
    session is created via the single shared `create_session` call site,
    always passing `fernet=deps.fernet`, and a new `thread_sessions` mapping
    row is written.
    """
    ma_session_id: str | None = None
    mapping_id: uuid.UUID | None = None
    watermark: str | None = None
    reused = False

    if reuse_existing:
        async with deps.sessionmaker() as session:
            existing = await get_live_thread_session(
                session,
                tenant_id=tenant_id,
                platform=platform,
                thread_id=thread_id,
                account_id=session_account_id,
            )
        if existing is not None:
            ma_session_id = existing.ma_session_id
            mapping_id = existing.id
            watermark = existing.watermark_message_id
            reused = True

    if not reused:
        agent_uuid = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=str(admission.agent.id))
        ma_session = await create_session(
            deps.anthropic,
            agent=admission.agent,
            environment=admission.environment,
            mcp_settings=deps.mcp,
            account_id=admission.account_id,
            tenant_id=tenant_id,
            agent_uuid=agent_uuid,
            session_factory=deps.sessionmaker,
            fernet=deps.fernet,
            github_fallback_pat=deps.github_fallback_pat,
            github_app_id=deps.github_app_id,
            github_app_private_key=deps.github_app_private_key,
        )
        ma_session_id = ma_session.id

        async with deps.sessionmaker() as session:
            row = await create_thread_session(
                session,
                tenant_id=tenant_id,
                platform=platform,
                thread_id=thread_id,
                account_id=session_account_id,
                ma_session_id=ma_session_id,
            )
            await session.commit()
        mapping_id = row.id
        watermark = None

    assert ma_session_id is not None, "ma_session_id must be resolved on every code path"

    record = _bind_recorder(
        deps,
        admission,
        tenant_id=tenant_id,
        external_user_id=external_user_id,
        ma_session_id=ma_session_id,
    )

    return PreparedTurn(
        admission=admission,
        ma_session_id=ma_session_id,
        mapping_id=mapping_id,
        watermark=watermark,
        reused=reused,
        session_account_id=session_account_id,
        _record=record,
    )
