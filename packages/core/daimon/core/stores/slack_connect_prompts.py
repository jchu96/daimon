"""Async store for slack_connect_prompts — once-ever first-mention nudge marker."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from daimon.core._models import SlackConnectPrompt
from sqlalchemy import CursorResult, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession


async def mark_connect_prompted(
    session: AsyncSession,
    *,
    team_id: str,
    slack_user_id: str,
    now: datetime,
) -> None:
    """Record that the nudge was shown. ON CONFLICT DO NOTHING (nudge races)."""
    stmt = (
        pg_insert(SlackConnectPrompt)
        .values(team_id=team_id, slack_user_id=slack_user_id, prompted_at=now)
        .on_conflict_do_nothing(
            index_elements=[SlackConnectPrompt.team_id, SlackConnectPrompt.slack_user_id]
        )
    )
    await session.execute(stmt)
    await session.flush()


async def was_connect_prompted(
    session: AsyncSession,
    *,
    team_id: str,
    slack_user_id: str,
) -> bool:
    return await session.get(SlackConnectPrompt, (team_id, slack_user_id)) is not None


async def delete_connect_prompts_for_team(session: AsyncSession, *, team_id: str) -> int:
    """Delete every slack_connect_prompts row for a workspace. Idempotent.

    Returns rowcount; never raises on 0. Used by the Slack uninstall teardown.
    """
    result = await session.execute(
        delete(SlackConnectPrompt).where(SlackConnectPrompt.team_id == team_id)
    )
    rowcount = cast(CursorResult[Any], result).rowcount
    await session.flush()
    return rowcount
