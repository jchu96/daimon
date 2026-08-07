"""Record the in-flight turn's embed so a restart can lay it to rest.

Revision ID: 0008_active_turn_marker
Revises: 14d362cce079
Create Date: 2026-08-07

downgrade: safe

Both columns are nullable with no backfill: a row with a NULL marker is a
thread with no turn in flight, which is the correct reading for every row that
exists when this lands.

The marker cannot be inferred from `status`. That column tracks whether the
SESSION MAPPING is usable ('live'/'dead'), not whether a turn is running --
every healthy thread carries a live row forever. Sweeping on `status` would
mark every working thread failed, so the in-flight turn needs a carrier of its
own.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0008_active_turn_marker"
down_revision: str | None = "14d362cce079"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "thread_sessions",
        sa.Column("active_turn_message_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "thread_sessions",
        sa.Column("active_turn_started_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("thread_sessions", "active_turn_started_at")
    op.drop_column("thread_sessions", "active_turn_message_id")
