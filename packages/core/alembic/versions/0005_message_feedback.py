"""message_feedback — one row per (tenant, message, voter) reaction vote.

`platform_user_id` is the real identity key, not `account_id`: a person who
reacts without ever having taken a turn has no `accounts` row, so
`account_id` resolves to NULL, and NULL cannot serve as an upsert conflict
target. `account_id` is kept as a nullable best-effort foreign key so later
reads can join, and so an account-scoped erasure reaches these rows
directly. `tenant_id` is NOT NULL and cascades on tenant teardown.

downgrade: safe
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0005_message_feedback"
down_revision: str | None = "0004_wizard_session"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "message_feedback",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("account_id", UUID(as_uuid=True), nullable=True),
        sa.Column("platform", sa.Text(), nullable=False),
        sa.Column("message_id", sa.Text(), nullable=False),
        sa.Column("channel_id", sa.Text(), nullable=False),
        sa.Column("platform_user_id", sa.Text(), nullable=False),
        sa.Column("ma_session_id", sa.Text(), nullable=True),
        sa.Column("vote", sa.Text(), nullable=False),
        sa.Column("feedback_text", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_message_feedback"),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            ondelete="CASCADE",
            name="fk_message_feedback_tenant_id",
        ),
        sa.ForeignKeyConstraint(
            ["account_id"],
            ["accounts.id"],
            ondelete="CASCADE",
            name="fk_message_feedback_account_id",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "message_id",
            "platform_user_id",
            name="uq_message_feedback_tenant_message_user",
        ),
    )
    op.create_index(
        "message_feedback_tenant_idx",
        "message_feedback",
        ["tenant_id"],
    )


def downgrade() -> None:
    op.drop_index("message_feedback_tenant_idx", table_name="message_feedback")
    op.drop_table("message_feedback")
