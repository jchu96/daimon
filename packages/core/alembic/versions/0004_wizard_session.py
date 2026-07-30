"""wizard_session — durable state for the multi-step Discord wizard.

One row per wizard run, keyed by its short id (the value carried in every
tap's `custom_id`). `tenant_id` cascades on tenant teardown. `account_id` is
a nullable FK to `accounts.id` kept purely so the schema-reflecting purge
drift guard sees this table — `SET NULL` rather than `CASCADE` because
erasure is actually driven by the platform-user-scoped delete helper in
`daimon.core.stores.wizard_session`, not by this FK. No cleanup sweep in
this migration — `abandon_expired_wizard_sessions` is a store-level helper a
scheduler tick calls separately.

downgrade: safe
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0004_wizard_session"
down_revision: str | None = "0003_credential_requests"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "wizard_session",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("account_id", UUID(as_uuid=True), nullable=True),
        sa.Column("requester_platform_user_id", sa.Text(), nullable=False),
        sa.Column("channel_id", sa.Text(), nullable=False),
        sa.Column("message_id", sa.Text(), nullable=False),
        sa.Column("spec", JSONB(), nullable=False),
        sa.Column("answers", JSONB(), nullable=False),
        sa.Column("current_step", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_wizard_session"),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            ondelete="CASCADE",
            name="fk_wizard_session_tenant_id",
        ),
        sa.ForeignKeyConstraint(
            ["account_id"],
            ["accounts.id"],
            ondelete="SET NULL",
            name="fk_wizard_session_account_id",
        ),
    )
    op.create_index(
        "ix_wizard_session_tenant_requester",
        "wizard_session",
        ["tenant_id", "requester_platform_user_id"],
    )
    op.create_index(
        "ix_wizard_session_status_expires_at",
        "wizard_session",
        ["status", "expires_at"],
    )


def downgrade() -> None:
    op.drop_table("wizard_session")
