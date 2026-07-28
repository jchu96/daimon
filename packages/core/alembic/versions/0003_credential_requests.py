"""credential_requests — chat-initiated credential-button handshake.

One row per credential-request button posted in a thread. `token` is the
opaque value carried in the button's `custom_id`; `used_at` plus
`expires_at` back the single-use, TTL-bounded consume in
`daimon.core.stores.credential_requests`. No cleanup sweep this phase —
expired rows accumulate deliberately, one row per credential request.

downgrade: safe
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0003_credential_requests"
down_revision: str | None = "0002_agent_memory_store"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "credential_requests",
        sa.Column("token", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", UUID(as_uuid=True), nullable=False),
        sa.Column("account_id", UUID(as_uuid=True), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column("mcp_server_url", sa.Text(), nullable=True),
        sa.Column("requester_platform_user_id", sa.Text(), nullable=False),
        sa.Column("channel_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("token", name="pk_credential_requests"),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            ondelete="CASCADE",
            name="fk_credential_requests_tenant_id",
        ),
    )


def downgrade() -> None:
    op.drop_table("credential_requests")
