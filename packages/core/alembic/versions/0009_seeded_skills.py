"""Content fingerprint per seeded skill, so an edit can reach a live install.

Revision ID: 0009_seeded_skills
Revises: 0008_active_turn_marker
Create Date: 2026-08-07

downgrade: safe

No backfill. An absent row reads as "MA holds unknown content", which drives
exactly one version upload per seeded skill on the first reconcile after this
lands — the intended behaviour, since every existing install is carrying
whatever content it was seeded with and nothing knows what that was.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0009_seeded_skills"
down_revision: str | None = "0008_active_turn_marker"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "seeded_skills",
        sa.Column("tenant_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("anthropic_id", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("tenant_id", "name", name="pk_seeded_skills"),
    )


def downgrade() -> None:
    op.drop_table("seeded_skills")
