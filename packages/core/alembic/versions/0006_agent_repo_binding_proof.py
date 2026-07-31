"""agent_repo_binding_proof — bind-time proof-of-access columns.

Records how a repo binding's access was demonstrated at bind time: the kind
of proof (a PAT that could read the repo, or the repo being publicly
readable), when it was recorded, and which account established it. Existing
rows land with all three columns NULL — no backfill runs here, so a
proof-consuming read gate treats every pre-migration binding as unproven
until it is re-bound or explicitly backfilled elsewhere. `proof_account_id`
is `ON DELETE SET NULL` rather than `CASCADE`: the binding is tenant-owned
data that must survive an account erasure, only the attribution should be
dropped.

downgrade: destructive
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0006_agent_repo_binding_proof"
down_revision: str | None = "0005_message_feedback"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "agent_repo_binding",
        sa.Column("proof_kind", sa.Text(), nullable=True),
    )
    op.add_column(
        "agent_repo_binding",
        sa.Column("proof_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "agent_repo_binding",
        sa.Column("proof_account_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_agent_repo_binding_proof_account_id",
        "agent_repo_binding",
        "accounts",
        ["proof_account_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_agent_repo_binding_proof_account_id", "agent_repo_binding", type_="foreignkey"
    )
    op.drop_column("agent_repo_binding", "proof_account_id")
    op.drop_column("agent_repo_binding", "proof_at")
    op.drop_column("agent_repo_binding", "proof_kind")
