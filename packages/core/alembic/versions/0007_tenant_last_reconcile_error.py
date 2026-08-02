"""tenant_last_reconcile_error — persist why a tenant's defaults reconcile last failed.

Container logs are destroyed on every deploy, so a boot-time seeded-defaults
reconcile failure is exactly the kind of failure that vanishes with them. This
column records the reason on the tenant row itself, so it survives the
container that logged it and is readable by plain SQL afterward.

Existing rows land NULL, meaning "no recorded failure". A successful reconcile
also writes NULL back, so a stale reason from an earlier failure can never be
mistaken for a current one — the column reflects only the outcome of the most
recent reconcile attempt.

downgrade: safe
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0007_tenant_last_reconcile_error"
down_revision: str | None = "0006_agent_repo_binding_proof"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("last_reconcile_error", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tenants", "last_reconcile_error")
