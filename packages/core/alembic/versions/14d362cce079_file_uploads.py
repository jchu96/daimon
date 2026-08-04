"""file_uploads

Revision ID: 14d362cce079
Revises: 0007_tenant_last_reconcile_error
Create Date: 2026-08-04 07:51:32.849481

downgrade: safe
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "14d362cce079"
down_revision: str | None = "0007_tenant_last_reconcile_error"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "file_uploads",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.UUID(), nullable=False),
        sa.Column("upload_token", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("display_filename", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_file_uploads_created_at", "file_uploads", ["created_at"], unique=False)
    op.create_index("ix_file_uploads_upload_token", "file_uploads", ["upload_token"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_file_uploads_upload_token", table_name="file_uploads")
    op.drop_index("ix_file_uploads_created_at", table_name="file_uploads")
    op.drop_table("file_uploads")
