"""Add lifecycle status columns to research_trained_models."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_trained_models",
        sa.Column(
            "lifecycle_status",
            sa.String(length=32),
            nullable=False,
            server_default="candidate",
        ),
    )
    op.add_column(
        "research_trained_models",
        sa.Column("lifecycle_notes", sa.Text(), nullable=True),
    )
    op.add_column(
        "research_trained_models",
        sa.Column("lifecycle_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_research_trained_models_lifecycle_status",
        "research_trained_models",
        ["lifecycle_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_trained_models_lifecycle_status",
        table_name="research_trained_models",
    )
    op.drop_column("research_trained_models", "lifecycle_updated_at")
    op.drop_column("research_trained_models", "lifecycle_notes")
    op.drop_column("research_trained_models", "lifecycle_status")
