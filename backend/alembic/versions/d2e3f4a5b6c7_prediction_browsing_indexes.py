"""Indexes for paged prediction-set browsing.

Revision ID: d2e3f4a5b6c7
Revises: c2d3e4f5a6b7
"""

from alembic import op

revision = "d2e3f4a5b6c7"
down_revision = "c2d3e4f5a6b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_model_predictions_model_uncertainty_created",
        "research_model_predictions",
        ["trained_model_id", "uncertainty", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_model_predictions_model_uncertainty_created", table_name="research_model_predictions"
    )
