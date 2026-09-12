"""Stage prediction rows until their run completes."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "r1a2b3c4d5e6"
down_revision = "q5c6d7e8f9a0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_prediction_sets",
        sa.Column("status", sa.String(length=32), nullable=False, server_default="published"),
    )
    op.create_index(
        "ix_research_prediction_sets_status",
        "research_prediction_sets",
        ["status"],
        unique=False,
    )
    op.add_column(
        "research_model_predictions",
        sa.Column("prediction_set_id", sa.String(), nullable=True),
    )
    op.create_foreign_key(
        "fk_research_model_predictions_prediction_set_id",
        "research_model_predictions",
        "research_prediction_sets",
        ["prediction_set_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_research_model_predictions_prediction_set_id",
        "research_model_predictions",
        ["prediction_set_id"],
        unique=False,
    )
    op.drop_constraint(
        "uq_prediction_model_unit",
        "research_model_predictions",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_prediction_set_unit",
        "research_model_predictions",
        ["prediction_set_id", "text_unit_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_prediction_set_unit",
        "research_model_predictions",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_prediction_model_unit",
        "research_model_predictions",
        ["trained_model_id", "text_unit_id"],
    )
    op.drop_index(
        "ix_research_model_predictions_prediction_set_id",
        table_name="research_model_predictions",
    )
    op.drop_constraint(
        "fk_research_model_predictions_prediction_set_id",
        "research_model_predictions",
        type_="foreignkey",
    )
    op.drop_column("research_model_predictions", "prediction_set_id")
    op.drop_index("ix_research_prediction_sets_status", table_name="research_prediction_sets")
    op.drop_column("research_prediction_sets", "status")
