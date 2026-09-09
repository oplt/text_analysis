"""Add research_prediction_sets table."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_prediction_sets",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("corpus_id", sa.String(), nullable=False),
        sa.Column("trained_model_id", sa.String(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("dataset_snapshot_id", sa.String(), nullable=True),
        sa.Column("analysis_run_id", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["analysis_run_id"], ["research_analysis_runs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["corpus_id"], ["research_corpora.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["dataset_snapshot_id"], ["research_training_dataset_snapshots.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["trained_model_id"], ["research_trained_models.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_research_prediction_sets_project_id",
        "research_prediction_sets",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_prediction_sets_corpus_id",
        "research_prediction_sets",
        ["corpus_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_prediction_sets_trained_model_id",
        "research_prediction_sets",
        ["trained_model_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_prediction_sets_analysis_run_id",
        "research_prediction_sets",
        ["analysis_run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_prediction_sets_analysis_run_id", table_name="research_prediction_sets"
    )
    op.drop_index(
        "ix_research_prediction_sets_trained_model_id", table_name="research_prediction_sets"
    )
    op.drop_index("ix_research_prediction_sets_corpus_id", table_name="research_prediction_sets")
    op.drop_index("ix_research_prediction_sets_project_id", table_name="research_prediction_sets")
    op.drop_table("research_prediction_sets")
