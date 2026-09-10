"""Make model lifecycle vocabulary auditable.

Revision ID: c2d3e4f5a6b7
Revises: d1e2f3a4b5c6
"""

import sqlalchemy as sa

from alembic import op

revision = "c2d3e4f5a6b7"
down_revision = "d1e2f3a4b5c6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE research_trained_models SET lifecycle_status = 'production' WHERE lifecycle_status = 'approved'"
    )
    op.create_table(
        "research_model_lifecycle_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "model_id",
            sa.String(),
            sa.ForeignKey("research_trained_models.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("from_status", sa.String(length=32), nullable=True),
        sa.Column("to_status", sa.String(length=32), nullable=False),
        sa.Column(
            "actor_id", sa.String(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "run_id",
            sa.String(),
            sa.ForeignKey("research_analysis_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_research_model_lifecycle_events_model_id",
        "research_model_lifecycle_events",
        ["model_id"],
    )
    op.create_index(
        "ix_research_model_lifecycle_events_actor_id",
        "research_model_lifecycle_events",
        ["actor_id"],
    )
    op.create_index(
        "ix_research_model_lifecycle_events_run_id", "research_model_lifecycle_events", ["run_id"]
    )


def downgrade() -> None:
    op.drop_table("research_model_lifecycle_events")
    op.execute(
        "UPDATE research_trained_models SET lifecycle_status = 'approved' WHERE lifecycle_status = 'production'"
    )
