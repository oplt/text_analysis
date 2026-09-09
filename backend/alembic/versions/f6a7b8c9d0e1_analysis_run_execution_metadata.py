"""Add execution metadata for reliable text-research background runs."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "f6a7b8c9d0e1"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_analysis_runs",
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "research_analysis_runs",
        sa.Column("execution_key", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "research_analysis_runs",
        sa.Column("artifact_namespace", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "research_analysis_runs",
        sa.Column(
            "cancellation_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_index(
        "ix_research_analysis_runs_celery_task_id",
        "research_analysis_runs",
        ["celery_task_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_analysis_runs_execution_key",
        "research_analysis_runs",
        ["execution_key"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_research_analysis_runs_execution_key", table_name="research_analysis_runs")
    op.drop_index("ix_research_analysis_runs_celery_task_id", table_name="research_analysis_runs")
    op.drop_column("research_analysis_runs", "cancellation_requested")
    op.drop_column("research_analysis_runs", "artifact_namespace")
    op.drop_column("research_analysis_runs", "execution_key")
    op.drop_column("research_analysis_runs", "celery_task_id")
