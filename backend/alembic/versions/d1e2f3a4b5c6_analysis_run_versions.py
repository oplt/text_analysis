"""Add durable ordering for analysis-run SSE events.

Revision ID: d1e2f3a4b5c6
Revises: c0d1e2f3a4b5
"""

from alembic import op
import sqlalchemy as sa

revision = "d1e2f3a4b5c6"
down_revision = "c0d1e2f3a4b5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_analysis_runs",
        sa.Column("run_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.alter_column("research_analysis_runs", "run_version", server_default=None)


def downgrade() -> None:
    op.drop_column("research_analysis_runs", "run_version")
