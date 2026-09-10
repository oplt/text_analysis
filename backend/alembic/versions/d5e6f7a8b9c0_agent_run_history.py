"""Persist Agent run identity for scoped history and detail retrieval.

Revision ID: d5e6f7a8b9c0
Revises: d3e4f5a6b7c8
"""

import sqlalchemy as sa

from alembic import op

revision = "d5e6f7a8b9c0"
down_revision = "d3e4f5a6b7c8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ai_runs", sa.Column("agent_id", sa.String(length=128), nullable=True))
    op.add_column("ai_runs", sa.Column("agent_run_id", sa.String(length=64), nullable=True))
    op.create_index("ix_ai_runs_agent_id", "ai_runs", ["agent_id"])


def downgrade() -> None:
    op.drop_index("ix_ai_runs_agent_id", table_name="ai_runs")
    op.drop_column("ai_runs", "agent_run_id")
    op.drop_column("ai_runs", "agent_id")
