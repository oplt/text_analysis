"""add AI run degradation metadata

Revision ID: 7f3c1a9d2e44
Revises: 28a5b9ada6fc
Create Date: 2026-07-06
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "7f3c1a9d2e44"
down_revision: str | Sequence[str] | None = "28a5b9ada6fc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ai_runs",
        sa.Column("retrieval_degraded", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "ai_runs",
        sa.Column("memory_degraded", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("ai_runs", sa.Column("degradation_reason", sa.String(length=128)))
    op.add_column(
        "ai_runs",
        sa.Column("injection_chunks_filtered", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("ai_evaluation_cases", sa.Column("retrieval_query", sa.Text()))
    op.add_column(
        "ai_evaluation_cases",
        sa.Column(
            "document_ids_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
    )
    op.add_column(
        "ai_evaluation_cases",
        sa.Column(
            "expected_chunk_ids_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
    )


def downgrade() -> None:
    op.drop_column("ai_evaluation_cases", "expected_chunk_ids_json")
    op.drop_column("ai_evaluation_cases", "document_ids_json")
    op.drop_column("ai_evaluation_cases", "retrieval_query")
    op.drop_column("ai_runs", "injection_chunks_filtered")
    op.drop_column("ai_runs", "degradation_reason")
    op.drop_column("ai_runs", "memory_degraded")
    op.drop_column("ai_runs", "retrieval_degraded")
