"""Alembic: research-assistant provenance columns on rag_messages (additive)."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "f7a8b9c0d1e2"
down_revision = "e7f8a9b0c1d2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_messages",
        sa.Column("citation_validation_status", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "rag_messages",
        sa.Column("resolved_retrieval_query", sa.Text(), nullable=True),
    )
    op.add_column(
        "rag_messages",
        sa.Column("context_message_ids_json", sa.Text(), nullable=True),
    )
    op.add_column(
        "rag_messages",
        sa.Column("ai_run_id", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("rag_messages", "ai_run_id")
    op.drop_column("rag_messages", "context_message_ids_json")
    op.drop_column("rag_messages", "resolved_retrieval_query")
    op.drop_column("rag_messages", "citation_validation_status")
