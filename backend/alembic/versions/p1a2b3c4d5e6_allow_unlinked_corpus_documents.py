"""Allow corpus records to account for unavailable RAG sources."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "p1a2b3c4d5e6"
down_revision = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("research_corpus_documents") as batch_op:
        batch_op.alter_column(
            "rag_document_id",
            existing_type=sa.String(),
            nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("research_corpus_documents") as batch_op:
        batch_op.alter_column(
            "rag_document_id",
            existing_type=sa.String(),
            nullable=False,
        )
