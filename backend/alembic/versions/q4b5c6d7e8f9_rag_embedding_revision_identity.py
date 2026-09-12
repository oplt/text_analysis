"""Persist immutable embedding identity on RAG index revisions."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "q4b5c6d7e8f9"
down_revision = "p3a4b5c6d7e8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_document_revisions",
        sa.Column("embedding_provider", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "rag_document_revisions",
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "rag_document_revisions",
        sa.Column("embedding_model_version", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "rag_document_revisions",
        sa.Column("embedding_dimensions", sa.Integer(), nullable=True),
    )
    op.add_column(
        "rag_document_revisions",
        sa.Column("embedding_preprocessing_version", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("rag_document_revisions", "embedding_preprocessing_version")
    op.drop_column("rag_document_revisions", "embedding_dimensions")
    op.drop_column("rag_document_revisions", "embedding_model_version")
    op.drop_column("rag_document_revisions", "embedding_model")
    op.drop_column("rag_document_revisions", "embedding_provider")
