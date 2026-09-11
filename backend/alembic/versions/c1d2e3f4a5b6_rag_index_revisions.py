"""Persist immutable RAG document index revisions and trace references."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "c1d2e3f4a5b6"
down_revision = "b1c2d3e4f5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rag_document_revisions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(),
            sa.ForeignKey("rag_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_content_hash", sa.String(length=64), nullable=True),
        sa.Column("parser_version", sa.String(length=64), nullable=True),
        sa.Column("chunker_version", sa.String(length=64), nullable=True),
        sa.Column("index_version", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_rag_document_revisions_document_id",
        "rag_document_revisions",
        ["document_id"],
    )
    op.create_index(
        "ix_rag_document_revisions_created_at",
        "rag_document_revisions",
        ["created_at"],
    )
    op.add_column(
        "rag_documents",
        sa.Column("current_revision_id", sa.String(), nullable=True),
    )
    op.create_foreign_key(
        "fk_rag_documents_current_revision_id",
        "rag_documents",
        "rag_document_revisions",
        ["current_revision_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_rag_documents_current_revision_id",
        "rag_documents",
        ["current_revision_id"],
    )
    op.add_column("rag_chunks", sa.Column("revision_id", sa.String(), nullable=True))
    op.create_foreign_key(
        "fk_rag_chunks_revision_id",
        "rag_chunks",
        "rag_document_revisions",
        ["revision_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_rag_chunks_revision_id", "rag_chunks", ["revision_id"])
    op.add_column(
        "rag_retrieval_traces",
        sa.Column("index_revision_ids_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("rag_retrieval_traces", "index_revision_ids_json")
    op.drop_index("ix_rag_chunks_revision_id", table_name="rag_chunks")
    op.drop_constraint("fk_rag_chunks_revision_id", "rag_chunks", type_="foreignkey")
    op.drop_column("rag_chunks", "revision_id")
    op.drop_index("ix_rag_documents_current_revision_id", table_name="rag_documents")
    op.drop_constraint(
        "fk_rag_documents_current_revision_id",
        "rag_documents",
        type_="foreignkey",
    )
    op.drop_column("rag_documents", "current_revision_id")
    op.drop_index(
        "ix_rag_document_revisions_created_at",
        table_name="rag_document_revisions",
    )
    op.drop_index(
        "ix_rag_document_revisions_document_id",
        table_name="rag_document_revisions",
    )
    op.drop_table("rag_document_revisions")
