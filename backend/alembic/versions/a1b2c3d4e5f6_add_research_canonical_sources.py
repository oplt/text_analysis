"""Add immutable canonical research source table."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a1b2c3d4e5f6"
down_revision = "b7e4c2d8f1a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_canonical_sources",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "corpus_document_id",
            sa.String(),
            sa.ForeignKey("research_corpus_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("canonical_text", sa.Text(), nullable=False),
        sa.Column("canonical_text_checksum", sa.String(length=64), nullable=False),
        sa.Column("original_file_checksum", sa.String(length=64), nullable=True),
        sa.Column("parser_name", sa.String(length=128), nullable=False),
        sa.Column("parser_version", sa.String(length=64), nullable=False),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "source_rag_document_id",
            sa.String(),
            sa.ForeignKey("rag_documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source_storage_path", sa.String(length=1024), nullable=True),
        sa.Column("source_filename", sa.String(length=512), nullable=True),
        sa.Column("language", sa.String(length=64), nullable=True),
        sa.Column("page_provenance_json", sa.Text(), nullable=True),
        sa.Column("transformation_metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_research_canonical_sources_corpus_document_id",
        "research_canonical_sources",
        ["corpus_document_id"],
        unique=True,
    )
    op.create_index(
        "ix_research_canonical_sources_canonical_text_checksum",
        "research_canonical_sources",
        ["canonical_text_checksum"],
        unique=False,
    )
    op.create_index(
        "ix_research_canonical_sources_source_rag_document_id",
        "research_canonical_sources",
        ["source_rag_document_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_canonical_sources_source_rag_document_id",
        table_name="research_canonical_sources",
    )
    op.drop_index(
        "ix_research_canonical_sources_canonical_text_checksum",
        table_name="research_canonical_sources",
    )
    op.drop_index(
        "ix_research_canonical_sources_corpus_document_id",
        table_name="research_canonical_sources",
    )
    op.drop_table("research_canonical_sources")
