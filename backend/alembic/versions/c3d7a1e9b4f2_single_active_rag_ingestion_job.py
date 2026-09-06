"""Enforce one active indexing job per RAG document."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "c3d7a1e9b4f2"
down_revision = "a4c8e7b2d1f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing deployments may already have duplicate pending/running rows. Keep
    # the newest one active so the partial unique index can be created safely.
    op.execute(
        """
        UPDATE rag_ingestion_jobs
        SET status = 'failed', error_message = 'Superseded by a newer ingestion job'
        WHERE id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY document_id ORDER BY created_at DESC, id DESC
                ) AS row_number
                FROM rag_ingestion_jobs
                WHERE status IN ('pending', 'running')
            ) active_jobs
            WHERE active_jobs.row_number > 1
        )
        """
    )
    op.create_index(
        "uq_rag_ingestion_jobs_active_document",
        "rag_ingestion_jobs",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'running')"),
        sqlite_where=sa.text("status IN ('pending', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("uq_rag_ingestion_jobs_active_document", table_name="rag_ingestion_jobs")
