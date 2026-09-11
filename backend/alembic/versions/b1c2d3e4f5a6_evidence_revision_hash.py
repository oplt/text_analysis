"""Persist deterministic indexed-evidence revisions for reproducibility."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "b1c2d3e4f5a6"
down_revision = "a1b2c3d4e5f7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = (
        ("rag_retrieval_traces", "evidence_revision_hash"),
        ("rag_messages", "evidence_revision_hash"),
        ("research_assistant_threads", "evidence_revision_hash"),
        ("research_assistant_scope_snapshots", "evidence_revision_hash"),
        ("research_assistant_scope_events", "evidence_revision_hash"),
        ("research_analysis_runs", "evidence_revision_hash"),
    )
    for table, column in columns:
        op.add_column(table, sa.Column(column, sa.String(length=64), nullable=True))

    for table in (
        "rag_retrieval_traces",
        "rag_messages",
        "research_assistant_threads",
        "research_assistant_scope_snapshots",
        "research_analysis_runs",
    ):
        op.create_index(
            f"ix_{table}_evidence_revision_hash",
            table,
            ["evidence_revision_hash"],
        )


def downgrade() -> None:
    for table in (
        "research_analysis_runs",
        "research_assistant_scope_snapshots",
        "research_assistant_threads",
        "rag_messages",
        "rag_retrieval_traces",
    ):
        op.drop_index(
            f"ix_{table}_evidence_revision_hash",
            table_name=table,
        )

    for table, column in (
        ("research_analysis_runs", "evidence_revision_hash"),
        ("research_assistant_scope_events", "evidence_revision_hash"),
        ("research_assistant_scope_snapshots", "evidence_revision_hash"),
        ("research_assistant_threads", "evidence_revision_hash"),
        ("rag_messages", "evidence_revision_hash"),
        ("rag_retrieval_traces", "evidence_revision_hash"),
    ):
        op.drop_column(table, column)
