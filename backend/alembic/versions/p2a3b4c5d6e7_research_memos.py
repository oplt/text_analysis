"""Add durable research memos with evidence snapshots."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "p2a3b4c5d6e7"
down_revision = "p1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_memos",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("corpus_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("originating_assistant_message_id", sa.String(), nullable=True),
        sa.Column("originating_synthesis_run_id", sa.String(), nullable=True),
        sa.Column("evidence_revision_hash", sa.String(length=64), nullable=True),
        sa.Column("citations_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("claims_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("provenance_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["corpus_id"], ["research_corpora.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["originating_assistant_message_id"], ["rag_messages.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["originating_synthesis_run_id"], ["research_analysis_runs.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    for name, column in (
        ("ix_research_memos_project_id", "project_id"),
        ("ix_research_memos_corpus_id", "corpus_id"),
        ("ix_research_memos_user_id", "user_id"),
        ("ix_research_memos_source_type", "source_type"),
        ("ix_research_memos_status", "status"),
        ("ix_research_memos_originating_assistant_message_id", "originating_assistant_message_id"),
        ("ix_research_memos_originating_synthesis_run_id", "originating_synthesis_run_id"),
        ("ix_research_memos_evidence_revision_hash", "evidence_revision_hash"),
    ):
        op.create_index(name, "research_memos", [column])


def downgrade() -> None:
    for name in (
        "ix_research_memos_evidence_revision_hash",
        "ix_research_memos_originating_synthesis_run_id",
        "ix_research_memos_originating_assistant_message_id",
        "ix_research_memos_status",
        "ix_research_memos_source_type",
        "ix_research_memos_user_id",
        "ix_research_memos_corpus_id",
        "ix_research_memos_project_id",
    ):
        op.drop_index(name, table_name="research_memos")
    op.drop_table("research_memos")
