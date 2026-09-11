"""Persist immutable research-assistant thread scopes and their audit trail."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "a1b2c3d4e5f7"
down_revision = "f7a8b9c0d1e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_assistant_threads",
        sa.Column("scope_mode", sa.String(length=16), nullable=False, server_default="fixed"),
    )
    op.add_column(
        "research_assistant_threads",
        sa.Column("scope_snapshot_json", sa.Text(), nullable=True),
    )
    op.execute(
        """
        UPDATE research_assistant_threads
        SET scope_snapshot_json = (
            SELECT scope_snapshot_json
            FROM rag_conversations
            WHERE rag_conversations.id = research_assistant_threads.rag_conversation_id
        )
        WHERE scope_snapshot_json IS NULL
        """
    )
    op.add_column(
        "research_assistant_scope_snapshots",
        sa.Column("scope_mode", sa.String(length=16), nullable=False, server_default="fixed"),
    )

    op.create_table(
        "research_assistant_scope_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "thread_id",
            sa.String(),
            sa.ForeignKey("research_assistant_threads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "actor_id",
            sa.String(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("scope_mode", sa.String(length=16), nullable=False),
        sa.Column("corpus_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("previous_scope_hash", sa.String(length=64), nullable=True),
        sa.Column("new_scope_hash", sa.String(length=64), nullable=False),
        sa.Column("rag_document_ids_json", sa.Text(), nullable=False),
        sa.Column("corpus_document_ids_json", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_research_assistant_scope_events_thread_id",
        "research_assistant_scope_events",
        ["thread_id"],
    )
    op.create_index(
        "ix_research_assistant_scope_events_actor_id",
        "research_assistant_scope_events",
        ["actor_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_assistant_scope_events_actor_id",
        table_name="research_assistant_scope_events",
    )
    op.drop_index(
        "ix_research_assistant_scope_events_thread_id",
        table_name="research_assistant_scope_events",
    )
    op.drop_table("research_assistant_scope_events")
    op.drop_column("research_assistant_scope_snapshots", "scope_mode")
    op.drop_column("research_assistant_threads", "scope_snapshot_json")
    op.drop_column("research_assistant_threads", "scope_mode")
