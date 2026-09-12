"""Persist explicit corpus-to-RAG bindings in assistant scope snapshots."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "p3a4b5c6d7e8"
down_revision = "p2a3b4c5d6e7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_assistant_scope_snapshots",
        sa.Column(
            "document_bindings_json",
            sa.Text(),
            nullable=False,
            server_default="[]",
        ),
    )


def downgrade() -> None:
    op.drop_column("research_assistant_scope_snapshots", "document_bindings_json")
