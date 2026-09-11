"""Persist corpus-document availability accounting in scope snapshots."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "p1b2c3d4e5f6"
down_revision = "p1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_assistant_scope_snapshots",
        sa.Column(
            "unavailable_corpus_document_ids_json",
            sa.Text(),
            nullable=False,
            server_default="[]",
        ),
    )
    op.add_column(
        "research_assistant_scope_snapshots",
        sa.Column(
            "unavailable_reasons_json",
            sa.Text(),
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_column("research_assistant_scope_snapshots", "unavailable_reasons_json")
    op.drop_column(
        "research_assistant_scope_snapshots",
        "unavailable_corpus_document_ids_json",
    )
