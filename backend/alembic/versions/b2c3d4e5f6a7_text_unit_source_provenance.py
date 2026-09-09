"""Add text-unit source provenance columns."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("research_text_units", sa.Column("char_start", sa.Integer(), nullable=True))
    op.add_column("research_text_units", sa.Column("char_end", sa.Integer(), nullable=True))
    op.add_column(
        "research_text_units",
        sa.Column("section_heading", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "research_text_units",
        sa.Column("source_text_hash", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_research_text_units_source_text_hash",
        "research_text_units",
        ["source_text_hash"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_research_text_units_source_text_hash", table_name="research_text_units")
    op.drop_column("research_text_units", "source_text_hash")
    op.drop_column("research_text_units", "section_heading")
    op.drop_column("research_text_units", "char_end")
    op.drop_column("research_text_units", "char_start")
