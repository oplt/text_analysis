"""Add versioned cleaning profiles and raw-extract columns on canonical sources."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_cleaning_profiles",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("version", sa.String(length=64), nullable=False, server_default="1.0"),
        sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column(
            "created_by",
            sa.String(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_research_cleaning_profiles_project_id",
        "research_cleaning_profiles",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_cleaning_profiles_created_by",
        "research_cleaning_profiles",
        ["created_by"],
        unique=False,
    )

    op.add_column(
        "research_canonical_sources",
        sa.Column("raw_extracted_text", sa.Text(), nullable=True),
    )
    op.add_column(
        "research_canonical_sources",
        sa.Column("raw_extracted_checksum", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "research_canonical_sources",
        sa.Column(
            "cleaning_profile_id",
            sa.String(),
            sa.ForeignKey("research_cleaning_profiles.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_research_canonical_sources_raw_extracted_checksum",
        "research_canonical_sources",
        ["raw_extracted_checksum"],
        unique=False,
    )
    op.create_index(
        "ix_research_canonical_sources_cleaning_profile_id",
        "research_canonical_sources",
        ["cleaning_profile_id"],
        unique=False,
    )

    # Existing rows: raw == canonical until a cleaning profile is applied.
    op.execute(
        """
        UPDATE research_canonical_sources
        SET raw_extracted_text = canonical_text,
            raw_extracted_checksum = canonical_text_checksum
        WHERE raw_extracted_text IS NULL
        """
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_canonical_sources_cleaning_profile_id",
        table_name="research_canonical_sources",
    )
    op.drop_index(
        "ix_research_canonical_sources_raw_extracted_checksum",
        table_name="research_canonical_sources",
    )
    op.drop_column("research_canonical_sources", "cleaning_profile_id")
    op.drop_column("research_canonical_sources", "raw_extracted_checksum")
    op.drop_column("research_canonical_sources", "raw_extracted_text")
    op.drop_index(
        "ix_research_cleaning_profiles_created_by",
        table_name="research_cleaning_profiles",
    )
    op.drop_index(
        "ix_research_cleaning_profiles_project_id",
        table_name="research_cleaning_profiles",
    )
    op.drop_table("research_cleaning_profiles")
