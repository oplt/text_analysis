"""Persist durable research artifact metadata and run lineage."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "e6f7a8b9c0d1"
down_revision = "d5e6f7a8b9c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_artifacts",
        sa.Column("id", sa.String(length=255), primary_key=True),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("checksum", sa.String(length=128), nullable=False),
        sa.Column("storage_backend", sa.String(length=32), nullable=False),
        sa.Column("storage_key", sa.String(length=1024)),
        sa.Column("object_key", sa.String(length=1024)),
        sa.Column("content_type", sa.String(length=255), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("format", sa.String(length=128)),
        sa.Column("dimensions_json", sa.Text()),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("kind", "checksum", name="uq_research_artifact_kind_checksum"),
    )
    op.create_index("ix_research_artifacts_kind", "research_artifacts", ["kind"])
    op.create_index("ix_research_artifacts_checksum", "research_artifacts", ["checksum"])
    op.create_table(
        "research_analysis_run_artifacts",
        sa.Column("run_id", sa.String(), sa.ForeignKey("research_analysis_runs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("artifact_id", sa.String(length=255), sa.ForeignKey("research_artifacts.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("role", sa.String(length=128), primary_key=True),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("research_analysis_run_artifacts")
    op.drop_index("ix_research_artifacts_checksum", table_name="research_artifacts")
    op.drop_index("ix_research_artifacts_kind", table_name="research_artifacts")
    op.drop_table("research_artifacts")
