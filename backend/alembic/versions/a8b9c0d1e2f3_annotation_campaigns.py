"""Add annotation campaigns and campaign_id on annotation tasks."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "a8b9c0d1e2f3"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_annotation_campaigns",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("corpus_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("codebook_id", sa.String(), nullable=True),
        sa.Column("codebook_version", sa.String(length=64), nullable=True),
        sa.Column("unit_type", sa.String(length=32), nullable=False),
        sa.Column("sampling_strategy", sa.String(length=64), nullable=False),
        sa.Column("assignment_strategy", sa.String(length=64), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=True),
        sa.Column("overlap_count", sa.Integer(), nullable=True),
        sa.Column("overlap_percent", sa.Float(), nullable=True),
        sa.Column("blind_mode", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "ai_assistance_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column(
            "annotation_mode",
            sa.String(length=32),
            nullable=False,
            server_default="blind_reliability",
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("annotator_ids_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(["codebook_id"], ["research_codebooks.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["corpus_id"], ["research_corpora.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_research_annotation_campaigns_project_id",
        "research_annotation_campaigns",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_annotation_campaigns_corpus_id",
        "research_annotation_campaigns",
        ["corpus_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_annotation_campaigns_codebook_id",
        "research_annotation_campaigns",
        ["codebook_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_annotation_campaigns_status",
        "research_annotation_campaigns",
        ["status"],
        unique=False,
    )

    op.add_column(
        "research_annotation_tasks",
        sa.Column("campaign_id", sa.String(), nullable=True),
    )
    op.create_foreign_key(
        "fk_research_annotation_tasks_campaign_id",
        "research_annotation_tasks",
        "research_annotation_campaigns",
        ["campaign_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_research_annotation_tasks_campaign_id",
        "research_annotation_tasks",
        ["campaign_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_annotation_tasks_campaign_id", table_name="research_annotation_tasks"
    )
    op.drop_constraint(
        "fk_research_annotation_tasks_campaign_id",
        "research_annotation_tasks",
        type_="foreignkey",
    )
    op.drop_column("research_annotation_tasks", "campaign_id")

    op.drop_index(
        "ix_research_annotation_campaigns_status", table_name="research_annotation_campaigns"
    )
    op.drop_index(
        "ix_research_annotation_campaigns_codebook_id",
        table_name="research_annotation_campaigns",
    )
    op.drop_index(
        "ix_research_annotation_campaigns_corpus_id", table_name="research_annotation_campaigns"
    )
    op.drop_index(
        "ix_research_annotation_campaigns_project_id",
        table_name="research_annotation_campaigns",
    )
    op.drop_table("research_annotation_campaigns")
