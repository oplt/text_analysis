"""Scope annotation evidence and adjudications to immutable campaigns."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "c0d1e2f3a4b5"
down_revision = "b9c0d1e2f3a4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_annotation_campaigns",
        sa.Column(
            "reveal_after", sa.String(length=32), nullable=False, server_default="campaign_released"
        ),
    )

    op.drop_constraint(
        "uq_annotation_task_unit_annotator",
        "research_annotation_tasks",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_annotation_task_campaign_unit_annotator",
        "research_annotation_tasks",
        ["campaign_id", "text_unit_id", "annotator_id"],
    )

    op.add_column("research_annotations", sa.Column("campaign_id", sa.String(), nullable=True))
    op.create_foreign_key(
        "fk_research_annotations_campaign_id",
        "research_annotations",
        "research_annotation_campaigns",
        ["campaign_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_research_annotations_campaign_id",
        "research_annotations",
        ["campaign_id"],
        unique=False,
    )
    op.drop_constraint(
        "uq_annotation_unit_label_annotator_version",
        "research_annotations",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_annotation_campaign_unit_label_annotator_version",
        "research_annotations",
        ["campaign_id", "text_unit_id", "label_id", "annotator_id", "codebook_version"],
    )

    op.add_column("research_adjudications", sa.Column("campaign_id", sa.String(), nullable=True))
    op.create_foreign_key(
        "fk_research_adjudications_campaign_id",
        "research_adjudications",
        "research_annotation_campaigns",
        ["campaign_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_research_adjudications_campaign_id",
        "research_adjudications",
        ["campaign_id"],
        unique=False,
    )
    op.drop_constraint(
        "uq_adjudication_unit_label",
        "research_adjudications",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_adjudication_campaign_unit_label_version",
        "research_adjudications",
        ["campaign_id", "text_unit_id", "label_id", "codebook_version"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_adjudication_campaign_unit_label_version",
        "research_adjudications",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_adjudication_unit_label",
        "research_adjudications",
        ["text_unit_id", "label_id"],
    )
    op.drop_index("ix_research_adjudications_campaign_id", table_name="research_adjudications")
    op.drop_constraint(
        "fk_research_adjudications_campaign_id",
        "research_adjudications",
        type_="foreignkey",
    )
    op.drop_column("research_adjudications", "campaign_id")

    op.drop_constraint(
        "uq_annotation_campaign_unit_label_annotator_version",
        "research_annotations",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_annotation_unit_label_annotator_version",
        "research_annotations",
        ["text_unit_id", "label_id", "annotator_id", "codebook_version"],
    )
    op.drop_index("ix_research_annotations_campaign_id", table_name="research_annotations")
    op.drop_constraint(
        "fk_research_annotations_campaign_id",
        "research_annotations",
        type_="foreignkey",
    )
    op.drop_column("research_annotations", "campaign_id")

    op.drop_constraint(
        "uq_annotation_task_campaign_unit_annotator",
        "research_annotation_tasks",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_annotation_task_unit_annotator",
        "research_annotation_tasks",
        ["text_unit_id", "annotator_id"],
    )
    op.drop_column("research_annotation_campaigns", "reveal_after")
