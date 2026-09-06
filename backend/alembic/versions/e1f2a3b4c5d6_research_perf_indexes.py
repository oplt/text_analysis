"""Add composite indexes for common research query patterns."""

from __future__ import annotations

from alembic import op

revision = "e1f2a3b4c5d6"
down_revision = "d8ddea885de9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_research_text_units_doc_unit_type",
        "research_text_units",
        ["corpus_document_id", "unit_type"],
        unique=False,
    )
    op.create_index(
        "ix_research_annotation_tasks_annotator_status",
        "research_annotation_tasks",
        ["annotator_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_research_analysis_runs_project_corpus_created",
        "research_analysis_runs",
        ["project_id", "corpus_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_research_model_predictions_model_uncertainty",
        "research_model_predictions",
        ["trained_model_id", "uncertainty"],
        unique=False,
    )
    op.create_index(
        "ix_research_annotations_unit_version",
        "research_annotations",
        ["text_unit_id", "codebook_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_research_annotations_unit_version", table_name="research_annotations")
    op.drop_index(
        "ix_research_model_predictions_model_uncertainty",
        table_name="research_model_predictions",
    )
    op.drop_index(
        "ix_research_analysis_runs_project_corpus_created",
        table_name="research_analysis_runs",
    )
    op.drop_index(
        "ix_research_annotation_tasks_annotator_status",
        table_name="research_annotation_tasks",
    )
    op.drop_index("ix_research_text_units_doc_unit_type", table_name="research_text_units")
