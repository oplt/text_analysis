"""Composite indexes for large-corpus research query patterns (Phase 8).

Confirmed against hot paths:
- reliability: Annotation filtered by codebook_version (+ label_id)
- campaign reliability EXISTS: AnnotationTask by campaign_id + unit + annotator
- run listing / SSE reconcile: AnalysisRun by project_id + status + created_at
"""

from __future__ import annotations

from alembic import op

revision = "b9c0d1e2f3a4"
down_revision = "a8b9c0d1e2f3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_research_annotations_codebook_version_label",
        "research_annotations",
        ["codebook_version", "label_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_annotation_tasks_campaign_unit_annotator",
        "research_annotation_tasks",
        ["campaign_id", "text_unit_id", "annotator_id"],
        unique=False,
    )
    op.create_index(
        "ix_research_analysis_runs_project_status_created",
        "research_analysis_runs",
        ["project_id", "status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_analysis_runs_project_status_created",
        table_name="research_analysis_runs",
    )
    op.drop_index(
        "ix_research_annotation_tasks_campaign_unit_annotator",
        table_name="research_annotation_tasks",
    )
    op.drop_index(
        "ix_research_annotations_codebook_version_label",
        table_name="research_annotations",
    )
