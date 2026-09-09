"""Add research metadata/search indexes for high-volume workspace queries."""

from __future__ import annotations

from alembic import op

revision = "a4c8e7b2d1f0"
down_revision = "f2a6c1d4e9b0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    indexes = [
        (
            "ix_research_docs_corpus_organization",
            "research_corpus_documents",
            ["corpus_id", "organization"],
        ),
        (
            "ix_research_docs_corpus_year",
            "research_corpus_documents",
            ["corpus_id", "publication_year"],
        ),
        (
            "ix_research_docs_corpus_cultural_sphere",
            "research_corpus_documents",
            ["corpus_id", "cultural_sphere"],
        ),
        (
            "ix_research_docs_corpus_language",
            "research_corpus_documents",
            ["corpus_id", "language"],
        ),
        (
            "ix_research_tasks_annotator_status_assigned",
            "research_annotation_tasks",
            ["annotator_id", "status", "assigned_at"],
        ),
        (
            "ix_research_runs_project_type_status_created",
            "research_analysis_runs",
            ["project_id", "run_type", "status", "created_at"],
        ),
    ]
    for name, table, columns in indexes:
        op.create_index(name, table, columns, unique=False)
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        for name, column in (
            ("ix_research_docs_title_trgm", "title"),
            ("ix_research_docs_organization_trgm", "organization"),
            ("ix_research_docs_country_trgm", "country"),
        ):
            op.create_index(
                name,
                "research_corpus_documents",
                [column],
                postgresql_using="gin",
                postgresql_ops={column: "gin_trgm_ops"},
            )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        for name in (
            "ix_research_docs_country_trgm",
            "ix_research_docs_organization_trgm",
            "ix_research_docs_title_trgm",
        ):
            op.drop_index(name, table_name="research_corpus_documents")
    for name, table in (
        ("ix_research_runs_project_type_status_created", "research_analysis_runs"),
        ("ix_research_tasks_annotator_status_assigned", "research_annotation_tasks"),
        ("ix_research_docs_corpus_language", "research_corpus_documents"),
        ("ix_research_docs_corpus_cultural_sphere", "research_corpus_documents"),
        ("ix_research_docs_corpus_year", "research_corpus_documents"),
        ("ix_research_docs_corpus_organization", "research_corpus_documents"),
    ):
        op.drop_index(name, table_name=table)
