"""Add sanitized request diagnostics to RAG retrieval traces."""

import sqlalchemy as sa

from alembic import op

revision = "q5c6d7e8f9a0"
down_revision = "q4b5c6d7e8f9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("rag_retrieval_traces", sa.Column("request_id", sa.String(64), nullable=True))
    op.add_column("rag_retrieval_traces", sa.Column("stage_timings_json", sa.Text(), nullable=True))
    op.add_column("rag_retrieval_traces", sa.Column("branch_status_json", sa.Text(), nullable=True))
    op.create_index("ix_rag_retrieval_traces_request_id", "rag_retrieval_traces", ["request_id"])


def downgrade() -> None:
    op.drop_index("ix_rag_retrieval_traces_request_id", table_name="rag_retrieval_traces")
    op.drop_column("rag_retrieval_traces", "branch_status_json")
    op.drop_column("rag_retrieval_traces", "stage_timings_json")
    op.drop_column("rag_retrieval_traces", "request_id")
