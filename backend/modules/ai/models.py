from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class AiPromptTemplate(Base):
    __tablename__ = "ai_prompt_templates"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    active_version_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "ai_prompt_versions.id",
            ondelete="SET NULL",
            use_alter=True,
            name="fk_template_active_version",
        ),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class AiPromptVersion(Base):
    __tablename__ = "ai_prompt_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    prompt_template_id: Mapped[str] = mapped_column(
        ForeignKey("ai_prompt_templates.id", ondelete="CASCADE"),
        index=True,
    )
    version_number: Mapped[int] = mapped_column(Integer)
    provider_key: Mapped[str] = mapped_column(String(64), default="local")
    model_name: Mapped[str] = mapped_column(String(128))
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    user_prompt_template: Mapped[str] = mapped_column(Text)
    variable_definitions_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    response_format: Mapped[str] = mapped_column(String(32), default="text")
    temperature: Mapped[float] = mapped_column(Float, default=0.2)
    rollout_percentage: Mapped[int] = mapped_column(Integer, default=100)
    is_published: Mapped[bool] = mapped_column(Boolean, default=False)
    input_cost_per_million: Mapped[int] = mapped_column(Integer, default=0)
    output_cost_per_million: Mapped[int] = mapped_column(Integer, default=0)
    created_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class AiDocument(Base):
    """Legacy ORM mapping for ``ai_documents`` (unused when ``RAG_ENABLED=true``).

    Document ingestion and retrieval use ``rag_documents`` / ``rag_chunks`` instead.
    Kept for Alembic metadata until a drop-table migration removes the legacy tables.
    """

    __tablename__ = "ai_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_type: Mapped[str] = mapped_column(String(128), default="text/plain")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    ingestion_status: Mapped[str] = mapped_column(String(32), default="completed")
    source_text: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class AiDocumentChunk(Base):
    """Legacy ORM mapping for ``ai_document_chunks`` (superseded by ``rag_chunks``)."""

    __tablename__ = "ai_document_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(
        ForeignKey("ai_documents.id", ondelete="CASCADE"),
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_json: Mapped[list[float]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class AiRun(Base):
    __tablename__ = "ai_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    prompt_template_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_prompt_templates.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    prompt_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_prompt_versions.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    evaluation_dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_evaluation_datasets.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    evaluation_case_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_evaluation_cases.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    provider_key: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="completed")
    response_format: Mapped[str] = mapped_column(String(32), default="text")
    variables_json: Mapped[dict] = mapped_column(JSON, default=dict)
    retrieval_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    retrieval_degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    memory_degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    degradation_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    injection_chunks_filtered: Mapped[int] = mapped_column(Integer, default=0)
    input_messages_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_micros: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(String(32), default="not_requested")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AiReviewItem(Base):
    __tablename__ = "ai_review_items"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(
        ForeignKey("ai_runs.id", ondelete="CASCADE"),
        index=True,
    )
    requested_by_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    assigned_to_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    reviewed_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(32), default="pending")
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class AiFeedback(Base):
    __tablename__ = "ai_feedback"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(
        ForeignKey("ai_runs.id", ondelete="CASCADE"),
        index=True,
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    rating: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class AiEvaluationDataset(Base):
    __tablename__ = "ai_evaluation_datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class AiEvaluationCase(Base):
    __tablename__ = "ai_evaluation_cases"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("ai_evaluation_datasets.id", ondelete="CASCADE"),
        index=True,
    )
    input_variables_json: Mapped[dict] = mapped_column(JSON, default=dict)
    retrieval_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    expected_chunk_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    expected_output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class AiEvaluationRun(Base):
    __tablename__ = "ai_evaluation_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("ai_evaluation_datasets.id", ondelete="CASCADE"),
        index=True,
    )
    prompt_version_id: Mapped[str] = mapped_column(
        ForeignKey("ai_prompt_versions.id", ondelete="CASCADE"),
        index=True,
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    passed_cases: Mapped[int] = mapped_column(Integer, default=0)
    average_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AiEvaluationRunItem(Base):
    __tablename__ = "ai_evaluation_run_items"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    evaluation_run_id: Mapped[str] = mapped_column(
        ForeignKey("ai_evaluation_runs.id", ondelete="CASCADE"),
        index=True,
    )
    evaluation_case_id: Mapped[str] = mapped_column(
        ForeignKey("ai_evaluation_cases.id", ondelete="CASCADE"),
        index=True,
    )
    ai_run_id: Mapped[str] = mapped_column(
        ForeignKey("ai_runs.id", ondelete="CASCADE"),
        index=True,
    )
    score: Mapped[float] = mapped_column(Float, default=0.0)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
