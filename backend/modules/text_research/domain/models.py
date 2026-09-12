from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str)


def loads(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


class ResearchCorpus(Base):
    __tablename__ = "research_corpora"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class CorpusDocument(Base):
    __tablename__ = "research_corpus_documents"
    __table_args__ = (
        UniqueConstraint("corpus_id", "rag_document_id", name="uq_corpus_rag_document"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("research_corpora.id", ondelete="CASCADE"), index=True
    )
    rag_document_id: Mapped[str | None] = mapped_column(
        ForeignKey("rag_documents.id", ondelete="CASCADE"), index=True, nullable=True
    )
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    organization: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    organization_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    publication_type: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    country: Mapped[str | None] = mapped_column(String(128), nullable=True)
    region: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    cultural_sphere: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    language: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    education_level: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    research_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # Fixed columns that are eligible as user-selected grouping / temporal
    # fields (see §38/§39: robustness group_field, temporal_field). Kept as a
    # class attribute (not infrastructure) so any layer can validate a
    # requested field name without importing the ORM's SQLAlchemy internals.
    KNOWN_FACET_FIELDS: tuple[str, ...] = (
        "organization",
        "organization_type",
        "publication_year",
        "publication_type",
        "country",
        "region",
        "cultural_sphere",
        "language",
        "education_level",
    )

    def get_field_value(self, field_name: str) -> Any:
        """Resolve any user-selected facet/group field, generic across projects.

        Checks known fixed columns first (organization, region, country,
        publication_year, ...); anything else is looked up in the free-form
        ``metadata_json`` blob so custom project-specific fields work without
        schema changes. Returns ``None`` when the field is unknown/absent —
        callers must not assume it exists.
        """
        if field_name in self.KNOWN_FACET_FIELDS:
            return getattr(self, field_name, None)
        metadata = loads(self.metadata_json, {}) or {}
        return metadata.get(field_name)


class TextUnit(Base):
    __tablename__ = "research_text_units"
    __table_args__ = (
        UniqueConstraint(
            "corpus_document_id",
            "unit_type",
            "position",
            name="uq_text_unit_position",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    corpus_document_id: Mapped[str] = mapped_column(
        ForeignKey("research_corpus_documents.id", ondelete="CASCADE"), index=True
    )
    unit_type: Mapped[str] = mapped_column(String(32), index=True)
    position: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paragraph_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sentence_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_heading: Mapped[str | None] = mapped_column(String(512), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(String(64), index=True)
    source_text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class CanonicalResearchSource(Base):
    """Immutable full-document text used for research analysis.

    Kept separate from RAG retrieval chunks so overlapping chunk joins can never
    corrupt frequencies, DFM, annotation, or classifiers.
    """

    __tablename__ = "research_canonical_sources"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    corpus_document_id: Mapped[str] = mapped_column(
        ForeignKey("research_corpus_documents.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    canonical_text: Mapped[str] = mapped_column(Text)
    canonical_text_checksum: Mapped[str] = mapped_column(String(64), index=True)
    original_file_checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parser_name: Mapped[str] = mapped_column(String(128))
    parser_version: Mapped[str] = mapped_column(String(64))
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    source_rag_document_id: Mapped[str | None] = mapped_column(
        ForeignKey("rag_documents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    source_storage_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    language: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_provenance_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Raw extract is immutable; cleaning may rewrite canonical_text only.
    raw_extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_extracted_checksum: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    cleaning_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_cleaning_profiles.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    transformation_metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class CleaningProfile(Base):
    """Versioned document-cleaning config (raw extract → cleaned canonical).

    Distinct from PreprocessingProfile, which operates on research units for
    tokenization / vectorization after segmentation.
    """

    __tablename__ = "research_cleaning_profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(64), default="1.0")
    config_json: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class PreprocessingProfile(Base):
    __tablename__ = "research_preprocessing_profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_json: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Codebook(Base):
    __tablename__ = "research_codebooks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(64), default="1.0")
    is_frozen: Mapped[bool] = mapped_column(default=False)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AnnotationLabel(Base):
    __tablename__ = "research_annotation_labels"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    codebook_id: Mapped[str] = mapped_column(
        ForeignKey("research_codebooks.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    inclusion_criteria: Mapped[str | None] = mapped_column(Text, nullable=True)
    exclusion_criteria: Mapped[str | None] = mapped_column(Text, nullable=True)
    positive_examples_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    negative_examples_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_placeholder: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AnnotationCampaign(Base):
    """Controlled annotation study design (round / reliability set).

    Reliability and blind-coding rules are campaign-scoped so pilot rounds,
    AI-assisted coding, and blind human reliability coding do not mix.
    """

    __tablename__ = "research_annotation_campaigns"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("research_corpora.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    codebook_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_codebooks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    codebook_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    unit_type: Mapped[str] = mapped_column(String(32), default="paragraph")
    sampling_strategy: Mapped[str] = mapped_column(String(64), default="random")
    assignment_strategy: Mapped[str] = mapped_column(String(64), default="overlap")
    sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    overlap_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    overlap_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Derived from annotation_mode; kept explicit for query convenience.
    blind_mode: Mapped[bool] = mapped_column(default=True)
    ai_assistance_enabled: Mapped[bool] = mapped_column(default=False)
    annotation_mode: Mapped[str] = mapped_column(String(32), default="blind_reliability")
    reveal_after: Mapped[str] = mapped_column(String(32), default="campaign_released")
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    annotator_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class AnnotationTask(Base):
    __tablename__ = "research_annotation_tasks"
    __table_args__ = (
        UniqueConstraint(
            "campaign_id",
            "text_unit_id",
            "annotator_id",
            name="uq_annotation_task_campaign_unit_annotator",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    campaign_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_annotation_campaigns.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    text_unit_id: Mapped[str] = mapped_column(
        ForeignKey("research_text_units.id", ondelete="CASCADE"), index=True
    )
    annotator_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="assigned", index=True)
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Annotation(Base):
    __tablename__ = "research_annotations"
    __table_args__ = (
        UniqueConstraint(
            "campaign_id",
            "text_unit_id",
            "label_id",
            "annotator_id",
            "codebook_version",
            name="uq_annotation_campaign_unit_label_annotator_version",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    # NULL denotes legacy/unscoped evidence; it is never inferred into a campaign.
    campaign_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_annotation_campaigns.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    text_unit_id: Mapped[str] = mapped_column(
        ForeignKey("research_text_units.id", ondelete="CASCADE"), index=True
    )
    label_id: Mapped[str] = mapped_column(
        ForeignKey("research_annotation_labels.id", ondelete="CASCADE"), index=True
    )
    annotator_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    value: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    codebook_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Adjudication(Base):
    __tablename__ = "research_adjudications"
    __table_args__ = (
        UniqueConstraint(
            "campaign_id",
            "text_unit_id",
            "label_id",
            "codebook_version",
            name="uq_adjudication_campaign_unit_label_version",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    # NULL denotes a legacy/unscoped adjudication and is not assigned retroactively.
    campaign_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_annotation_campaigns.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    text_unit_id: Mapped[str] = mapped_column(
        ForeignKey("research_text_units.id", ondelete="CASCADE"), index=True
    )
    label_id: Mapped[str] = mapped_column(
        ForeignKey("research_annotation_labels.id", ondelete="CASCADE"), index=True
    )
    codebook_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    final_value: Mapped[str] = mapped_column(String(32))
    adjudicator_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class TrainingDatasetSnapshot(Base):
    __tablename__ = "research_training_dataset_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("research_corpora.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    unit_type: Mapped[str] = mapped_column(String(32))
    codebook_id: Mapped[str] = mapped_column(
        ForeignKey("research_codebooks.id", ondelete="CASCADE"), index=True
    )
    codebook_version: Mapped[str] = mapped_column(String(64))
    annotation_source: Mapped[str] = mapped_column(String(64))
    minimum_agreement: Mapped[float | None] = mapped_column(Float, nullable=True)
    annotation_campaign_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_annotation_campaigns.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    annotation_campaign_snapshot_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    adjudication_policy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    gold_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    unit_ids_json: Mapped[str] = mapped_column(Text)
    document_ids_json: Mapped[str] = mapped_column(Text)
    labels_json: Mapped[str] = mapped_column(Text)
    class_distribution_json: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AnalysisRun(Base):
    __tablename__ = "research_analysis_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    corpus_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_corpora.id", ondelete="SET NULL"), nullable=True, index=True
    )
    run_type: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    run_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    evidence_revision_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    progress_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parameters_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    results_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    execution_key: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    artifact_namespace: Mapped[str | None] = mapped_column(String(512), nullable=True)
    cancellation_requested: Mapped[bool] = mapped_column(default=False)
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ResearchMemo(Base):
    """User-owned research note with an immutable evidence/provenance snapshot."""

    __tablename__ = "research_memos"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    corpus_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_corpora.id", ondelete="SET NULL"), nullable=True, index=True
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(512))
    body: Mapped[str] = mapped_column(Text)
    originating_assistant_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("rag_messages.id", ondelete="SET NULL"), nullable=True, index=True
    )
    originating_synthesis_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_analysis_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    evidence_revision_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    citations_json: Mapped[str] = mapped_column(Text, default="[]", server_default="[]")
    claims_json: Mapped[str] = mapped_column(Text, default="[]", server_default="[]")
    provenance_json: Mapped[str] = mapped_column(Text, default="{}", server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TrainedModel(Base):
    __tablename__ = "research_trained_models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("research_corpora.id", ondelete="CASCADE"), index=True
    )
    analysis_run_id: Mapped[str] = mapped_column(
        ForeignKey("research_analysis_runs.id", ondelete="CASCADE"), index=True
    )
    training_dataset_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("research_training_dataset_snapshots.id", ondelete="CASCADE"), index=True
    )
    model_family: Mapped[str] = mapped_column(String(64))
    task_type: Mapped[str] = mapped_column(String(32))
    label_ids_json: Mapped[str] = mapped_column(Text)
    feature_config_json: Mapped[str] = mapped_column(Text)
    training_config_json: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[str] = mapped_column(Text)
    model_artifact_path: Mapped[str] = mapped_column(String(1024))
    vectorizer_artifact_path: Mapped[str] = mapped_column(String(1024))
    version: Mapped[int] = mapped_column(Integer, default=1)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lifecycle_status: Mapped[str] = mapped_column(String(32), default="candidate", index=True)
    lifecycle_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    lifecycle_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ModelLifecycleEvent(Base):
    """Append-only audit trail for trained-model lifecycle transitions."""

    __tablename__ = "research_model_lifecycle_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    model_id: Mapped[str] = mapped_column(
        ForeignKey("research_trained_models.id", ondelete="CASCADE"), index=True
    )
    from_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str] = mapped_column(String(32))
    actor_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_analysis_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ModelPrediction(Base):
    __tablename__ = "research_model_predictions"
    __table_args__ = (
        UniqueConstraint(
            "trained_model_id",
            "text_unit_id",
            name="uq_prediction_model_unit",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    trained_model_id: Mapped[str] = mapped_column(
        ForeignKey("research_trained_models.id", ondelete="CASCADE"), index=True
    )
    text_unit_id: Mapped[str] = mapped_column(
        ForeignKey("research_text_units.id", ondelete="CASCADE"), index=True
    )
    predicted_labels_json: Mapped[str] = mapped_column(Text)
    scores_json: Mapped[str] = mapped_column(Text)
    uncertainty: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PredictionSet(Base):
    """Header for a batch of model predictions produced by one analysis run.

    Wraps ``ModelPrediction`` rows without overwriting human annotations.
    """

    __tablename__ = "research_prediction_sets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("research_corpora.id", ondelete="CASCADE"), index=True
    )
    trained_model_id: Mapped[str] = mapped_column(
        ForeignKey("research_trained_models.id", ondelete="CASCADE"), index=True
    )
    model_version: Mapped[int] = mapped_column(Integer)
    dataset_snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_training_dataset_snapshots.id", ondelete="SET NULL"),
        nullable=True,
    )
    analysis_run_id: Mapped[str] = mapped_column(
        ForeignKey("research_analysis_runs.id", ondelete="CASCADE"), index=True
    )
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    metadata_json: Mapped[str] = mapped_column(Text)


class DictionaryDefinition(Base):
    __tablename__ = "research_dictionaries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[str] = mapped_column(String(64), default="1.0")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    terms_json: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class TopicLabel(Base):
    __tablename__ = "research_topic_labels"
    __table_args__ = (
        UniqueConstraint("analysis_run_id", "topic_id", name="uq_topic_label_run_topic"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    analysis_run_id: Mapped[str] = mapped_column(
        ForeignKey("research_analysis_runs.id", ondelete="CASCADE"), index=True
    )
    topic_id: Mapped[int] = mapped_column(Integer)
    human_name: Mapped[str] = mapped_column(String(255))
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ContextualDataset(Base):
    __tablename__ = "research_contextual_datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ContextualObservation(Base):
    __tablename__ = "research_contextual_observations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("research_contextual_datasets.id", ondelete="CASCADE"), index=True
    )
    country: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    values_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ResearchAssistantThread(Base):
    """Research-owned binding of a corpus to a generic RAG conversation (I3)."""

    __tablename__ = "research_assistant_threads"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("research_corpora.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    rag_conversation_id: Mapped[str] = mapped_column(
        ForeignKey("rag_conversations.id", ondelete="CASCADE"), unique=True, index=True
    )
    title: Mapped[str] = mapped_column(String(512), default="Ask Corpus")
    scope_mode: Mapped[str] = mapped_column(String(16), default="fixed", server_default="fixed")
    evidence_revision_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    scope_snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ResearchAssistantScopeSnapshot(Base):
    """Immutable corpus evidence scope for an assistant request (invariant I3)."""

    __tablename__ = "research_assistant_scope_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    thread_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_assistant_threads.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    rag_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("rag_messages.id", ondelete="SET NULL"), nullable=True
    )
    retrieval_trace_id: Mapped[str | None] = mapped_column(
        ForeignKey("rag_retrieval_traces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    corpus_id: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    scope_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_revision_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    rag_document_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    corpus_document_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    indexed_rag_document_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    unavailable_rag_document_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    unavailable_corpus_document_ids_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]", server_default="[]"
    )
    unavailable_reasons_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}", server_default="{}"
    )
    document_bindings_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]", server_default="[]"
    )
    scope_mode: Mapped[str] = mapped_column(String(16), default="fixed", server_default="fixed")
    index_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retrieval_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ResearchAssistantScopeEvent(Base):
    """Append-only audit trail for explicit assistant scope changes."""

    __tablename__ = "research_assistant_scope_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    thread_id: Mapped[str] = mapped_column(
        ForeignKey("research_assistant_threads.id", ondelete="CASCADE"), index=True
    )
    actor_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    corpus_id: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    previous_scope_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    new_scope_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_revision_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rag_document_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    corpus_document_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
