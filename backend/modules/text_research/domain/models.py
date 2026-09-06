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
    rag_document_id: Mapped[str] = mapped_column(
        ForeignKey("rag_documents.id", ondelete="CASCADE"), index=True
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
    text: Mapped[str] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


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


class AnnotationTask(Base):
    __tablename__ = "research_annotation_tasks"
    __table_args__ = (
        UniqueConstraint("text_unit_id", "annotator_id", name="uq_annotation_task_unit_annotator"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
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
            "text_unit_id",
            "label_id",
            "annotator_id",
            "codebook_version",
            name="uq_annotation_unit_label_annotator_version",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
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
        UniqueConstraint("text_unit_id", "label_id", name="uq_adjudication_unit_label"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
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
    progress_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parameters_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    results_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


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
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
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
