"""Async SQLAlchemy repository covering CRUD for all `text_research` entities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.pagination import DEFAULT_PAGE_LIMIT, paginate_scalars
from backend.modules.text_research.domain.models import (
    Adjudication,
    Annotation,
    AnnotationLabel,
    AnnotationTask,
    AnalysisRun,
    Codebook,
    ContextualDataset,
    ContextualObservation,
    CorpusDocument,
    DictionaryDefinition,
    ModelPrediction,
    PreprocessingProfile,
    ResearchCorpus,
    TextUnit,
    TopicLabel,
    TrainedModel,
    TrainingDatasetSnapshot,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ResearchRepository:
    """Repository for the text_research bounded context.

    Grouped by aggregate. All methods flush (not commit) so callers can compose
    multiple writes inside a single unit of work managed by the application layer.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # ------------------------------------------------------------------
    # ResearchCorpus
    # ------------------------------------------------------------------

    async def create_corpus(
        self,
        *,
        project_id: str,
        name: str,
        description: str | None,
        created_by: str,
    ) -> ResearchCorpus:
        row = ResearchCorpus(
            project_id=project_id,
            name=name,
            description=description,
            created_by=created_by,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_corpus(self, corpus_id: str) -> ResearchCorpus | None:
        result = await self.db.execute(
            select(ResearchCorpus).where(ResearchCorpus.id == corpus_id)
        )
        return result.scalar_one_or_none()

    async def list_corpora(self, project_id: str) -> list[ResearchCorpus]:
        result = await self.db.execute(
            select(ResearchCorpus)
            .where(ResearchCorpus.project_id == project_id)
            .order_by(ResearchCorpus.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_corpus(self, corpus: ResearchCorpus, **fields: Any) -> ResearchCorpus:
        for key, value in fields.items():
            if value is not None:
                setattr(corpus, key, value)
        corpus.updated_at = _utcnow()
        await self.db.flush()
        return corpus

    async def delete_corpus(self, corpus: ResearchCorpus) -> None:
        await self.db.delete(corpus)
        await self.db.flush()

    # ------------------------------------------------------------------
    # CorpusDocument
    # ------------------------------------------------------------------

    async def add_document(
        self,
        *,
        corpus_id: str,
        rag_document_id: str,
        title: str | None = None,
        organization: str | None = None,
        organization_type: str | None = None,
        publication_year: int | None = None,
        publication_type: str | None = None,
        country: str | None = None,
        region: str | None = None,
        cultural_sphere: str | None = None,
        language: str | None = None,
        education_level: str | None = None,
        source_url: str | None = None,
        research_notes: str | None = None,
        metadata_json: str | None = None,
    ) -> CorpusDocument:
        row = CorpusDocument(
            corpus_id=corpus_id,
            rag_document_id=rag_document_id,
            title=title,
            organization=organization,
            organization_type=organization_type,
            publication_year=publication_year,
            publication_type=publication_type,
            country=country,
            region=region,
            cultural_sphere=cultural_sphere,
            language=language,
            education_level=education_level,
            source_url=source_url,
            research_notes=research_notes,
            metadata_json=metadata_json,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_document(self, document_id: str) -> CorpusDocument | None:
        result = await self.db.execute(
            select(CorpusDocument).where(CorpusDocument.id == document_id)
        )
        return result.scalar_one_or_none()

    async def get_document_by_rag_id(
        self, *, corpus_id: str, rag_document_id: str
    ) -> CorpusDocument | None:
        result = await self.db.execute(
            select(CorpusDocument).where(
                CorpusDocument.corpus_id == corpus_id,
                CorpusDocument.rag_document_id == rag_document_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        corpus_id: str,
        *,
        organization: str | None = None,
        publication_year: int | None = None,
        region: str | None = None,
        cultural_sphere: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[CorpusDocument]:
        stmt = select(CorpusDocument).where(CorpusDocument.corpus_id == corpus_id)
        if organization:
            stmt = stmt.where(CorpusDocument.organization == organization)
        if publication_year:
            stmt = stmt.where(CorpusDocument.publication_year == publication_year)
        if region:
            stmt = stmt.where(CorpusDocument.region == region)
        if cultural_sphere:
            stmt = stmt.where(CorpusDocument.cultural_sphere == cultural_sphere)
        stmt = stmt.order_by(CorpusDocument.created_at.asc())
        if limit is not None:
            stmt = stmt.offset(offset).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def count_documents(self, corpus_id: str) -> int:
        result = await self.db.execute(
            select(func.count()).where(CorpusDocument.corpus_id == corpus_id)
        )
        return int(result.scalar() or 0)

    async def update_document(self, document: CorpusDocument, **fields: Any) -> CorpusDocument:
        for key, value in fields.items():
            if value is not None:
                setattr(document, key, value)
        document.updated_at = _utcnow()
        await self.db.flush()
        return document

    async def delete_document(self, document: CorpusDocument) -> None:
        await self.db.delete(document)
        await self.db.flush()

    # ------------------------------------------------------------------
    # TextUnit
    # ------------------------------------------------------------------

    async def bulk_create_text_units(self, rows: list[TextUnit]) -> list[TextUnit]:
        self.db.add_all(rows)
        await self.db.flush()
        return rows

    async def delete_text_units_for_document(
        self, corpus_document_id: str, unit_type: str
    ) -> None:
        await self.db.execute(
            delete(TextUnit).where(
                TextUnit.corpus_document_id == corpus_document_id,
                TextUnit.unit_type == unit_type,
            )
        )
        await self.db.flush()

    async def get_text_unit(self, unit_id: str) -> TextUnit | None:
        result = await self.db.execute(select(TextUnit).where(TextUnit.id == unit_id))
        return result.scalar_one_or_none()

    async def list_text_units_by_ids(self, unit_ids: list[str]) -> list[TextUnit]:
        if not unit_ids:
            return []
        result = await self.db.execute(select(TextUnit).where(TextUnit.id.in_(unit_ids)))
        return list(result.scalars().all())

    async def list_text_units_for_document(
        self, corpus_document_id: str, unit_type: str | None = None
    ) -> list[TextUnit]:
        stmt = select(TextUnit).where(TextUnit.corpus_document_id == corpus_document_id)
        if unit_type:
            stmt = stmt.where(TextUnit.unit_type == unit_type)
        stmt = stmt.order_by(TextUnit.position.asc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def list_text_units_for_corpus(
        self,
        corpus_id: str,
        *,
        unit_type: str | None = None,
        document_ids: list[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TextUnit]:
        stmt = (
            select(TextUnit)
            .join(CorpusDocument, CorpusDocument.id == TextUnit.corpus_document_id)
            .where(CorpusDocument.corpus_id == corpus_id)
        )
        if unit_type:
            stmt = stmt.where(TextUnit.unit_type == unit_type)
        if document_ids:
            stmt = stmt.where(TextUnit.corpus_document_id.in_(document_ids))
        stmt = stmt.order_by(TextUnit.corpus_document_id.asc(), TextUnit.position.asc())
        if limit is not None:
            stmt = stmt.offset(offset).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def count_text_units_for_corpus(
        self, corpus_id: str, *, unit_type: str | None = None
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(TextUnit)
            .join(CorpusDocument, CorpusDocument.id == TextUnit.corpus_document_id)
            .where(CorpusDocument.corpus_id == corpus_id)
        )
        if unit_type:
            stmt = stmt.where(TextUnit.unit_type == unit_type)
        result = await self.db.execute(stmt)
        return int(result.scalar() or 0)

    async def get_corpus_id_for_document(self, corpus_document_id: str) -> str | None:
        result = await self.db.execute(
            select(CorpusDocument.corpus_id).where(CorpusDocument.id == corpus_document_id)
        )
        return result.scalar_one_or_none()

    async def get_corpus_id_for_text_unit(self, text_unit_id: str) -> str | None:
        result = await self.db.execute(
            select(CorpusDocument.corpus_id)
            .join(TextUnit, TextUnit.corpus_document_id == CorpusDocument.id)
            .where(TextUnit.id == text_unit_id)
        )
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # PreprocessingProfile
    # ------------------------------------------------------------------

    async def create_preprocessing_profile(
        self,
        *,
        project_id: str,
        name: str,
        description: str | None,
        config_json: str,
        created_by: str,
    ) -> PreprocessingProfile:
        row = PreprocessingProfile(
            project_id=project_id,
            name=name,
            description=description,
            config_json=config_json,
            created_by=created_by,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_preprocessing_profile(self, profile_id: str) -> PreprocessingProfile | None:
        result = await self.db.execute(
            select(PreprocessingProfile).where(PreprocessingProfile.id == profile_id)
        )
        return result.scalar_one_or_none()

    async def list_preprocessing_profiles(self, project_id: str) -> list[PreprocessingProfile]:
        result = await self.db.execute(
            select(PreprocessingProfile)
            .where(PreprocessingProfile.project_id == project_id)
            .order_by(PreprocessingProfile.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_preprocessing_profile(
        self, profile: PreprocessingProfile, **fields: Any
    ) -> PreprocessingProfile:
        for key, value in fields.items():
            if value is not None:
                setattr(profile, key, value)
        profile.updated_at = _utcnow()
        await self.db.flush()
        return profile

    async def delete_preprocessing_profile(self, profile: PreprocessingProfile) -> None:
        await self.db.delete(profile)
        await self.db.flush()

    # ------------------------------------------------------------------
    # Codebook / AnnotationLabel
    # ------------------------------------------------------------------

    async def create_codebook(
        self,
        *,
        project_id: str,
        name: str,
        description: str | None,
        version: str,
        created_by: str,
    ) -> Codebook:
        row = Codebook(
            project_id=project_id,
            name=name,
            description=description,
            version=version,
            created_by=created_by,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_codebook(self, codebook_id: str) -> Codebook | None:
        result = await self.db.execute(select(Codebook).where(Codebook.id == codebook_id))
        return result.scalar_one_or_none()

    async def list_codebooks(self, project_id: str) -> list[Codebook]:
        result = await self.db.execute(
            select(Codebook)
            .where(Codebook.project_id == project_id)
            .order_by(Codebook.created_at.desc())
        )
        return list(result.scalars().all())

    async def list_codebook_versions(self, project_id: str, name: str) -> list[Codebook]:
        result = await self.db.execute(
            select(Codebook)
            .where(Codebook.project_id == project_id, Codebook.name == name)
            .order_by(Codebook.created_at.asc())
        )
        return list(result.scalars().all())

    async def freeze_codebook(self, codebook: Codebook) -> Codebook:
        codebook.is_frozen = True
        await self.db.flush()
        return codebook

    async def create_label(
        self,
        *,
        codebook_id: str,
        name: str,
        description: str | None = None,
        inclusion_criteria: str | None = None,
        exclusion_criteria: str | None = None,
        positive_examples_json: str | None = None,
        negative_examples_json: str | None = None,
        is_placeholder: bool = True,
    ) -> AnnotationLabel:
        row = AnnotationLabel(
            codebook_id=codebook_id,
            name=name,
            description=description,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            positive_examples_json=positive_examples_json,
            negative_examples_json=negative_examples_json,
            is_placeholder=is_placeholder,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_label(self, label_id: str) -> AnnotationLabel | None:
        result = await self.db.execute(
            select(AnnotationLabel).where(AnnotationLabel.id == label_id)
        )
        return result.scalar_one_or_none()

    async def list_labels(self, codebook_id: str) -> list[AnnotationLabel]:
        result = await self.db.execute(
            select(AnnotationLabel)
            .where(AnnotationLabel.codebook_id == codebook_id)
            .order_by(AnnotationLabel.created_at.asc())
        )
        return list(result.scalars().all())

    async def update_label(self, label: AnnotationLabel, **fields: Any) -> AnnotationLabel:
        for key, value in fields.items():
            if value is not None:
                setattr(label, key, value)
        await self.db.flush()
        return label

    # ------------------------------------------------------------------
    # AnnotationTask
    # ------------------------------------------------------------------

    async def get_task(self, text_unit_id: str, annotator_id: str) -> AnnotationTask | None:
        result = await self.db.execute(
            select(AnnotationTask).where(
                AnnotationTask.text_unit_id == text_unit_id,
                AnnotationTask.annotator_id == annotator_id,
            )
        )
        return result.scalar_one_or_none()

    async def create_task(
        self, *, text_unit_id: str, annotator_id: str, status: str = "assigned"
    ) -> AnnotationTask:
        row = AnnotationTask(
            text_unit_id=text_unit_id, annotator_id=annotator_id, status=status
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def list_tasks_for_annotator(
        self, annotator_id: str, *, text_unit_ids: list[str] | None = None
    ) -> list[AnnotationTask]:
        stmt = select(AnnotationTask).where(AnnotationTask.annotator_id == annotator_id)
        if text_unit_ids is not None:
            stmt = stmt.where(AnnotationTask.text_unit_id.in_(text_unit_ids))
        stmt = stmt.order_by(AnnotationTask.assigned_at.asc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def list_tasks_for_units(self, text_unit_ids: list[str]) -> list[AnnotationTask]:
        if not text_unit_ids:
            return []
        result = await self.db.execute(
            select(AnnotationTask).where(AnnotationTask.text_unit_id.in_(text_unit_ids))
        )
        return list(result.scalars().all())

    async def update_task_status(self, task: AnnotationTask, status: str) -> AnnotationTask:
        task.status = status
        if status == "completed":
            task.completed_at = _utcnow()
        await self.db.flush()
        return task

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    async def get_annotation(
        self,
        *,
        text_unit_id: str,
        label_id: str,
        annotator_id: str,
        codebook_version: str,
    ) -> Annotation | None:
        result = await self.db.execute(
            select(Annotation).where(
                Annotation.text_unit_id == text_unit_id,
                Annotation.label_id == label_id,
                Annotation.annotator_id == annotator_id,
                Annotation.codebook_version == codebook_version,
            )
        )
        return result.scalar_one_or_none()

    async def upsert_annotation(
        self,
        *,
        text_unit_id: str,
        label_id: str,
        annotator_id: str,
        codebook_version: str,
        value: str,
        confidence: float | None = None,
        comment: str | None = None,
    ) -> Annotation:
        """Update-in-place by unique key; never delete annotation history silently."""
        existing = await self.get_annotation(
            text_unit_id=text_unit_id,
            label_id=label_id,
            annotator_id=annotator_id,
            codebook_version=codebook_version,
        )
        if existing is not None:
            existing.value = value
            existing.confidence = confidence
            existing.comment = comment
            existing.updated_at = _utcnow()
            await self.db.flush()
            return existing
        row = Annotation(
            text_unit_id=text_unit_id,
            label_id=label_id,
            annotator_id=annotator_id,
            codebook_version=codebook_version,
            value=value,
            confidence=confidence,
            comment=comment,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def list_annotations_for_unit(self, text_unit_id: str) -> list[Annotation]:
        result = await self.db.execute(
            select(Annotation).where(Annotation.text_unit_id == text_unit_id)
        )
        return list(result.scalars().all())

    async def list_annotations_for_units(self, text_unit_ids: list[str]) -> list[Annotation]:
        if not text_unit_ids:
            return []
        result = await self.db.execute(
            select(Annotation).where(Annotation.text_unit_id.in_(text_unit_ids))
        )
        return list(result.scalars().all())

    async def list_annotations_for_corpus(self, corpus_id: str) -> list[Annotation]:
        result = await self.db.execute(
            select(Annotation)
            .join(TextUnit, TextUnit.id == Annotation.text_unit_id)
            .join(CorpusDocument, CorpusDocument.id == TextUnit.corpus_document_id)
            .where(CorpusDocument.corpus_id == corpus_id)
        )
        return list(result.scalars().all())

    async def list_annotations_for_annotator(
        self, annotator_id: str, *, text_unit_ids: list[str] | None = None
    ) -> list[Annotation]:
        stmt = select(Annotation).where(Annotation.annotator_id == annotator_id)
        if text_unit_ids is not None:
            stmt = stmt.where(Annotation.text_unit_id.in_(text_unit_ids))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Adjudication
    # ------------------------------------------------------------------

    async def get_adjudication(
        self, *, text_unit_id: str, label_id: str
    ) -> Adjudication | None:
        result = await self.db.execute(
            select(Adjudication).where(
                Adjudication.text_unit_id == text_unit_id,
                Adjudication.label_id == label_id,
            )
        )
        return result.scalar_one_or_none()

    async def upsert_adjudication(
        self,
        *,
        text_unit_id: str,
        label_id: str,
        final_value: str,
        adjudicator_id: str,
        comment: str | None = None,
    ) -> Adjudication:
        existing = await self.get_adjudication(text_unit_id=text_unit_id, label_id=label_id)
        if existing is not None:
            existing.final_value = final_value
            existing.adjudicator_id = adjudicator_id
            existing.comment = comment
            await self.db.flush()
            return existing
        row = Adjudication(
            text_unit_id=text_unit_id,
            label_id=label_id,
            final_value=final_value,
            adjudicator_id=adjudicator_id,
            comment=comment,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def list_adjudications_for_units(self, text_unit_ids: list[str]) -> list[Adjudication]:
        if not text_unit_ids:
            return []
        result = await self.db.execute(
            select(Adjudication).where(Adjudication.text_unit_id.in_(text_unit_ids))
        )
        return list(result.scalars().all())

    async def list_adjudications_for_corpus(self, corpus_id: str) -> list[Adjudication]:
        result = await self.db.execute(
            select(Adjudication)
            .join(TextUnit, TextUnit.id == Adjudication.text_unit_id)
            .join(CorpusDocument, CorpusDocument.id == TextUnit.corpus_document_id)
            .where(CorpusDocument.corpus_id == corpus_id)
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # TrainingDatasetSnapshot
    # ------------------------------------------------------------------

    async def create_snapshot(self, snapshot: TrainingDatasetSnapshot) -> TrainingDatasetSnapshot:
        self.db.add(snapshot)
        await self.db.flush()
        return snapshot

    async def get_snapshot(self, snapshot_id: str) -> TrainingDatasetSnapshot | None:
        result = await self.db.execute(
            select(TrainingDatasetSnapshot).where(TrainingDatasetSnapshot.id == snapshot_id)
        )
        return result.scalar_one_or_none()

    async def list_snapshots(self, project_id: str, *, corpus_id: str | None = None):
        stmt = select(TrainingDatasetSnapshot).where(
            TrainingDatasetSnapshot.project_id == project_id
        )
        if corpus_id:
            stmt = stmt.where(TrainingDatasetSnapshot.corpus_id == corpus_id)
        stmt = stmt.order_by(TrainingDatasetSnapshot.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # AnalysisRun
    # ------------------------------------------------------------------

    async def create_run(self, run: AnalysisRun) -> AnalysisRun:
        self.db.add(run)
        await self.db.flush()
        return run

    async def get_run(self, run_id: str) -> AnalysisRun | None:
        result = await self.db.execute(select(AnalysisRun).where(AnalysisRun.id == run_id))
        return result.scalar_one_or_none()

    async def update_run(self, run: AnalysisRun, **fields: Any) -> AnalysisRun:
        for key, value in fields.items():
            setattr(run, key, value)
        await self.db.flush()
        return run

    async def list_runs(
        self,
        project_id: str,
        *,
        corpus_id: str | None = None,
        run_type: str | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[AnalysisRun], int]:
        stmt = select(AnalysisRun).where(AnalysisRun.project_id == project_id)
        if corpus_id:
            stmt = stmt.where(AnalysisRun.corpus_id == corpus_id)
        if run_type:
            stmt = stmt.where(AnalysisRun.run_type == run_type)
        stmt = stmt.order_by(AnalysisRun.created_at.desc())
        return await paginate_scalars(self.db, stmt, limit=limit, offset=offset)

    # ------------------------------------------------------------------
    # TrainedModel
    # ------------------------------------------------------------------

    async def create_model(self, model: TrainedModel) -> TrainedModel:
        self.db.add(model)
        await self.db.flush()
        return model

    async def get_model(self, model_id: str) -> TrainedModel | None:
        result = await self.db.execute(select(TrainedModel).where(TrainedModel.id == model_id))
        return result.scalar_one_or_none()

    async def list_models(self, project_id: str, *, corpus_id: str | None = None):
        stmt = select(TrainedModel).where(TrainedModel.project_id == project_id)
        if corpus_id:
            stmt = stmt.where(TrainedModel.corpus_id == corpus_id)
        stmt = stmt.order_by(TrainedModel.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def next_model_version(self, corpus_id: str) -> int:
        result = await self.db.execute(
            select(func.max(TrainedModel.version)).where(TrainedModel.corpus_id == corpus_id)
        )
        current = result.scalar()
        return int(current or 0) + 1

    # ------------------------------------------------------------------
    # ModelPrediction
    # ------------------------------------------------------------------

    async def upsert_prediction(
        self,
        *,
        trained_model_id: str,
        text_unit_id: str,
        predicted_labels_json: str,
        scores_json: str,
        uncertainty: float | None,
    ) -> ModelPrediction:
        result = await self.db.execute(
            select(ModelPrediction).where(
                ModelPrediction.trained_model_id == trained_model_id,
                ModelPrediction.text_unit_id == text_unit_id,
            )
        )
        existing = result.scalar_one_or_none()
        if existing is not None:
            existing.predicted_labels_json = predicted_labels_json
            existing.scores_json = scores_json
            existing.uncertainty = uncertainty
            await self.db.flush()
            return existing
        row = ModelPrediction(
            trained_model_id=trained_model_id,
            text_unit_id=text_unit_id,
            predicted_labels_json=predicted_labels_json,
            scores_json=scores_json,
            uncertainty=uncertainty,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def list_predictions_for_model(
        self,
        trained_model_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
        order_by_uncertainty: bool = False,
    ) -> tuple[list[ModelPrediction], int]:
        stmt = select(ModelPrediction).where(ModelPrediction.trained_model_id == trained_model_id)
        if order_by_uncertainty:
            # uncertainty convention: higher value == more uncertain (see
            # infrastructure/classifiers.py); most-uncertain-first ranking.
            stmt = stmt.order_by(ModelPrediction.uncertainty.desc().nulls_last())
        else:
            stmt = stmt.order_by(ModelPrediction.created_at.desc())
        return await paginate_scalars(self.db, stmt, limit=limit, offset=offset)

    async def list_predicted_unit_ids(self, trained_model_id: str) -> set[str]:
        result = await self.db.execute(
            select(ModelPrediction.text_unit_id).where(
                ModelPrediction.trained_model_id == trained_model_id
            )
        )
        return set(result.scalars().all())

    # ------------------------------------------------------------------
    # DictionaryDefinition
    # ------------------------------------------------------------------

    async def create_dictionary(self, dictionary: DictionaryDefinition) -> DictionaryDefinition:
        self.db.add(dictionary)
        await self.db.flush()
        return dictionary

    async def get_dictionary(self, dictionary_id: str) -> DictionaryDefinition | None:
        result = await self.db.execute(
            select(DictionaryDefinition).where(DictionaryDefinition.id == dictionary_id)
        )
        return result.scalar_one_or_none()

    async def list_dictionaries(self, project_id: str) -> list[DictionaryDefinition]:
        result = await self.db.execute(
            select(DictionaryDefinition)
            .where(DictionaryDefinition.project_id == project_id)
            .order_by(DictionaryDefinition.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_dictionary(
        self, dictionary: DictionaryDefinition, **fields: Any
    ) -> DictionaryDefinition:
        for key, value in fields.items():
            if value is not None:
                setattr(dictionary, key, value)
        await self.db.flush()
        return dictionary

    # ------------------------------------------------------------------
    # TopicLabel
    # ------------------------------------------------------------------

    async def upsert_topic_label(
        self, *, analysis_run_id: str, topic_id: int, human_name: str, created_by: str
    ) -> TopicLabel:
        result = await self.db.execute(
            select(TopicLabel).where(
                TopicLabel.analysis_run_id == analysis_run_id,
                TopicLabel.topic_id == topic_id,
            )
        )
        existing = result.scalar_one_or_none()
        if existing is not None:
            existing.human_name = human_name
            await self.db.flush()
            return existing
        row = TopicLabel(
            analysis_run_id=analysis_run_id,
            topic_id=topic_id,
            human_name=human_name,
            created_by=created_by,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def list_topic_labels(self, analysis_run_id: str) -> list[TopicLabel]:
        result = await self.db.execute(
            select(TopicLabel).where(TopicLabel.analysis_run_id == analysis_run_id)
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # ContextualDataset / ContextualObservation
    # ------------------------------------------------------------------

    async def create_contextual_dataset(
        self, *, project_id: str, name: str, description: str | None, created_by: str
    ) -> ContextualDataset:
        row = ContextualDataset(
            project_id=project_id, name=name, description=description, created_by=created_by
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def list_contextual_datasets(self, project_id: str) -> list[ContextualDataset]:
        result = await self.db.execute(
            select(ContextualDataset).where(ContextualDataset.project_id == project_id)
        )
        return list(result.scalars().all())

    async def get_contextual_dataset(self, dataset_id: str) -> ContextualDataset | None:
        result = await self.db.execute(
            select(ContextualDataset).where(ContextualDataset.id == dataset_id)
        )
        return result.scalar_one_or_none()

    async def bulk_create_observations(
        self, rows: list[ContextualObservation]
    ) -> list[ContextualObservation]:
        self.db.add_all(rows)
        await self.db.flush()
        return rows

    async def list_observations(self, dataset_id: str) -> list[ContextualObservation]:
        result = await self.db.execute(
            select(ContextualObservation).where(ContextualObservation.dataset_id == dataset_id)
        )
        return list(result.scalars().all())
