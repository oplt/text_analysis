"""Reproducible exports: CSV extracts for text units / annotations /
predictions, a JSON `analysis_manifest.json` describing the exact corpus
state and analysis history, and an optional R script showing how to load the
unit-level CSV export into `quanteda` for independent replication.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import Any

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.quantitative_analysis_service import (
    _apply_document_filters,
)
from backend.modules.text_research.domain.models import dumps, loads

APP_NAME = "Policy Text Lab"
APP_SUBTITLE = "Computational Analysis of Global Education Policy Discourses"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ExportService(ResearchAccessMixin):
    async def export_units_csv(
        self, corpus_id: str, *, user_id: str, unit_type: str, **filters: Any
    ) -> str:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        documents = await self.repo.list_documents(corpus_id)
        filtered_docs = _apply_document_filters(documents, filters or {})
        doc_ids = [d.id for d in filtered_docs]
        doc_lookup = {d.id: d for d in filtered_docs}
        units = await self.repo.list_text_units_for_corpus(
            corpus_id, unit_type=unit_type, document_ids=doc_ids if filters else None
        )

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "text_unit_id",
                "corpus_document_id",
                "document_title",
                "organization",
                "organization_type",
                "region",
                "cultural_sphere",
                "country",
                "language",
                "publication_year",
                "publication_type",
                "unit_type",
                "position",
                "text",
            ]
        )
        for unit in units:
            doc = doc_lookup.get(unit.corpus_document_id)
            writer.writerow(
                [
                    unit.id,
                    unit.corpus_document_id,
                    doc.title if doc else "",
                    doc.organization if doc else "",
                    doc.organization_type if doc else "",
                    doc.region if doc else "",
                    doc.cultural_sphere if doc else "",
                    doc.country if doc else "",
                    doc.language if doc else "",
                    doc.publication_year if doc else "",
                    doc.publication_type if doc else "",
                    unit.unit_type,
                    unit.position,
                    unit.text,
                ]
            )
        return buffer.getvalue()

    async def export_annotations_csv(
        self, corpus_id: str, *, user_id: str, codebook_id: str | None = None
    ) -> str:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        annotations = await self.repo.list_annotations_for_corpus(corpus_id)
        if codebook_id:
            codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)
            annotations = [a for a in annotations if a.codebook_version == codebook.version]

        label_ids = {a.label_id for a in annotations}
        label_names = {}
        for label_id in label_ids:
            label = await self.repo.get_label(label_id)
            label_names[label_id] = label.name if label else label_id

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "text_unit_id",
                "label_id",
                "label_name",
                "annotator_id",
                "value",
                "confidence",
                "comment",
                "codebook_version",
                "created_at",
                "updated_at",
            ]
        )
        for a in annotations:
            writer.writerow(
                [
                    a.text_unit_id,
                    a.label_id,
                    label_names.get(a.label_id, a.label_id),
                    a.annotator_id,
                    a.value,
                    a.confidence if a.confidence is not None else "",
                    a.comment or "",
                    a.codebook_version,
                    a.created_at.isoformat() if a.created_at else "",
                    a.updated_at.isoformat() if a.updated_at else "",
                ]
            )
        return buffer.getvalue()

    async def export_predictions_csv(self, model_id: str, *, user_id: str) -> str:
        model = await self.get_model_or_404(model_id, user_id=user_id)
        predictions, _total = await self.repo.list_predictions_for_model(
            model.id, limit=1_000_000, offset=0
        )

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            ["text_unit_id", "trained_model_id", "predicted_labels", "scores", "uncertainty", "created_at"]
        )
        for prediction in predictions:
            writer.writerow(
                [
                    prediction.text_unit_id,
                    prediction.trained_model_id,
                    "|".join(loads(prediction.predicted_labels_json, [])),
                    dumps(loads(prediction.scores_json, {})),
                    prediction.uncertainty if prediction.uncertainty is not None else "",
                    prediction.created_at.isoformat() if prediction.created_at else "",
                ]
            )
        return buffer.getvalue()

    async def build_manifest(self, corpus_id: str, *, user_id: str) -> dict[str, Any]:
        """`analysis_manifest.json` — a reproducibility snapshot: exact corpus
        composition, codebooks/versions, preprocessing profiles, frozen
        training snapshots, trained models with metrics + artifact paths, and
        a summary of every analysis run, so a third party can audit or
        reproduce the pipeline end to end."""
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        documents = await self.repo.list_documents(corpus_id)
        codebooks = await self.repo.list_codebooks(corpus.project_id)
        profiles = await self.repo.list_preprocessing_profiles(corpus.project_id)
        models = await self.repo.list_models(corpus.project_id, corpus_id=corpus_id)
        snapshots = await self.repo.list_snapshots(corpus.project_id, corpus_id=corpus_id)
        runs, total_runs = await self.repo.list_runs(
            corpus.project_id, corpus_id=corpus_id, limit=200, offset=0
        )

        return {
            "app": APP_NAME,
            "subtitle": APP_SUBTITLE,
            "generated_at": _utcnow().isoformat(),
            "corpus": {
                "id": corpus.id,
                "name": corpus.name,
                "description": corpus.description,
                "document_count": len(documents),
            },
            "documents": [
                {
                    "id": d.id,
                    "title": d.title,
                    "organization": d.organization,
                    "region": d.region,
                    "cultural_sphere": d.cultural_sphere,
                    "country": d.country,
                    "language": d.language,
                    "publication_year": d.publication_year,
                    "publication_type": d.publication_type,
                }
                for d in documents
            ],
            "codebooks": [
                {"id": c.id, "name": c.name, "version": c.version, "is_frozen": c.is_frozen}
                for c in codebooks
            ],
            "preprocessing_profiles": [
                {"id": p.id, "name": p.name, "config": loads(p.config_json, {})} for p in profiles
            ],
            "training_dataset_snapshots": [
                {
                    "id": s.id,
                    "name": s.name,
                    "unit_type": s.unit_type,
                    "codebook_id": s.codebook_id,
                    "codebook_version": s.codebook_version,
                    "annotation_source": s.annotation_source,
                    "unit_count": len(loads(s.unit_ids_json, [])),
                }
                for s in snapshots
            ],
            "trained_models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "version": m.version,
                    "model_family": m.model_family,
                    "task_type": m.task_type,
                    "labels": loads(m.label_ids_json, []),
                    "metrics": loads(m.metrics_json, {}),
                    "training_dataset_snapshot_id": m.training_dataset_snapshot_id,
                    "model_artifact_path": m.model_artifact_path,
                    "vectorizer_artifact_path": m.vectorizer_artifact_path,
                }
                for m in models
            ],
            "analysis_runs": {
                "total": total_runs,
                "included": len(runs),
                "runs": [
                    {
                        "id": r.id,
                        "run_type": r.run_type,
                        "status": r.status,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                        "parameters": loads(r.parameters_json, {}),
                        "metrics": loads(r.metrics_json, {}),
                        "results": loads(r.results_json, {}),
                        "artifact_path": r.artifact_path,
                        "random_seed": r.random_seed,
                        "error_message": r.error_message,
                    }
                    for r in runs
                ],
            },
        }

    async def export_run_json(self, run_id: str, *, user_id: str) -> dict[str, Any]:
        run = await self.get_run_or_404(run_id, user_id=user_id)
        return {
            "id": run.id,
            "project_id": run.project_id,
            "corpus_id": run.corpus_id,
            "run_type": run.run_type,
            "status": run.status,
            "progress_stage": run.progress_stage,
            "parameters": loads(run.parameters_json, {}),
            "metrics": loads(run.metrics_json, {}),
            "results": loads(run.results_json, {}),
            "artifact_path": run.artifact_path,
            "random_seed": run.random_seed,
            "created_by": run.created_by,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "error_message": run.error_message,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }

    async def export_codebook_json(self, codebook_id: str, *, user_id: str) -> dict[str, Any]:
        codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)
        labels = await self.repo.list_labels(codebook_id)
        return {
            "id": codebook.id,
            "project_id": codebook.project_id,
            "name": codebook.name,
            "description": codebook.description,
            "version": codebook.version,
            "is_frozen": codebook.is_frozen,
            "labels": [
                {
                    "id": label.id,
                    "name": label.name,
                    "description": label.description,
                    "inclusion_criteria": label.inclusion_criteria,
                    "exclusion_criteria": label.exclusion_criteria,
                    "positive_examples": loads(label.positive_examples_json, []),
                    "negative_examples": loads(label.negative_examples_json, []),
                    "is_placeholder": label.is_placeholder,
                }
                for label in labels
            ],
        }

    async def export_preprocessing_profile_json(
        self, profile_id: str, *, user_id: str
    ) -> dict[str, Any]:
        profile = await self.get_preprocessing_profile_or_404(profile_id, user_id=user_id)
        return {
            "id": profile.id,
            "project_id": profile.project_id,
            "name": profile.name,
            "description": profile.description,
            "config": loads(profile.config_json, {}),
            "created_by": profile.created_by,
            "created_at": profile.created_at.isoformat() if profile.created_at else None,
            "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
        }

    async def export_model_metrics(self, model_id: str, *, user_id: str) -> dict[str, Any]:
        model = await self.get_model_or_404(model_id, user_id=user_id)
        return {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "model_family": model.model_family,
            "task_type": model.task_type,
            "labels": loads(model.label_ids_json, []),
            "feature_config": loads(model.feature_config_json, {}),
            "training_config": loads(model.training_config_json, {}),
            "metrics": loads(model.metrics_json, {}),
            "model_artifact_path": model.model_artifact_path,
            "vectorizer_artifact_path": model.vectorizer_artifact_path,
            "created_at": model.created_at.isoformat() if model.created_at else None,
        }

    async def build_quanteda_script(
        self, corpus_id: str, *, user_id: str, csv_filename: str = "units_export.csv"
    ) -> str:
        """A minimal R script demonstrating independent replication of the
        exported unit-level CSV in `quanteda`. Provided for researchers who
        want to cross-check results outside this application; it is *not*
        executed by the backend."""
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        return f'''# {APP_NAME} -- {APP_SUBTITLE}
# Reproducibility script for corpus "{corpus.name}" ({corpus.id})
# Generated {_utcnow().isoformat()}
#
# Loads the unit-level CSV export produced by this application into
# quanteda for independent verification of quantitative results.

library(quanteda)
library(readr)

units <- read_csv("{csv_filename}")

corp <- corpus(
  units,
  docid_field = "text_unit_id",
  text_field = "text"
)

toks <- tokens(
  corp,
  remove_punct = TRUE,
  remove_numbers = FALSE
)

dfmat <- dfm(toks) |> dfm_tolower()

# Example: term frequencies
topfeatures(dfmat, 25)

# Example: group document-feature matrix by organization
dfmat_by_org <- dfm_group(dfmat, groups = docvars(corp, "organization"))
'''
