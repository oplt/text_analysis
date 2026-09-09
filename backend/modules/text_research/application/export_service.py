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
from backend.modules.text_research.infrastructure.provenance import (
    container_image_digest,
    extract_reproduce_request,
    git_commit_sha,
    library_versions,
    runtime_environment,
)

APP_NAME = "Text Research"

APP_SUBTITLE = "Generic computational text analysis workspace"


def _csv_line(row: list[Any]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer).writerow(row)
    return buffer.getvalue()


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ExportService(ResearchAccessMixin):
    async def iter_units_csv(
        self, corpus_id: str, *, user_id: str, unit_type: str, **filters: Any
    ):
        """Yield unit CSV rows incrementally for HTTP streaming exports."""
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        documents = await self.repo.list_documents(corpus_id)
        filtered_docs = _apply_document_filters(documents, filters or {})
        doc_lookup = {document.id: document for document in filtered_docs}
        units = await self.repo.list_text_units_for_corpus(
            corpus_id, unit_type=unit_type, document_ids=list(doc_lookup) if filters else None
        )
        yield _csv_line(["text_unit_id", "corpus_document_id", "document_title", "text"])
        for unit in units:
            document = doc_lookup.get(unit.corpus_document_id)
            yield _csv_line([unit.id, unit.corpus_document_id, document.title if document else "", unit.text])

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
                "page_number",
                "paragraph_number",
                "sentence_number",
                "char_start",
                "char_end",
                "section_heading",
                "text_hash",
                "source_text_hash",
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
                    unit.page_number if unit.page_number is not None else "",
                    unit.paragraph_number if unit.paragraph_number is not None else "",
                    unit.sentence_number if unit.sentence_number is not None else "",
                    unit.char_start if unit.char_start is not None else "",
                    unit.char_end if unit.char_end is not None else "",
                    unit.section_heading or "",
                    unit.text_hash,
                    unit.source_text_hash or "",
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
        label_names = {
            label.id: label.name for label in await self.repo.list_labels_by_ids(label_ids)
        }

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
        canonical_sources = await self.repo.list_canonical_sources_for_corpus(corpus_id)
        canonical_by_doc = {row.corpus_document_id: row for row in canonical_sources}
        cleaning_profiles = await self.repo.list_cleaning_profiles(corpus.project_id)
        runtime = runtime_environment()

        return {
            "app": APP_NAME,
            "subtitle": APP_SUBTITLE,
            "generated_at": _utcnow().isoformat(),
            "reproducibility": {
                "library_versions": library_versions(),
                "git_commit": git_commit_sha(),
                "container_image_digest": container_image_digest(),
                "engine_version": runtime["engine_version"],
                "preprocessing_implementation": runtime["preprocessing_implementation"],
                "preprocessing_implementation_version": runtime[
                    "preprocessing_implementation_version"
                ],
            },
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
                    "canonical_source": (
                        {
                            "checksum": canonical_by_doc[d.id].canonical_text_checksum,
                            "parser_name": canonical_by_doc[d.id].parser_name,
                            "parser_version": canonical_by_doc[d.id].parser_version,
                            "extracted_at": canonical_by_doc[d.id].extracted_at.isoformat(),
                            "original_file_checksum": canonical_by_doc[d.id].original_file_checksum,
                            "cleaning_profile_id": canonical_by_doc[d.id].cleaning_profile_id,
                        }
                        if d.id in canonical_by_doc
                        else None
                    ),
                }
                for d in documents
            ],
            "codebooks": [
                {"id": c.id, "name": c.name, "version": c.version, "is_frozen": c.is_frozen}
                for c in codebooks
            ],
            "cleaning_profiles": [
                {
                    "id": p.id,
                    "name": p.name,
                    "version": p.version,
                    "config": loads(p.config_json, {}),
                }
                for p in cleaning_profiles
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
                        "provenance": (loads(r.parameters_json, {}) or {}).get("provenance"),
                    }
                    for r in runs
                ],
            },
        }

    async def export_run_json(self, run_id: str, *, user_id: str) -> dict[str, Any]:
        run = await self.get_run_or_404(run_id, user_id=user_id)
        parameters = loads(run.parameters_json, {})
        return {
            "id": run.id,
            "project_id": run.project_id,
            "corpus_id": run.corpus_id,
            "run_type": run.run_type,
            "status": run.status,
            "progress_stage": run.progress_stage,
            "parameters": parameters,
            "metrics": loads(run.metrics_json, {}),
            "results": loads(run.results_json, {}),
            "artifact_path": run.artifact_path,
            "random_seed": run.random_seed,
            "created_by": run.created_by,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "error_message": run.error_message,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "provenance": parameters.get("provenance"),
            "reproduce": extract_reproduce_request(
                parameters, run_type=run.run_type, run_id=run.id
            ),
            "runtime": runtime_environment(),
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
