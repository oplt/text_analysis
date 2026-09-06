"""Cross-organization / cross-region prevalence comparison ("Western bias
explorer" support), with explicit provenance tracking: results can be
computed from human annotations only, model predictions only, or a
human-preferred blend (human value used when present, model value as
fallback) — the provenance mode and per-unit source counts are always
persisted alongside the numbers so findings are never mistaken for ground
truth when they are actually model output.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.quantitative_analysis_service import (
    _filter_kwargs,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType, ProvenanceMode
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads


def _utcnow() -> datetime:
    return datetime.now(UTC)

_ALLOWED_GROUP_BY = {
    "organization",
    "organization_type",
    "region",
    "cultural_sphere",
    "language",
    "publication_year",
    "publication_type",
    "country",
}


class ComparativeAnalysisService(ResearchAccessMixin):
    async def prevalence_by_metadata(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        codebook_id: str,
        label_ids: list[str],
        group_by: str,
        provenance_mode: str = ProvenanceMode.HUMAN_ONLY.value,
        model_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        if group_by not in _ALLOWED_GROUP_BY:
            raise HTTPException(status_code=400, detail=f"Unsupported group_by: {group_by}")

        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)
        if not label_ids:
            raise HTTPException(status_code=400, detail="At least one label is required")

        filter_kwargs = _filter_kwargs(filters or {})
        documents = await self.repo.list_documents(corpus_id, **filter_kwargs)
        doc_ids = [document.id for document in documents]
        doc_lookup = {document.id: document for document in documents}
        units = await self.repo.list_text_units_for_corpus(
            corpus_id, unit_type=unit_type, document_ids=doc_ids if filter_kwargs else None
        )
        if not units:
            raise HTTPException(
                status_code=422, detail="No text units match the requested corpus/unit_type/filters"
            )
        unit_ids = [u.id for u in units]
        unit_id_set = set(unit_ids)

        # --- Human-sourced values (adjudicated > majority vote across annotators) ---
        human_values: dict[str, dict[str, str]] = {}
        if provenance_mode != ProvenanceMode.MODEL_ONLY.value:
            annotations = await self.repo.list_annotations_for_units(unit_ids)
            adjudications = await self.repo.list_adjudications_for_units(unit_ids)
            adjudication_lookup = {(a.text_unit_id, a.label_id): a.final_value for a in adjudications}
            by_label_unit: dict[str, dict[str, list[str]]] = {}
            for annotation in annotations:
                if annotation.codebook_version != codebook.version or annotation.label_id not in label_ids:
                    continue
                by_label_unit.setdefault(annotation.label_id, {}).setdefault(
                    annotation.text_unit_id, []
                ).append(annotation.value)
            for label_id, per_unit in by_label_unit.items():
                for unit_id, values in per_unit.items():
                    adjudicated = adjudication_lookup.get((unit_id, label_id))
                    if adjudicated is not None:
                        human_values.setdefault(label_id, {})[unit_id] = adjudicated
                    else:
                        top_value, _count = Counter(values).most_common(1)[0]
                        human_values.setdefault(label_id, {})[unit_id] = top_value

        # --- Model-sourced values ---
        model_values: dict[str, dict[str, str]] = {}
        model_uncertainty: dict[str, float] = {}
        if provenance_mode != ProvenanceMode.HUMAN_ONLY.value:
            if not model_id:
                raise HTTPException(
                    status_code=400,
                    detail="model_id is required when provenance_mode is model_only or human_preferred",
                )
            model = await self.get_model_or_404(model_id, user_id=user_id)
            label_names = loads(model.label_ids_json, [])
            codebook_labels = await self.repo.list_labels(codebook_id)
            name_to_id = {label.name: label.id for label in codebook_labels}
            predictions, _total = await self.repo.list_predictions_for_model(
                model.id, limit=max(len(unit_ids), 1), offset=0
            )
            for prediction in predictions:
                if prediction.text_unit_id not in unit_id_set:
                    continue
                if prediction.uncertainty is not None:
                    model_uncertainty[prediction.text_unit_id] = prediction.uncertainty
                predicted_labels = set(loads(prediction.predicted_labels_json, []))
                for label_name in label_names:
                    label_id = name_to_id.get(label_name)
                    if label_id is None or label_id not in label_ids:
                        continue
                    value = "yes" if label_name in predicted_labels else "no"
                    model_values.setdefault(label_id, {})[prediction.text_unit_id] = value

        def resolve(label_id: str, unit_id: str) -> tuple[str | None, str]:
            human = human_values.get(label_id, {}).get(unit_id)
            model_value = model_values.get(label_id, {}).get(unit_id)
            if provenance_mode == ProvenanceMode.MODEL_ONLY.value:
                return model_value, "model" if model_value is not None else "missing"
            if provenance_mode == ProvenanceMode.HUMAN_ONLY.value:
                return human, "human" if human is not None else "missing"
            # human_preferred
            if human is not None:
                return human, "human"
            if model_value is not None:
                return model_value, "model"
            return None, "missing"

        codebook_labels = await self.repo.list_labels(codebook_id)
        label_name_by_id = {label.id: label.name for label in codebook_labels}

        prevalence: dict[str, dict[str, dict[str, Any]]] = {}
        examples: dict[str, dict[str, list[dict[str, Any]]]] = {}
        provenance_counts = {"human": 0, "model": 0, "missing": 0}
        for label_id in label_ids:
            label_name = label_name_by_id.get(label_id, label_id)
            group_tally: dict[str, dict[str, Any]] = {}
            group_examples: dict[str, list[dict[str, Any]]] = {}
            for unit in units:
                doc = doc_lookup.get(unit.corpus_document_id)
                key = str(getattr(doc, group_by, None) or "unspecified") if doc else "unspecified"
                value, source = resolve(label_id, unit.id)
                provenance_counts[source] += 1
                bucket = group_tally.setdefault(
                    key,
                    {
                        "total": 0,
                        "yes": 0,
                        "document_ids": set(),
                        "provenance_counts": Counter(),
                        "uncertainties": [],
                        "temporal_distribution": Counter(),
                    },
                )
                bucket["total"] += 1
                bucket["document_ids"].add(unit.corpus_document_id)
                bucket["provenance_counts"][source] += 1
                if source == "model" and unit.id in model_uncertainty:
                    bucket["uncertainties"].append(model_uncertainty[unit.id])
                if doc and doc.publication_year is not None:
                    bucket["temporal_distribution"][str(doc.publication_year)] += 1
                if value == "yes":
                    bucket["yes"] += 1
                    samples = group_examples.setdefault(key, [])
                    if len(samples) < 5:
                        samples.append(
                            {
                                "text_unit_id": unit.id,
                                "text": unit.text[:500],
                                "document_id": unit.corpus_document_id,
                                "document_title": doc.title if doc else None,
                                "organization": doc.organization if doc else None,
                                "publication_year": doc.publication_year if doc else None,
                                "country": doc.country if doc else None,
                                "provenance": source,
                                "uncertainty": model_uncertainty.get(unit.id)
                                if source == "model"
                                else None,
                                "document_metadata": {
                                    "organization_type": doc.organization_type if doc else None,
                                    "region": doc.region if doc else None,
                                    "cultural_sphere": doc.cultural_sphere if doc else None,
                                    "language": doc.language if doc else None,
                                    "publication_type": doc.publication_type if doc else None,
                                    "source_url": doc.source_url if doc else None,
                                },
                            }
                        )
            prevalence[label_name] = {
                key: {
                    "total": bucket["total"],
                    "yes": bucket["yes"],
                    "prevalence": bucket["yes"] / bucket["total"] if bucket["total"] else 0.0,
                    "document_count": len(bucket["document_ids"]),
                    "provenance_counts": dict(bucket["provenance_counts"]),
                    "mean_uncertainty": (
                        sum(bucket["uncertainties"]) / len(bucket["uncertainties"])
                        if bucket["uncertainties"]
                        else None
                    ),
                    "temporal_distribution": dict(bucket["temporal_distribution"]),
                }
                for key, bucket in sorted(group_tally.items())
            }
            examples[label_name] = {
                key: group_examples.get(key, []) for key in prevalence[label_name]
            }

        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus_id,
                run_type=AnalysisRunType.COMPARATIVE_ANALYSIS.value,
                status=AnalysisRunStatus.COMPLETED.value,
                parameters_json=dumps(
                    {
                        "unit_type": unit_type,
                        "codebook_id": codebook_id,
                        "label_ids": label_ids,
                        "group_by": group_by,
                        "provenance_mode": provenance_mode,
                        "model_id": model_id,
                        "filters": filters,
                    }
                ),
                metrics_json=dumps({"unit_count": len(units), "provenance_counts": provenance_counts}),
                results_json=dumps({"prevalence": prevalence, "examples": examples}),
                created_by=user_id,
                started_at=_utcnow(),
                completed_at=_utcnow(),
            )
        )
        await self.db.commit()
        return run
