"""Inter-coder reliability computation, persisted as an `AnalysisRun`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps
from backend.modules.text_research.infrastructure.reliability import (
    coder_pair_agreement_matrix,
    cohens_kappa,
    disagreement_units,
    krippendorff_alpha_nominal,
    raw_agreement,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _build_value_matrix(values_by_coder: dict[str, dict[str, str]]) -> list[list[str | None]]:
    coders = sorted(values_by_coder)
    unit_ids = sorted({unit for per_unit in values_by_coder.values() for unit in per_unit})
    return [[values_by_coder[coder].get(unit) for coder in coders] for unit in unit_ids]


class ReliabilityService(ResearchAccessMixin):
    async def compute_reliability(
        self,
        corpus_id: str,
        *,
        user_id: str,
        codebook_id: str,
        label_ids: list[str] | None = None,
    ) -> AnalysisRun:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)

        labels = await self.repo.list_labels(codebook_id)
        if label_ids:
            labels = [label for label in labels if label.id in label_ids]

        annotations = await self.repo.list_annotations_for_corpus(corpus_id)
        annotations = [a for a in annotations if a.codebook_version == codebook.version]

        by_label: dict[str, dict[str, dict[str, str]]] = {}
        for annotation in annotations:
            by_label.setdefault(annotation.label_id, {}).setdefault(annotation.annotator_id, {})[
                annotation.text_unit_id
            ] = annotation.value

        results_by_label: dict[str, Any] = {}
        for label in labels:
            values_by_coder = by_label.get(label.id, {})
            coders = sorted(values_by_coder)

            raw: dict[str, Any] | None = None
            kappa: dict[str, Any] | None = None
            if len(coders) == 2:
                shared = sorted(set(values_by_coder[coders[0]]) & set(values_by_coder[coders[1]]))
                values_a = [values_by_coder[coders[0]][unit] for unit in shared]
                values_b = [values_by_coder[coders[1]][unit] for unit in shared]
                raw = raw_agreement(values_a, values_b)
                kappa = cohens_kappa(values_a, values_b)

            value_matrix = _build_value_matrix(values_by_coder)
            alpha = krippendorff_alpha_nominal(value_matrix)
            pair_agreement = coder_pair_agreement_matrix(values_by_coder)
            disagreements = disagreement_units(values_by_coder)

            results_by_label[label.name] = {
                "label_id": label.id,
                "n_coders": len(coders),
                "raw_agreement": raw,
                "cohens_kappa": kappa,
                "krippendorff_alpha": alpha,
                "coder_pair_agreement": pair_agreement,
                "disagreement_count": len(disagreements),
                "disagreements": disagreements,
            }

        alphas = [
            row["krippendorff_alpha"]["alpha"]
            for row in results_by_label.values()
            if row["krippendorff_alpha"]["alpha"] is not None
        ]
        kappas = [
            row["cohens_kappa"]["kappa"]
            for row in results_by_label.values()
            if row["cohens_kappa"] and row["cohens_kappa"]["kappa"] is not None
        ]
        summary = {
            "mean_krippendorff_alpha": sum(alphas) / len(alphas) if alphas else None,
            "mean_cohens_kappa": sum(kappas) / len(kappas) if kappas else None,
            "labels_evaluated": len(results_by_label),
        }

        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus_id,
                run_type=AnalysisRunType.RELIABILITY.value,
                status=AnalysisRunStatus.COMPLETED.value,
                parameters_json=dumps(
                    {
                        "codebook_id": codebook_id,
                        "codebook_version": codebook.version,
                        "label_ids": [label.id for label in labels],
                    }
                ),
                metrics_json=dumps(summary),
                results_json=dumps({"by_label": results_by_label}),
                created_by=user_id,
                started_at=_utcnow(),
                completed_at=_utcnow(),
            )
        )
        await self.db.commit()
        return run
