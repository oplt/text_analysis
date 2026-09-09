"""Inter-coder reliability computation, persisted as an `AnalysisRun`."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads
from backend.modules.text_research.infrastructure.reliability import (
    attach_ci,
    bootstrap_unit_statistic,
    coder_pair_agreement_matrix,
    cohens_kappa,
    disagreement_units,
    fleiss_kappa,
    krippendorff_alpha_nominal,
    raw_agreement,
    reliability_diagnostics,
    reliability_metadata,
)
from backend.modules.text_research.infrastructure.scientific_warnings import (
    annotation_category_prevalence,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _build_value_matrix(
    values_by_coder: dict[str, dict[str, str]],
) -> tuple[list[str], list[list[str | None]]]:
    coders = sorted(values_by_coder)
    unit_ids = sorted({unit for per_unit in values_by_coder.values() for unit in per_unit})
    matrix = [[values_by_coder[coder].get(unit) for coder in coders] for unit in unit_ids]
    return unit_ids, matrix


def _categories_observed(value_matrix: list[list[str | None]]) -> int:
    return len({value for row in value_matrix for value in row if value is not None})


class ReliabilityService(ResearchAccessMixin):
    async def compute_reliability(
        self,
        corpus_id: str,
        *,
        user_id: str,
        codebook_id: str,
        label_ids: list[str] | None = None,
        campaign_id: str | None = None,
        unit_type: str | None = None,
        annotator_ids: list[str] | None = None,
        bootstrap_samples: int = 2000,
        confidence_level: float = 0.95,
        random_seed: int | None = None,
    ) -> AnalysisRun:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)

        campaign = None
        resolved_annotator_ids = annotator_ids
        resolved_unit_type = unit_type
        if campaign_id:
            campaign = await self.repo.get_annotation_campaign(campaign_id)
            if campaign is None:
                raise HTTPException(status_code=404, detail="Annotation campaign not found")
            if campaign.corpus_id != corpus_id:
                raise HTTPException(
                    status_code=400, detail="Campaign does not belong to this corpus"
                )
            await self.ensure_project_access(user_id=user_id, project_id=campaign.project_id)
            resolved_unit_type = resolved_unit_type or campaign.unit_type
            if resolved_annotator_ids is None:
                resolved_annotator_ids = loads(campaign.annotator_ids_json, []) or None
            if campaign.codebook_id and campaign.codebook_id != codebook_id:
                raise HTTPException(
                    status_code=400,
                    detail="codebook_id does not match the campaign codebook",
                )
            campaign_status = getattr(campaign, "status", "released")
            campaign_released = campaign_status == "released" or (
                getattr(campaign, "reveal_after", "campaign_released") == "campaign_completed"
                and campaign_status == "completed"
            )
            if (
                campaign.blind_mode
                and not campaign_released
                and getattr(campaign, "created_by", None) != user_id
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Campaign reliability remains blind until released",
                )

        labels = await self.repo.list_labels(codebook_id)
        if label_ids:
            labels = [label for label in labels if label.id in label_ids]
        label_id_filter = [label.id for label in labels] or None

        seed = 42 if random_seed is None else int(random_seed)
        rows = await self.repo.list_reliability_annotation_rows(
            corpus_id,
            codebook_version=codebook.version,
            campaign_id=campaign_id,
            unit_type=resolved_unit_type,
            annotator_ids=resolved_annotator_ids,
            label_ids=label_id_filter,
        )

        by_label: dict[str, dict[str, dict[str, str]]] = {}
        for text_unit_id, label_id, annotator_id, value in rows:
            by_label.setdefault(label_id, {}).setdefault(annotator_id, {})[text_unit_id] = value

        results_by_label: dict[str, Any] = {}
        for label in labels:
            values_by_coder = by_label.get(label.id, {})
            coders = sorted(values_by_coder)
            unit_ids, value_matrix = _build_value_matrix(values_by_coder)
            pairable_unit_count = sum(
                1 for values in value_matrix if sum(v is not None for v in values) >= 2
            )
            evaluation_messages: list[str] = []

            raw: dict[str, Any] | None = None
            kappa: dict[str, Any] | None = None
            fleiss: dict[str, Any] | None = None
            pairwise_cohen: dict[str, Any] = {}
            statistics_used: list[str] = []

            if len(coders) == 2:
                shared = sorted(set(values_by_coder[coders[0]]) & set(values_by_coder[coders[1]]))
                if shared:
                    values_a = [values_by_coder[coders[0]][unit] for unit in shared]
                    values_b = [values_by_coder[coders[1]][unit] for unit in shared]
                    agreement = raw_agreement(values_a, values_b)
                    kappa = cohens_kappa(values_a, values_b)
                    raw = {"agreement": agreement, "sample_size": len(shared)}
                    statistics_used.extend(["raw_agreement", "cohens_kappa"])

                    shared_pairs = list(zip(values_a, values_b, strict=True))

                    def _raw_stat(sample: list[tuple[str, str]]) -> float:
                        return raw_agreement([a for a, _ in sample], [b for _, b in sample])

                    def _kappa_stat(sample: list[tuple[str, str]]) -> float | None:
                        result = cohens_kappa([a for a, _ in sample], [b for _, b in sample])
                        return result.get("kappa")

                    raw = attach_ci(
                        raw,
                        bootstrap_unit_statistic(
                            shared_pairs,
                            _raw_stat,
                            bootstrap_samples=bootstrap_samples,
                            confidence_level=confidence_level,
                            random_seed=seed,
                        ),
                    )
                    kappa = attach_ci(
                        kappa,
                        bootstrap_unit_statistic(
                            shared_pairs,
                            _kappa_stat,
                            bootstrap_samples=bootstrap_samples,
                            confidence_level=confidence_level,
                            random_seed=seed + 1,
                        ),
                    )
                else:
                    evaluation_messages.append(
                        "Cohen's κ is not evaluable: no overlapping annotated units between coders."
                    )
            elif len(coders) < 2:
                evaluation_messages.append(
                    "Reliability is not evaluable: fewer than two coders contributed annotations."
                )
            else:
                evaluation_messages.append(
                    "Cohen's κ is not reported because this label has more than two coders; "
                    "see Fleiss' κ and Krippendorff's α instead."
                )
                fleiss = fleiss_kappa(value_matrix)
                if fleiss.get("kappa") is not None:
                    statistics_used.append("fleiss_kappa")

                    def _fleiss_stat(sample: list[list[str | None]]) -> float | None:
                        result = fleiss_kappa(sample)
                        return result.get("kappa")

                    fleiss = attach_ci(
                        fleiss,
                        bootstrap_unit_statistic(
                            value_matrix,
                            _fleiss_stat,
                            bootstrap_samples=bootstrap_samples,
                            confidence_level=confidence_level,
                            random_seed=seed + 2,
                        ),
                    )
                else:
                    reason = fleiss.get("reason", "Fleiss' κ is not evaluable for this label.")
                    evaluation_messages.append(f"Fleiss' κ is not evaluable: {reason}")

                # Optional pairwise Cohen (not a single overall Cohen for 3+).
                pairwise_cohen: dict[str, Any] = {}
                for i, coder_a in enumerate(coders):
                    for coder_b in coders[i + 1 :]:
                        shared = sorted(
                            set(values_by_coder[coder_a]) & set(values_by_coder[coder_b])
                        )
                        if not shared:
                            continue
                        pair_kappa = cohens_kappa(
                            [values_by_coder[coder_a][u] for u in shared],
                            [values_by_coder[coder_b][u] for u in shared],
                        )
                        pairs = [
                            (values_by_coder[coder_a][unit], values_by_coder[coder_b][unit])
                            for unit in shared
                        ]

                        def _pair_stat(sample: list[tuple[str, str]]) -> float | None:
                            return cohens_kappa(
                                [left for left, _ in sample],
                                [right for _, right in sample],
                            ).get("kappa")

                        pair_kappa = attach_ci(
                            pair_kappa,
                            bootstrap_unit_statistic(
                                pairs,
                                _pair_stat,
                                bootstrap_samples=bootstrap_samples,
                                confidence_level=confidence_level,
                                random_seed=seed + 10 + i,
                            ),
                        )
                        pairwise_cohen[f"{coder_a}|{coder_b}"] = pair_kappa

            alpha = krippendorff_alpha_nominal(value_matrix)
            if pairable_unit_count == 0:
                evaluation_messages.append(
                    "Krippendorff's α is not evaluable: no unit has annotations from >=2 coders."
                )
            elif alpha.get("alpha") is not None:
                statistics_used.append("krippendorff_alpha")

                def _alpha_stat(sample: list[list[str | None]]) -> float | None:
                    result = krippendorff_alpha_nominal(sample)
                    return result.get("alpha")

                alpha = attach_ci(
                    alpha,
                    bootstrap_unit_statistic(
                        value_matrix,
                        _alpha_stat,
                        bootstrap_samples=bootstrap_samples,
                        confidence_level=confidence_level,
                        random_seed=seed + 3,
                    ),
                )

            pair_agreement = coder_pair_agreement_matrix(values_by_coder)
            disagreements = disagreement_units(values_by_coder)
            metadata = reliability_metadata(
                value_matrix,
                scale="nominal",
                statistics_used=statistics_used,
            )
            diagnostics = reliability_diagnostics(
                n_coders=len(coders),
                pairable_unit_count=pairable_unit_count,
                missingness=alpha.get("missingness") if alpha else None,
                categories_observed=_categories_observed(value_matrix),
                fleiss=fleiss,
                kappa=kappa,
                alpha=alpha,
                class_prevalence=annotation_category_prevalence(value_matrix),
            )
            evaluation_messages.extend(msg for msg in diagnostics if msg not in evaluation_messages)

            primary_metric = None
            primary_ci = None
            if len(coders) == 2 and kappa and kappa.get("kappa") is not None:
                primary_metric = {"name": "cohens_kappa", "value": kappa["kappa"]}
                primary_ci = kappa.get("ci")
            elif fleiss and fleiss.get("kappa") is not None:
                primary_metric = {"name": "fleiss_kappa", "value": fleiss["kappa"]}
                primary_ci = fleiss.get("ci")
            elif alpha and alpha.get("alpha") is not None:
                primary_metric = {"name": "krippendorff_alpha", "value": alpha["alpha"]}
                primary_ci = alpha.get("ci")

            prevalence = annotation_category_prevalence(value_matrix)
            ci_payload = primary_ci if isinstance(primary_ci, dict) else None
            results_by_label[label.name] = {
                "label_id": label.id,
                "n_coders": len(coders),
                "n_pairable_units": pairable_unit_count,
                "n_units": len(unit_ids),
                "raw_agreement": raw,
                "cohens_kappa": kappa,
                "fleiss_kappa": fleiss,
                "krippendorff_alpha": alpha,
                "pairwise_cohens_kappa": pairwise_cohen or None,
                "coder_pair_agreement": pair_agreement,
                "disagreement_count": len(disagreements),
                "disagreements": disagreements,
                "evaluation_messages": evaluation_messages,
                "diagnostics": diagnostics,
                "scientific_warnings": diagnostics,
                "class_prevalence": prevalence,
                "primary_metric": primary_metric,
                "primary_ci": primary_ci,
                "primary_ci_width": (
                    (float(ci_payload["upper"]) - float(ci_payload["lower"]))
                    if ci_payload
                    and ci_payload.get("upper") is not None
                    and ci_payload.get("lower") is not None
                    else None
                ),
                "missingness": alpha.get("missingness") if alpha else None,
                "metadata": metadata,
            }

        alphas = [
            row["krippendorff_alpha"]["alpha"]
            for row in results_by_label.values()
            if row["krippendorff_alpha"] and row["krippendorff_alpha"]["alpha"] is not None
        ]
        kappas = [
            row["cohens_kappa"]["kappa"]
            for row in results_by_label.values()
            if row["cohens_kappa"] and row["cohens_kappa"]["kappa"] is not None
        ]
        fleiss_kappas = [
            row["fleiss_kappa"]["kappa"]
            for row in results_by_label.values()
            if row["fleiss_kappa"] and row["fleiss_kappa"]["kappa"] is not None
        ]
        summary = {
            "mean_krippendorff_alpha": sum(alphas) / len(alphas) if alphas else None,
            "mean_cohens_kappa": sum(kappas) / len(kappas) if kappas else None,
            "mean_fleiss_kappa": sum(fleiss_kappas) / len(fleiss_kappas) if fleiss_kappas else None,
            "labels_evaluated": len(results_by_label),
            "scale": "nominal",
            "campaign_id": campaign_id,
            "bootstrap": {
                "bootstrap_samples": bootstrap_samples,
                "confidence_level": confidence_level,
                "random_seed": seed,
            },
            "scientific_warnings": [
                warning
                for row in results_by_label.values()
                for warning in (row.get("scientific_warnings") or [])
            ],
            "labels_with_warnings": sum(
                1 for row in results_by_label.values() if row.get("scientific_warnings")
            ),
        }

        overlap_by_unit: dict[str, set[str]] = {}
        for text_unit_id, _label_id, annotator_id, _value in rows:
            overlap_by_unit.setdefault(str(text_unit_id), set()).add(str(annotator_id))
        overlapping_unit_ids = sorted(
            unit_id for unit_id, coders in overlap_by_unit.items() if len(coders) >= 2
        )
        campaign_snapshot = (
            {
                "id": campaign.id,
                "name": campaign.name,
                "status": getattr(campaign, "status", None),
                "codebook_id": campaign.codebook_id,
                "codebook_version": getattr(campaign, "codebook_version", None),
                "unit_type": campaign.unit_type,
                "annotator_ids": loads(campaign.annotator_ids_json, []) or [],
                "assignment_strategy": getattr(campaign, "assignment_strategy", None),
                "overlap_count": getattr(campaign, "overlap_count", None),
                "overlap_percent": getattr(campaign, "overlap_percent", None),
                "metadata": loads(getattr(campaign, "metadata_json", "{}"), {}) or {},
            }
            if campaign
            else None
        )

        parameters = {
            "codebook_id": codebook_id,
            "codebook_version": codebook.version,
            "label_ids": [label.id for label in labels],
            "campaign_id": campaign_id,
            "unit_type": resolved_unit_type,
            "annotator_ids": resolved_annotator_ids,
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "random_seed": seed,
            "campaign_status": getattr(campaign, "status", None) if campaign else None,
            "campaign_name": campaign.name if campaign else None,
            "campaign_snapshot": campaign_snapshot,
            "campaign_snapshot_hash": (
                sha256(dumps(campaign_snapshot).encode()).hexdigest() if campaign_snapshot else None
            ),
            "overlap_unit_ids": overlapping_unit_ids,
            "overlap_unit_ids_hash": sha256(dumps(overlapping_unit_ids).encode()).hexdigest(),
        }
        from backend.modules.text_research.infrastructure.provenance import attach_provenance

        parameters = attach_provenance(
            parameters,
            random_seed=seed,
            campaign_id=campaign_id,
            codebook_id=codebook_id,
            codebook_version=codebook.version,
            unit_type=resolved_unit_type,
            extra={
                "campaign_name": campaign.name if campaign else None,
                "annotation_mode": campaign.annotation_mode if campaign else None,
                "blind_mode": campaign.blind_mode if campaign else None,
                "campaign_status": getattr(campaign, "status", None) if campaign else None,
                "campaign_snapshot_hash": parameters["campaign_snapshot_hash"],
                "overlap_unit_ids_hash": parameters["overlap_unit_ids_hash"],
                "overlap_unit_count": len(overlapping_unit_ids),
                "bootstrap_samples": bootstrap_samples,
                "confidence_level": confidence_level,
            },
        )

        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus_id,
                run_type=AnalysisRunType.RELIABILITY.value,
                status=AnalysisRunStatus.COMPLETED.value,
                parameters_json=dumps(parameters),
                metrics_json=dumps(summary),
                results_json=dumps({"by_label": results_by_label}),
                random_seed=seed,
                created_by=user_id,
                started_at=_utcnow(),
                completed_at=_utcnow(),
            )
        )
        await self.db.commit()
        return run
