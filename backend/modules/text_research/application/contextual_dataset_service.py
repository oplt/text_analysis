"""Contextual / mixed-method datasets: country–year indicators linked
exploratorily to discourse prevalence.

These joins are descriptive only. They do not claim causal effects.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.comparative_analysis_service import (
    ComparativeAnalysisService,
)
from backend.modules.text_research.domain.models import ContextualObservation, dumps, loads


def _parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "null", "."}:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


class ContextualDatasetService(ResearchAccessMixin):
    async def create_dataset(
        self,
        *,
        project_id: str,
        user_id: str,
        name: str,
        description: str | None = None,
    ):
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        dataset = await self.repo.create_contextual_dataset(
            project_id=project_id,
            name=name.strip(),
            description=description,
            created_by=user_id,
        )
        await self.db.commit()
        return dataset

    async def list_datasets(self, *, project_id: str, user_id: str) -> list[dict[str, Any]]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        datasets = await self.repo.list_contextual_datasets_with_counts(project_id)
        return [
            {
                "id": dataset.id,
                "project_id": dataset.project_id,
                "name": dataset.name,
                "description": dataset.description,
                "created_by": dataset.created_by,
                "created_at": dataset.created_at,
                "observation_count": count,
            }
            for dataset, count in datasets
        ]

    async def get_dataset(self, dataset_id: str, *, user_id: str) -> dict[str, Any]:
        dataset = await self.get_contextual_dataset_or_404(dataset_id, user_id=user_id)
        observations = await self.repo.list_observations(dataset_id)
        indicator_keys: set[str] = set()
        for row in observations:
            values = loads(row.values_json, {}) or {}
            if isinstance(values, dict):
                indicator_keys.update(str(key) for key in values)
        return {
            "id": dataset.id,
            "project_id": dataset.project_id,
            "name": dataset.name,
            "description": dataset.description,
            "created_by": dataset.created_by,
            "created_at": dataset.created_at,
            "observation_count": len(observations),
            "indicator_keys": sorted(indicator_keys),
        }

    async def list_observations(
        self, dataset_id: str, *, user_id: str, limit: int, offset: int
    ) -> dict[str, Any]:
        await self.get_contextual_dataset_or_404(dataset_id, user_id=user_id)
        rows = await self.repo.list_observations_page(dataset_id, limit=limit, offset=offset)
        total = await self.repo.count_observations(dataset_id)
        return {
            "items": [
                {
                    "id": row.id,
                    "country": row.country,
                    "year": row.year,
                    "values": loads(row.values_json, {}) or {},
                    "created_at": row.created_at,
                }
                for row in rows
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def delete_dataset(self, dataset_id: str, *, user_id: str) -> None:
        dataset = await self.get_contextual_dataset_or_404(dataset_id, user_id=user_id)
        await self.repo.delete_contextual_dataset(dataset)
        await self.db.commit()

    async def import_csv(
        self,
        dataset_id: str,
        *,
        user_id: str,
        csv_text: str | None = None,
        reader: csv.DictReader | None = None,
        replace_existing: bool = True,
    ) -> dict[str, Any]:
        dataset = await self.get_contextual_dataset_or_404(dataset_id, user_id=user_id)
        if reader is None:
            if csv_text is None:
                raise ValueError("csv_text or reader is required")
            reader = csv.DictReader(io.StringIO(csv_text))
        if not reader.fieldnames:
            raise HTTPException(status_code=422, detail="CSV must include a header row.")

        normalized = {name.strip().lower(): name for name in reader.fieldnames if name}
        country_col = next(
            (normalized[key] for key in ("country", "nation", "iso3", "iso_code") if key in normalized),
            None,
        )
        year_col = next(
            (normalized[key] for key in ("year", "publication_year") if key in normalized),
            None,
        )
        if country_col is None and year_col is None:
            raise HTTPException(
                status_code=422,
                detail="CSV must include at least a country or year column.",
            )

        reserved = {country_col, year_col}
        value_cols = [name for name in reader.fieldnames if name and name not in reserved]
        if not value_cols:
            raise HTTPException(
                status_code=422,
                detail="CSV must include at least one indicator value column.",
            )

        rows: list[ContextualObservation] = []
        skipped = 0
        for record in reader:
            country = (record.get(country_col) or "").strip() if country_col else None
            year_raw = (record.get(year_col) or "").strip() if year_col else ""
            year: int | None = None
            if year_raw:
                try:
                    year = int(float(year_raw))
                except ValueError:
                    skipped += 1
                    continue
            values: dict[str, float] = {}
            for col in value_cols:
                parsed = _parse_numeric(record.get(col))
                if parsed is not None:
                    values[col.strip()] = parsed
            if not values:
                skipped += 1
                continue
            if not country and year is None:
                skipped += 1
                continue
            rows.append(
                ContextualObservation(
                    dataset_id=dataset.id,
                    country=country or None,
                    year=year,
                    values_json=dumps(values),
                )
            )

        if not rows:
            raise HTTPException(
                status_code=422,
                detail="No usable observations found in CSV after parsing.",
            )

        if replace_existing:
            created = await self.repo.replace_observations(dataset.id, rows)
        else:
            created = await self.repo.bulk_create_observations(rows)
        await self.db.commit()
        return {
            "dataset_id": dataset.id,
            "imported": len(created),
            "skipped": skipped,
            "indicator_keys": value_cols,
            "replaced": replace_existing,
        }

    async def link_discourse(
        self,
        dataset_id: str,
        *,
        user_id: str,
        corpus_id: str,
        codebook_id: str,
        label_ids: list[str],
        indicator_key: str,
        unit_type: str = "paragraph",
        group_by: str = "country",
        join_on_year: bool = True,
        provenance_mode: str = "human_only",
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Join discourse prevalence with a contextual indicator (exploratory)."""
        if group_by not in {"country", "publication_year"}:
            raise HTTPException(
                status_code=400,
                detail="group_by must be 'country' or 'publication_year' for contextual joins.",
            )
        dataset = await self.get_contextual_dataset_or_404(dataset_id, user_id=user_id)
        observations = await self.repo.list_observations(dataset_id)
        if not observations:
            raise HTTPException(status_code=422, detail="Contextual dataset has no observations.")

        run = await ComparativeAnalysisService(self.db).prevalence_by_metadata(
            corpus_id,
            user_id=user_id,
            unit_type=unit_type,
            codebook_id=codebook_id,
            label_ids=label_ids,
            group_by=group_by,
            provenance_mode=provenance_mode,
            model_id=model_id,
        )
        results = loads(run.results_json, {}) or {}
        prevalence = results.get("prevalence") or {}

        # Index contextual values by (country, year) and by country-only / year-only.
        by_country_year: dict[tuple[str, int], float] = {}
        by_country: dict[str, list[float]] = {}
        by_year: dict[int, list[float]] = {}
        for row in observations:
            values = loads(row.values_json, {}) or {}
            if not isinstance(values, dict) or indicator_key not in values:
                continue
            numeric = _parse_numeric(values.get(indicator_key))
            if numeric is None:
                continue
            country = (row.country or "").strip()
            if country and row.year is not None:
                by_country_year[(country.lower(), int(row.year))] = numeric
            if country:
                by_country.setdefault(country.lower(), []).append(numeric)
            if row.year is not None:
                by_year.setdefault(int(row.year), []).append(numeric)

        points: list[dict[str, Any]] = []
        unmatched = 0
        for label_name, groups in prevalence.items():
            if not isinstance(groups, dict):
                continue
            for group_key, stats in groups.items():
                if not isinstance(stats, dict):
                    continue
                indicator_value: float | None = None
                if group_by == "country":
                    key = str(group_key).strip().lower()
                    if join_on_year:
                        # Prefer exact country mean when year isn't on discourse axis.
                        values = by_country.get(key) or []
                        indicator_value = sum(values) / len(values) if values else None
                    else:
                        values = by_country.get(key) or []
                        indicator_value = sum(values) / len(values) if values else None
                else:
                    try:
                        year = int(group_key)
                    except (TypeError, ValueError):
                        unmatched += 1
                        continue
                    values = by_year.get(year) or []
                    indicator_value = sum(values) / len(values) if values else None

                if indicator_value is None:
                    unmatched += 1
                    continue
                points.append(
                    {
                        "label": label_name,
                        "group": str(group_key),
                        "group_by": group_by,
                        "prevalence": float(stats.get("prevalence") or 0.0),
                        "yes": int(stats.get("yes") or 0),
                        "total": int(stats.get("total") or 0),
                        "indicator_key": indicator_key,
                        "indicator_value": indicator_value,
                    }
                )

        # Pearson correlation per label when enough points exist.
        correlations: dict[str, dict[str, Any]] = {}
        by_label: dict[str, list[dict[str, Any]]] = {}
        for point in points:
            by_label.setdefault(point["label"], []).append(point)
        for label_name, label_points in by_label.items():
            if len(label_points) < 3:
                correlations[label_name] = {
                    "n": len(label_points),
                    "pearson_r": None,
                    "status": "not_enough_points",
                }
                continue
            xs = [p["indicator_value"] for p in label_points]
            ys = [p["prevalence"] for p in label_points]
            correlations[label_name] = {
                "n": len(label_points),
                "pearson_r": _pearson(xs, ys),
                "status": "exploratory",
            }

        return {
            "exploratory": True,
            "disclaimer": (
                "Joined discourse prevalence with contextual indicators for exploration only. "
                "Do not interpret associations as causal effects."
            ),
            "dataset": {"id": dataset.id, "name": dataset.name},
            "indicator_key": indicator_key,
            "group_by": group_by,
            "prevalence_run_id": run.id,
            "point_count": len(points),
            "unmatched_groups": unmatched,
            "points": points,
            "correlations": correlations,
        }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)
