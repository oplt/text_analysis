"""Dashboard KPI aggregation — reads only already-persisted data (corpus
composition, annotation progress, trained models, analysis run history).
Never retrains or recomputes statistics; that would belong in the dedicated
analysis services.
"""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType, UnitType
from backend.modules.text_research.domain.models import loads


class DashboardService(ResearchAccessMixin):
    async def summary(self, corpus_id: str, *, user_id: str) -> dict[str, Any]:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        documents = await self.repo.list_documents(corpus_id)

        unit_counts = {
            unit_type.value: await self.repo.count_text_units_for_corpus(
                corpus_id, unit_type=unit_type.value
            )
            for unit_type in (UnitType.DOCUMENT, UnitType.PARAGRAPH, UnitType.SENTENCE)
        }

        codebooks = await self.repo.list_codebooks(corpus.project_id)
        models = await self.repo.list_models(corpus.project_id, corpus_id=corpus_id)
        snapshots = await self.repo.list_snapshots(corpus.project_id, corpus_id=corpus_id)
        runs, _total_runs = await self.repo.list_runs(
            corpus.project_id, corpus_id=corpus_id, limit=200, offset=0
        )

        runs_by_type: dict[str, int] = {}
        runs_by_status: dict[str, int] = {}
        latest_reliability_run = None
        for run in runs:
            runs_by_type[run.run_type] = runs_by_type.get(run.run_type, 0) + 1
            runs_by_status[run.status] = runs_by_status.get(run.status, 0) + 1
            if (
                run.run_type == AnalysisRunType.RELIABILITY.value
                and run.status == AnalysisRunStatus.COMPLETED.value
                and (latest_reliability_run is None or run.created_at > latest_reliability_run.created_at)
            ):
                latest_reliability_run = run

        latest_model = max(models, key=lambda m: m.created_at) if models else None

        all_units = await self.repo.list_text_units_for_corpus(corpus_id)
        unit_ids = [u.id for u in all_units]
        tasks = await self.repo.list_tasks_for_units(unit_ids) if unit_ids else []
        completed_tasks = sum(1 for t in tasks if t.status == "completed")

        return {
            "corpus": {"id": corpus.id, "name": corpus.name},
            "document_count": len(documents),
            "text_unit_counts": unit_counts,
            "codebook_count": len(codebooks),
            "training_dataset_snapshot_count": len(snapshots),
            "trained_model_count": len(models),
            "latest_model": (
                {
                    "id": latest_model.id,
                    "name": latest_model.name,
                    "version": latest_model.version,
                    "metrics": loads(latest_model.metrics_json, {}),
                }
                if latest_model
                else None
            ),
            "analysis_run_counts_by_type": runs_by_type,
            "analysis_run_counts_by_status": runs_by_status,
            "latest_reliability": (
                {
                    "run_id": latest_reliability_run.id,
                    "metrics": loads(latest_reliability_run.metrics_json, {}),
                }
                if latest_reliability_run
                else None
            ),
            "annotation_task_count": len(tasks),
            "annotation_completed_count": completed_tasks,
            "annotation_completion_rate": completed_tasks / len(tasks) if tasks else 0.0,
        }
