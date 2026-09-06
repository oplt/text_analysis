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

        document_count = await self.repo.count_documents(corpus_id)
        grouped_unit_counts = await self.repo.count_text_units_grouped_by_type(corpus_id)
        unit_counts = {
            unit_type.value: grouped_unit_counts.get(unit_type.value, 0)
            for unit_type in (UnitType.DOCUMENT, UnitType.PARAGRAPH, UnitType.SENTENCE)
        }

        codebook_count = await self.repo.count_codebooks(corpus.project_id)
        trained_model_count = await self.repo.count_models(corpus.project_id, corpus_id=corpus_id)
        snapshot_count = await self.repo.count_snapshots(corpus.project_id, corpus_id=corpus_id)
        runs_by_type = await self.repo.count_runs_grouped(
            corpus.project_id, corpus_id=corpus_id, group_by="run_type"
        )
        runs_by_status = await self.repo.count_runs_grouped(
            corpus.project_id, corpus_id=corpus_id, group_by="status"
        )
        task_counts = await self.repo.count_annotation_tasks_for_corpus(corpus_id)
        latest_reliability_run = await self.repo.get_latest_run(
            corpus.project_id,
            corpus_id=corpus_id,
            run_type=AnalysisRunType.RELIABILITY.value,
            status=AnalysisRunStatus.COMPLETED.value,
        )
        latest_model = await self.repo.get_latest_model(corpus.project_id, corpus_id=corpus_id)

        annotation_task_count = int(task_counts.get("total", 0))
        completed_tasks = int(task_counts.get("completed", 0))

        return {
            "corpus": {"id": corpus.id, "name": corpus.name},
            "document_count": document_count,
            "text_unit_counts": unit_counts,
            "codebook_count": codebook_count,
            "training_dataset_snapshot_count": snapshot_count,
            "trained_model_count": trained_model_count,
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
            "annotation_task_count": annotation_task_count,
            "annotation_completed_count": completed_tasks,
            "annotation_completion_rate": (
                completed_tasks / annotation_task_count if annotation_task_count else 0.0
            ),
        }
