"""Deterministic corpus segmentation into `TextUnit`s, tracked via `AnalysisRun`."""

from __future__ import annotations

from datetime import UTC, datetime

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.analysis_executor import (
    attach_run_identity,
    build_spec_from_request,
)
from backend.modules.text_research.application.corpus_service import CorpusService
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, TextUnit, dumps, loads
from backend.modules.text_research.infrastructure.language_processing import resolve_language
from backend.modules.text_research.infrastructure.segmentation import hash_text, segment_text


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SegmentationService(ResearchAccessMixin):
    async def start_segmentation(
        self, corpus_id: str, *, user_id: str, unit_type: str
    ) -> AnalysisRun:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        documents = await self.repo.list_documents(corpus_id)

        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus_id,
                run_type=AnalysisRunType.SEGMENTATION.value,
                status=AnalysisRunStatus.QUEUED.value,
                progress_stage="queued",
                parameters_json=dumps(
                    attach_run_identity(
                        {
                            "unit_type": unit_type,
                            "document_count": len(documents),
                            "language_aware": True,
                        },
                        build_spec_from_request(
                            "frequencies",
                            corpus_id,
                            unit_type=unit_type,
                            analysis_parameters={"mode": "segmentation"},
                        ),
                    )
                ),
                metrics_json=dumps(
                    {
                        "unit_type": unit_type,
                        "documents_total": len(documents),
                        "documents_segmented": 0,
                        "text_units_created": 0,
                    }
                ),
                created_by=user_id,
            )
        )
        await self.db.commit()

        from backend.modules.text_research.application.execution_service import ExecutionService

        await ExecutionService.submit(
            db=self.db, run=run, operation="segmentation", user_id=user_id
        )

        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def execute_segmentation(self, run_id: str) -> AnalysisRun:
        """Perform the actual segmentation for a queued/running AnalysisRun.

        Called by the background worker using its own DB session.
        """
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        if run.status in {AnalysisRunStatus.COMPLETED.value, AnalysisRunStatus.CANCELLED.value}:
            return run

        params = loads(run.parameters_json, {})
        unit_type = params.get("unit_type")

        documents = await self.repo.list_documents(run.corpus_id)
        document_total = len(documents)

        await self.repo.update_run(
            run,
            status=AnalysisRunStatus.RUNNING.value,
            progress_stage="preparing",
            started_at=_utcnow(),
            metrics_json=dumps(
                {
                    "unit_type": unit_type,
                    "documents_total": document_total,
                    "documents_segmented": 0,
                    "text_units_created": 0,
                }
            ),
        )
        await self.db.commit()

        corpus_service = CorpusService(self.db)
        try:
            total_units = 0
            per_document: list[dict] = []
            for index, document in enumerate(documents, start=1):
                if index == 1:
                    await self.repo.update_run(
                        run,
                        progress_stage="segmenting_documents",
                        metrics_json=dumps(
                            {
                                "unit_type": unit_type,
                                "documents_total": document_total,
                                "documents_segmented": 0,
                                "text_units_created": 0,
                            }
                        ),
                    )
                    await self.db.commit()

                text = await corpus_service.get_source_text(document.id, user_id=run.created_by)
                canonical = await corpus_service.repo.get_canonical_source(document.id)
                page_provenance = (
                    loads(canonical.page_provenance_json, []) if canonical is not None else None
                )
                doc_language = document.language or (canonical.language if canonical else None)
                lang_profile = resolve_language(doc_language)
                await self.repo.delete_text_units_for_document(document.id, unit_type)
                segments = segment_text(
                    text,
                    unit_type,
                    page_provenance=page_provenance,
                    language=doc_language,
                )
                rows = [
                    TextUnit(
                        corpus_document_id=document.id,
                        unit_type=unit_type,
                        position=segment["position"],
                        page_number=segment.get("page_number"),
                        paragraph_number=segment.get("paragraph_number"),
                        sentence_number=segment.get("sentence_number"),
                        char_start=segment.get("char_start"),
                        char_end=segment.get("char_end"),
                        section_heading=segment.get("section_heading"),
                        text=segment["text"],
                        text_hash=segment.get("text_hash") or hash_text(segment["text"]),
                        source_text_hash=segment.get("source_text_hash"),
                    )
                    for segment in segments
                ]
                if rows:
                    await self.repo.bulk_create_text_units(rows)
                total_units += len(rows)
                per_document.append(
                    {
                        "document_id": document.id,
                        "unit_count": len(rows),
                        "language": lang_profile.code,
                        "language_degraded": lang_profile.degraded,
                        "sentence_segmentation": lang_profile.sentence_segmentation,
                    }
                )

                await self.repo.update_run(
                    run,
                    progress_stage="segmenting_documents",
                    metrics_json=dumps(
                        {
                            "unit_type": unit_type,
                            "documents_total": document_total,
                            "documents_segmented": index,
                            "text_units_created": total_units,
                        }
                    ),
                )
                await self.db.commit()

            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.COMPLETED.value,
                progress_stage="completed",
                completed_at=_utcnow(),
                metrics_json=dumps(
                    {
                        "unit_type": unit_type,
                        "documents_total": document_total,
                        "documents_segmented": document_total,
                        "text_units_created": total_units,
                    }
                ),
                results_json=dumps({"per_document": per_document}),
            )
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.FAILED.value,
                progress_stage="failed",
                completed_at=_utcnow(),
                error_message=str(exc),
            )
            await self.db.commit()
            raise

        refreshed = await self.repo.get_run(run_id)
        assert refreshed is not None
        return refreshed
