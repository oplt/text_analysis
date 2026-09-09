"""Run ingestion quality diagnostics over a research corpus.

Never mutates canonical or raw extracts. Results are persisted as an
``AnalysisRun`` (type ``ingestion_qa``) for inspection.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads
from backend.modules.text_research.infrastructure.ingestion_qa import (
    analyze_document_text,
    detect_exact_duplicates,
    detect_near_duplicates,
    detect_token_count_outliers,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class IngestionQaService(ResearchAccessMixin):
    async def run_corpus_qa(self, corpus_id: str, *, user_id: str) -> AnalysisRun:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        documents = await self.repo.list_documents(corpus_id)
        canonical_rows = await self.repo.list_canonical_sources_for_corpus(corpus_id)
        canonical_by_doc = {row.corpus_document_id: row for row in canonical_rows}

        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus_id,
                run_type=AnalysisRunType.INGESTION_QA.value,
                status=AnalysisRunStatus.RUNNING.value,
                progress_stage="scanning",
                parameters_json=dumps(
                    {
                        "document_count": len(documents),
                        "mutates_text": False,
                        "representations_policy": "inspect_only",
                    }
                ),
                created_by=user_id,
                started_at=_utcnow(),
            )
        )
        await self.db.commit()

        try:
            rag_repo = RagRepository(self.db)
            per_document: list[dict[str, Any]] = []
            corpus_findings: list[dict[str, Any]] = []
            checksum_docs: list[dict[str, Any]] = []
            near_docs: list[dict[str, Any]] = []
            token_docs: list[dict[str, Any]] = []

            for document in documents:
                canonical = canonical_by_doc.get(document.id)
                rag_document = await rag_repo.get_document(document.rag_document_id)
                filename = (
                    (canonical.source_filename if canonical else None)
                    or (rag_document.original_filename if rag_document else None)
                )
                content_type = rag_document.content_type if rag_document else None

                if canonical is None:
                    findings = [
                        {
                            "code": "extraction_failure",
                            "severity": "error",
                            "message": "No canonical research source — extraction/persist incomplete.",
                            "details": {},
                        }
                    ]
                    metrics = {
                        "char_count": 0,
                        "token_count": 0,
                        "text_mutated": False,
                        "has_canonical": False,
                    }
                    text = ""
                    raw_checksum = None
                    canonical_checksum = None
                    transformation = {}
                else:
                    text = canonical.canonical_text
                    raw_text = canonical.raw_extracted_text or canonical.canonical_text
                    transformation = loads(canonical.transformation_metadata_json, {}) or {}
                    findings_objs, metrics = analyze_document_text(
                        text,
                        language=document.language or canonical.language,
                        filename=filename,
                        content_type=content_type,
                        has_extraction_failure=False,
                    )
                    findings = [f.to_dict() for f in findings_objs]
                    raw_checksum = (
                        canonical.raw_extracted_checksum
                        or metrics["text_checksum"]
                    )
                    metrics = {
                        **metrics,
                        "has_canonical": True,
                        "raw_extracted_checksum": raw_checksum,
                        "canonical_checksum": canonical.canonical_text_checksum,
                        "original_file_checksum": canonical.original_file_checksum,
                        "parser_name": canonical.parser_name,
                        "parser_version": canonical.parser_version,
                        "raw_equals_canonical": raw_text == text,
                        "cleaning_profile_id": canonical.cleaning_profile_id,
                        "transformation_metadata": transformation,
                    }
                    canonical_checksum = canonical.canonical_text_checksum
                    checksum_docs.append(
                        {"document_id": document.id, "checksum": canonical.canonical_text_checksum}
                    )
                    near_docs.append({"document_id": document.id, "text": text})
                    token_docs.append(
                        {"document_id": document.id, "token_count": metrics["token_count"]}
                    )

                per_document.append(
                    {
                        "document_id": document.id,
                        "title": document.title,
                        "language": document.language,
                        "findings": findings,
                        "metrics": metrics,
                        "representations_unchanged": True,
                        "representations_checksum": raw_checksum,
                        "canonical_checksum": canonical_checksum,
                    }
                )

            corpus_findings.extend(f.to_dict() for f in detect_exact_duplicates(checksum_docs))
            corpus_findings.extend(f.to_dict() for f in detect_near_duplicates(near_docs))
            outlier_findings = detect_token_count_outliers(token_docs)
            corpus_findings.extend(f.to_dict() for f in outlier_findings)

            # Attach outlier findings onto matching per-document rows as well.
            outliers_by_doc = {
                f.details.get("document_id"): f.to_dict()
                for f in outlier_findings
                if f.details.get("document_id")
            }
            for row in per_document:
                extra = outliers_by_doc.get(row["document_id"])
                if extra:
                    row["findings"].append(extra)

            severity_counts = {"error": 0, "warning": 0, "info": 0}
            for row in per_document:
                for finding in row["findings"]:
                    severity_counts[finding.get("severity", "info")] = (
                        severity_counts.get(finding.get("severity", "info"), 0) + 1
                    )
            for finding in corpus_findings:
                severity_counts[finding.get("severity", "info")] = (
                    severity_counts.get(finding.get("severity", "info"), 0) + 1
                )

            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.COMPLETED.value,
                progress_stage="completed",
                completed_at=_utcnow(),
                metrics_json=dumps(
                    {
                        "documents_scanned": len(documents),
                        "documents_with_canonical": len(canonical_by_doc),
                        "finding_counts": severity_counts,
                        "text_mutated": False,
                    }
                ),
                results_json=dumps(
                    {
                        "policy": {
                            "mutates_text": False,
                            "raw_representation": "CanonicalResearchSource.raw_extracted_text",
                            "cleaned_representation": "CanonicalResearchSource.canonical_text",
                            "transformation_metadata": "CanonicalResearchSource.transformation_metadata_json",
                        },
                        "corpus_findings": corpus_findings,
                        "documents": per_document,
                    }
                ),
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

        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def get_document_qa_slice(
        self, document_id: str, *, user_id: str, run_id: str | None = None
    ) -> dict[str, Any]:
        document, corpus = await self.get_document_or_404(document_id, user_id=user_id)
        if run_id:
            run = await self.repo.get_run(run_id)
            if run is None or run.corpus_id != corpus.id:
                raise HTTPException(status_code=404, detail="Ingestion QA run not found")
        else:
            runs, _ = await self.repo.list_runs(
                corpus.project_id,
                corpus_id=corpus.id,
                run_type=AnalysisRunType.INGESTION_QA.value,
                limit=1,
                offset=0,
            )
            run = runs[0] if runs else None
            if run is None:
                raise HTTPException(
                    status_code=404,
                    detail="No ingestion QA run yet. POST /corpora/{id}/ingestion-qa first.",
                )

        results = loads(run.results_json, {}) or {}
        for row in results.get("documents") or []:
            if row.get("document_id") == document.id:
                return {
                    "run_id": run.id,
                    "document_id": document.id,
                    "title": document.title,
                    **row,
                }
        raise HTTPException(status_code=404, detail="Document not present in QA run results")
