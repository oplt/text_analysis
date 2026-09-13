"""Ingest benchmark files through the production pipeline; bind qrels to runtime IDs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from backend.modules.rag.application.document_ingestion_service import DocumentIngestionService
from backend.modules.rag.eval.end_to_end_eval import EndToEndCase

CORPUS = Path(__file__).with_name("fixtures") / "corpus"


@dataclass(frozen=True)
class FixtureIndex:
    document_ids: dict[str, str]
    chunk_ids: dict[str, str]
    revision_ids: tuple[str, ...]
    source_coordinates: dict[str, dict]

    def bind(self, cases: list[EndToEndCase]) -> list[EndToEndCase]:
        # Missing evidence is an invalid fixture, never a silently empty qrel.
        return [
            replace(
                case,
                expected_document_ids=tuple(
                    self.document_ids[key] for key in case.expected_document_ids
                ),
                expected_chunk_ids=tuple(self.chunk_ids[key] for key in case.expected_chunk_ids),
            )
            for case in cases
        ]


async def ingest_fixture_corpus(
    ingestion: DocumentIngestionService,
    *,
    user_id: str,
    corpus: Path = CORPUS,
) -> FixtureIndex:
    documents: dict[str, str] = {}
    chunks: dict[str, str] = {}
    coordinates: dict[str, dict] = {}
    revisions: list[str] = []
    for path in sorted(corpus.glob("*.txt")):
        content = path.read_bytes()
        key = f"fixture-{path.stem}"
        document = await ingestion.repo.create_document(
            user_id=user_id,
            filename=path.name,
            original_filename=path.name,
            content_type="text/plain",
            storage_path=None,
            project_id=None,
            organization_id=None,
            source_type="benchmark",
            metadata={
                "checksum_sha256": hashlib.sha256(content).hexdigest(),
                "benchmark_fixture": key,
            },
        )
        await ingestion.db.commit()
        document, rows, _ = await ingestion.index_document(
            document_id=document.id,
            user_id=user_id,
            file_content=content,
        )
        documents[key] = document.id
        revisions.append(document.current_revision_id)
        for row in rows:
            if row.chunk_index < 0:
                continue
            # The qrel key is explicitly document-relative, not a runtime UUID.
            stable_key = f"{key}#{row.chunk_index}"
            chunks[stable_key] = row.id
            metadata = json.loads(row.metadata_json or "{}")
            coordinates[stable_key] = {
                "content_hash": hashlib.sha256(row.content.encode()).hexdigest(),
                "source_spans": metadata.get("source_spans", []),
                "revision_id": row.revision_id,
            }
    if not documents:
        raise ValueError("Benchmark fixture corpus is empty")
    return FixtureIndex(documents, chunks, tuple(revisions), coordinates)
