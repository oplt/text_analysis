"""Deterministic identity for the indexed evidence used by RAG."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from backend.modules.rag.domain.models import RetrievedChunk


class StaleEvidenceRevisionError(ValueError):
    """Retrieved chunks do not match the frozen revision allow-list."""

    code = "stale_evidence_revision"


def assert_chunks_match_revision_allow_list(
    chunks: Iterable[RetrievedChunk],
    index_revision_ids: list[str] | None,
) -> None:
    """Fail closed when frozen retrieval returns chunks outside its revision set."""
    if index_revision_ids is None:
        return
    allowed = set(index_revision_ids)
    for chunk in chunks:
        revision = chunk.index_revision_id or (chunk.metadata or {}).get("revision_id")
        if revision is None or revision not in allowed:
            raise StaleEvidenceRevisionError(
                "stale_evidence_revision: retrieved chunk outside frozen revision allow-list"
            )


def _config_identity(config: Any) -> dict[str, Any]:
    if all(hasattr(config, name) for name in ("parsing", "chunking", "embedding")):
        identity = {
            "parsing": config.parsing.canonical_provenance_dict(),
            "parsing_fingerprint": config.parsing.fingerprint(),
            "chunking": config.chunking.canonical_provenance_dict(),
            "chunking_fingerprint": config.chunking.fingerprint(),
            "embedding": config.embedding.canonical_provenance_dict(),
            "embedding_fingerprint": config.embedding.fingerprint(),
        }
        if hasattr(config, "generation"):
            identity["generation"] = config.generation.canonical_provenance_dict()
            identity["generation_fingerprint"] = config.generation.fingerprint()
        if hasattr(config, "synthesis"):
            identity["synthesis"] = config.synthesis.canonical_provenance_dict()
            identity["synthesis_fingerprint"] = config.synthesis.fingerprint()
        if hasattr(config, "reranking"):
            identity["reranking"] = config.reranking.canonical_provenance_dict()
            identity["reranking_fingerprint"] = config.reranking.fingerprint()
        return identity
    return {
        "parser_version": getattr(config, "parser_version", None),
        "chunker_version": getattr(config, "chunker_version", None),
        "embedding_provider": getattr(config, "embedding_provider", None),
        "embedding_model": getattr(config, "embedding_model", None),
        "embedding_model_version": getattr(config, "embedding_model_version", None),
        "embedding_dimensions": getattr(config, "embedding_dimensions", None),
        "embedding_preprocessing_version": getattr(config, "embedding_preprocessing_version", None),
        "retrieval_algorithm_version": getattr(config, "retrieval_algorithm_version", None),
        "index_version": getattr(config, "index_version", None),
        "fusion_method": getattr(config, "fusion_method", None),
        "rrf_k": getattr(config, "rrf_k", None),
        "rerank_enabled": getattr(config, "rerank_enabled", None),
        "rerank_heuristic_enabled": getattr(config, "rerank_heuristic_enabled", None),
        "parent_context_enabled": getattr(config, "parent_context_enabled", None),
    }


def chunk_revision_identity(chunk: Any) -> dict[str, Any]:
    content = getattr(chunk, "content", None) or ""
    content_hash = getattr(chunk, "content_hash", None)
    metadata = getattr(chunk, "metadata", None) or {}
    if not metadata:
        raw_metadata = getattr(chunk, "metadata_json", None)
        if raw_metadata:
            try:
                metadata = json.loads(raw_metadata)
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "chunk_id": getattr(chunk, "id", None) or getattr(chunk, "chunk_id", None),
        "index_revision_id": getattr(chunk, "revision_id", None)
        or getattr(chunk, "index_revision_id", None)
        or metadata.get("document_revision"),
        "chunk_index": getattr(chunk, "chunk_index", None),
        "content_hash": content_hash
        or metadata.get("content_hash")
        or hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_fingerprint": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "parser_version": getattr(chunk, "parser_version", None) or metadata.get("parser_version"),
        "chunker_version": getattr(chunk, "chunker_version", None)
        or metadata.get("chunker_version"),
        "parent_chunk_id": getattr(chunk, "parent_chunk_id", None),
        "source_unit_ids": metadata.get("source_unit_ids") or metadata.get("source_span_ids"),
        "vector_external_id": getattr(chunk, "vector_external_id", None),
        "metadata": metadata,
    }


def build_evidence_revision_hash(
    *,
    corpus_id: str | None,
    project_id: str | None,
    document_revisions: Iterable[dict[str, Any]],
    config: Any,
) -> str:
    """Hash corpus identity, indexed document revisions, and RAG versions."""
    documents = []
    for document in document_revisions:
        normalized = dict(document)
        chunks = normalized.get("chunks")
        if isinstance(chunks, list):
            normalized["chunks"] = sorted(
                chunks,
                key=lambda item: (
                    str(item.get("chunk_index", "")),
                    str(item.get("chunk_id", "")),
                ),
            )
        documents.append(normalized)
    documents.sort(key=lambda item: str(item.get("rag_document_id", "")))
    payload = {
        "schema_version": "1",
        "corpus_id": corpus_id,
        "project_id": project_id,
        "documents": documents,
        "rag_config": _config_identity(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_retrieved_evidence_revision_hash(
    *,
    project_id: str | None,
    document_ids: list[str] | None,
    chunks: Iterable[RetrievedChunk],
    config: Any,
) -> str:
    """Provide a deterministic trace identity when no corpus snapshot is available."""
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk.document_id].append(chunk_revision_identity(chunk))
    documents = [
        {
            "rag_document_id": document_id,
            "chunks": sorted(
                by_document.get(document_id, []),
                key=lambda item: (str(item.get("chunk_index", "")), str(item.get("chunk_id", ""))),
            ),
        }
        for document_id in sorted(document_ids or by_document)
    ]
    return build_evidence_revision_hash(
        corpus_id=None,
        project_id=project_id,
        document_revisions=documents,
        config=config,
    )
