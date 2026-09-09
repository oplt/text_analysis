"""Immutable corpus snapshots and prepared-corpus artifacts for text research.

``CorpusSnapshot`` / ``DatasetView`` capture corpus identity at a point in
time. ``PreparedCorpusArtifact`` is the tokenized, checksum-addressable output
of the preprocessing pipeline — the shared input for DFM builders, embeddings,
and downstream models.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(UTC)


def compute_corpus_checksum(unit_ids: list[str], texts: list[str]) -> str:
    """Deterministic sha256 over sorted ``unit_id`` + original text pairs."""
    if len(unit_ids) != len(texts):
        raise ValueError("unit_ids and texts must have the same length")
    pairs = sorted(zip(unit_ids, texts, strict=True), key=lambda item: item[0])
    payload = "\x1e".join(f"{uid}\x1f{text}" for uid, text in pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_pipeline_checksum(
    preprocessing_config: dict[str, Any],
    impl_meta: dict[str, Any] | None = None,
) -> str:
    """Hash preprocessing profile plus tokenizer implementation metadata."""
    payload = {
        "preprocessing_config": preprocessing_config,
        "implementation": impl_meta or {},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _snapshot_fingerprint(*, corpus_id: str, corpus_checksum: str, unit_count: int) -> str:
    raw = json.dumps(
        {"corpus_id": corpus_id, "corpus_checksum": corpus_checksum, "unit_count": unit_count},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorpusSnapshot:
    """Frozen view of a corpus's units and original texts."""

    corpus_id: str
    unit_ids: tuple[str, ...]
    unit_texts: tuple[str, ...]
    document_ids: tuple[str | None, ...]
    metadata_by_unit: dict[str, dict[str, Any]]
    corpus_checksum: str
    created_at: datetime = field(default_factory=_utcnow)
    snapshot_fingerprint: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if len(self.unit_ids) != len(self.unit_texts):
            raise ValueError("unit_ids and unit_texts must have the same length")
        if self.document_ids and len(self.document_ids) != len(self.unit_ids):
            raise ValueError("document_ids must align 1:1 with unit_ids when provided")
        if not self.snapshot_fingerprint:
            object.__setattr__(
                self,
                "snapshot_fingerprint",
                _snapshot_fingerprint(
                    corpus_id=self.corpus_id,
                    corpus_checksum=self.corpus_checksum,
                    unit_count=len(self.unit_ids),
                ),
            )


@dataclass(frozen=True)
class DatasetView:
    """A filtered selection over a ``CorpusSnapshot`` (filters already applied)."""

    corpus_id: str
    unit_ids: tuple[str, ...]
    unit_texts: tuple[str, ...]
    document_ids: tuple[str | None, ...]
    metadata_by_unit: dict[str, dict[str, Any]]
    corpus_checksum: str
    filter_desc: str
    created_at: datetime = field(default_factory=_utcnow)
    snapshot_fingerprint: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if len(self.unit_ids) != len(self.unit_texts):
            raise ValueError("unit_ids and unit_texts must have the same length")
        if self.document_ids and len(self.document_ids) != len(self.unit_ids):
            raise ValueError("document_ids must align 1:1 with unit_ids when provided")
        if not self.snapshot_fingerprint:
            object.__setattr__(
                self,
                "snapshot_fingerprint",
                _snapshot_fingerprint(
                    corpus_id=self.corpus_id,
                    corpus_checksum=self.corpus_checksum,
                    unit_count=len(self.unit_ids),
                ),
            )


@dataclass(frozen=True)
class PreparedCorpusArtifact:
    """Tokenized corpus ready for feature extraction and modeling."""

    unit_ids: tuple[str, ...]
    original_units: tuple[str, ...]
    cleaned_units: tuple[str, ...]
    token_sequences: tuple[tuple[str, ...], ...]
    vocabulary: tuple[str, ...]
    texts_joined: tuple[str, ...]
    preprocessing_profile: dict[str, Any]
    metadata_by_unit: dict[str, dict[str, Any]]
    document_ids: tuple[str | None, ...]
    corpus_checksum: str
    pipeline_checksum: str
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        n = len(self.unit_ids)
        if len(self.original_units) != n:
            raise ValueError("original_units must align 1:1 with unit_ids")
        if len(self.cleaned_units) != n:
            raise ValueError("cleaned_units must align 1:1 with unit_ids")
        if len(self.token_sequences) != n:
            raise ValueError("token_sequences must align 1:1 with unit_ids")
        if len(self.texts_joined) != n:
            raise ValueError("texts_joined must align 1:1 with unit_ids")
        if self.document_ids and len(self.document_ids) != n:
            raise ValueError("document_ids must align 1:1 with unit_ids when provided")


def build_prepared_artifact(
    unit_ids: list[str],
    texts: list[str],
    token_sequences: list[list[str]],
    preprocessing_config: dict[str, Any],
    *,
    metadata_by_unit: dict[str, dict[str, Any]] | None = None,
    document_ids: list[str | None] | None = None,
    cleaned_texts: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    impl_meta: dict[str, Any] | None = None,
) -> PreparedCorpusArtifact:
    """Assemble a ``PreparedCorpusArtifact`` from preprocessing outputs."""
    if len(unit_ids) != len(texts):
        raise ValueError("unit_ids and texts must have the same length")
    if len(token_sequences) != len(texts):
        raise ValueError("token_sequences must align 1:1 with texts")

    originals = tuple(texts)
    cleaned = tuple(cleaned_texts) if cleaned_texts is not None else originals
    if len(cleaned) != len(texts):
        raise ValueError("cleaned_texts must align 1:1 with texts")

    tokens = tuple(tuple(seq) for seq in token_sequences)
    vocab = tuple(sorted({tok for seq in tokens for tok in seq}))
    joined = tuple(" ".join(seq) for seq in tokens)
    docs = tuple(document_ids) if document_ids is not None else tuple([None] * len(texts))

    return PreparedCorpusArtifact(
        unit_ids=tuple(unit_ids),
        original_units=originals,
        cleaned_units=cleaned,
        token_sequences=tokens,
        vocabulary=vocab,
        texts_joined=joined,
        preprocessing_profile=dict(preprocessing_config),
        metadata_by_unit=dict(metadata_by_unit or {}),
        document_ids=docs,
        corpus_checksum=compute_corpus_checksum(unit_ids, texts),
        pipeline_checksum=compute_pipeline_checksum(preprocessing_config, impl_meta),
        provenance=dict(provenance or {}),
    )
