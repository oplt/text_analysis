"""Canonical, output-affecting RAG component configuration identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ComponentConfig:
    def canonical_provenance_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_provenance_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ParsingConfig(ComponentConfig):
    parser_version: str
    pdf_parser: str
    unicode_normalization: str = "NFC"
    text_decode_max_replacement_ratio: float = 0.02
    pdf_ocr_enabled: bool = False
    pdf_ocr_min_text_chars: int = 40
    pdf_text_quality_threshold: float = 0.65
    pdf_table_extraction_enabled: bool = False
    pdf_header_footer_suppression: bool = True


@dataclass(frozen=True, slots=True)
class ChunkingConfig(ComponentConfig):
    chunker_version: str
    policy_version: str
    chunk_size: int
    chunk_overlap: int
    parent_window_size: int


@dataclass(frozen=True, slots=True)
class EmbeddingConfig(ComponentConfig):
    provider: str
    model: str
    model_version: str | None
    dimensions: int
    preprocessing_version: str


@dataclass(frozen=True, slots=True)
class RetrievalConfig(ComponentConfig):
    algorithm_version: str
    index_version: str
    fusion_method: str
    rrf_k: int
    dense_candidates: int
    lexical_candidates: int
    top_k: int
    score_threshold: float
    parent_context_enabled: bool
    retrieval_branch_concurrency: int = 2
    query_variant_concurrency: int = 3


@dataclass(frozen=True, slots=True)
class RerankingConfig(ComponentConfig):
    enabled: bool
    heuristic_enabled: bool
    max_depth: int
    # Intent-specific depth overrides; missing intents fall back to max_depth.
    # FACT is forced to 0 by the planner regardless of this map.
    intent_depth_overrides: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class GenerationConfig(ComponentConfig):
    max_context_tokens: int
    context_overlap_dedupe_threshold: float
    context_ordering_policy: str = "relevance"


@dataclass(frozen=True, slots=True)
class SynthesisConfig(ComponentConfig):
    passages_per_document: int
    reduce_token_budget: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class EvaluationConfig(ComponentConfig):
    """Offline evaluation harness identity; never affects live retrieval answers."""

    schema_version: int = 1
    end_to_end_qrels_version: str = "fixture-v1"
    citation_entailment_enabled: bool = False
    load_benchmark_scales: tuple[str, ...] = ("10k", "100k", "1M")
