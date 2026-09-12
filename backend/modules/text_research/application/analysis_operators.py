"""Canonical pure operators shared by API services and StageRunner/CLI.

Production FastAPI paths continue to use QuantitativeAnalysisService for
corpus selection, persistence, and async scheduling. StageRunner and the
headless CLI call these same operators on an already-prepared corpus so
analysis math does not drift between surfaces.
"""

from __future__ import annotations

from typing import Any, Callable

from backend.modules.text_research.infrastructure import quantitative
from backend.modules.text_research.infrastructure.prepared_corpus_builder import (
    PreparedCorpusArtifact,
)
from backend.modules.text_research.infrastructure.readability import readability_for_units


def tokenized_from_prepared(prepared: PreparedCorpusArtifact) -> list[list[str]]:
    return [list(seq) for seq in prepared.token_sequences]


def run_frequencies_operator(
    prepared: PreparedCorpusArtifact,
    *,
    top_n: int = 50,
    rate_per: float = 1000,
    group_keys: list[str] | None = None,
) -> dict[str, Any]:
    return quantitative.term_frequency_report(
        tokenized_from_prepared(prepared),
        top_n=top_n,
        rate_per=rate_per,
        unit_ids=list(prepared.unit_ids),
        document_ids=list(prepared.document_ids) if prepared.document_ids else None,
        group_keys=group_keys,
    )


def run_ngrams_operator(
    prepared: PreparedCorpusArtifact,
    *,
    n: int = 2,
    top_n: int = 50,
    rate_per: float = 1000,
    skip: int = 0,
) -> dict[str, Any]:
    return quantitative.ngram_frequency_report(
        tokenized_from_prepared(prepared),
        n=n,
        top_n=top_n,
        rate_per=rate_per,
        skip=skip,
        unit_ids=list(prepared.unit_ids),
        document_ids=list(prepared.document_ids) if prepared.document_ids else None,
    )


def run_dfm_operator(
    prepared: PreparedCorpusArtifact,
    *,
    weighting: str = "count",
    **build_kwargs: Any,
) -> dict[str, Any]:
    result = quantitative.build_dfm(
        tokenized_from_prepared(prepared),
        weighting=weighting,
        unit_ids=list(prepared.unit_ids),
        preprocessing_config=prepared.preprocessing_profile,
        **build_kwargs,
    )
    return {"dfm": result, "summary": quantitative.dfm_summary(result)}


def run_kwic_operator(
    prepared: PreparedCorpusArtifact,
    *,
    keyword: str,
    window_size: int = 5,
    case_sensitive: bool = False,
    query_mode: str = "auto",
    language: str | None = None,
    token_attribute: str | None = None,
    max_matches: int | None = None,
) -> dict[str, Any]:
    payload = [
        {
            "text": prepared.original_units[index],
            "text_unit_id": unit_id,
            "id": unit_id,
        }
        for index, unit_id in enumerate(prepared.unit_ids)
    ]
    matches = quantitative.kwic_search(
        payload,
        keyword,
        window_size=window_size,
        case_sensitive=case_sensitive,
        query_mode=query_mode,
        language=language,
        token_attribute=token_attribute,
        max_matches=max_matches,
    )
    return {"matches": matches, "match_count": len(matches)}


def run_readability_operator(
    prepared: PreparedCorpusArtifact,
) -> dict[str, Any]:
    return readability_for_units(
        list(prepared.original_units),
        list(prepared.unit_ids),
    )


OPERATOR_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "frequencies": run_frequencies_operator,
    "ngrams": run_ngrams_operator,
    "dfm": run_dfm_operator,
    "kwic": run_kwic_operator,
    "readability": run_readability_operator,
}


def list_registered_operators() -> list[str]:
    return sorted(OPERATOR_REGISTRY)
