"""Quantitative text analysis engines for text research.

Corpus statistics, term/n-gram frequencies, KWIC, dictionary analysis,
keyness (log-likelihood / G-test), co-occurrence (PMI), and document-feature
matrix (DFM) construction. All numbers are computed directly from the
provided (already segmented + tokenized) text; nothing here is fabricated.

Standardized workflow (quanteda-analogue):

``Corpus → Tokens → Preprocessing → DFM → Feature trimming →
Feature transformation → Statistical analysis / modeling``

See :func:`run_standardized_pipeline` and :func:`corpus_statistics`.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    build_count_vectorizer,
    describe_implementation,
    tokenize,
)
from backend.modules.text_research.infrastructure.weighting import (
    WeightingScheme,
    apply_weighting,
    describe_weighting,
    list_weighting_schemes,
    resolve_weighting_scheme,
)

PIPELINE_STAGES: tuple[str, ...] = (
    "corpus",
    "tokens",
    "preprocessing",
    "dfm",
    "feature_trimming",
    "feature_transformation",
    "statistical_analysis",
)

DEFAULT_MATTR_WINDOW = 50
DEFAULT_MSTTR_WINDOW = 100


def _safe_div(numerator: float, denominator: float) -> float:
    return (numerator / denominator) if denominator else 0.0


def type_token_ratio(tokens: list[str]) -> float:
    """Raw type-token ratio (types / tokens). Sensitive to text length."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def guiraud_r(tokens: list[str]) -> float:
    """Guiraud's R = V / sqrt(N); less length-sensitive than raw TTR."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / math.sqrt(len(tokens))


def herdan_c(tokens: list[str]) -> float:
    """Herdan's C = log(V) / log(N)."""
    n = len(tokens)
    v = len(set(tokens))
    if n <= 1 or v <= 1:
        return 0.0
    return math.log(v) / math.log(n)


def mean_segmental_ttr(tokens: list[str], window: int = DEFAULT_MSTTR_WINDOW) -> float | None:
    """Standardized / mean segmental TTR (MSTTR) over non-overlapping windows.

    Returns ``None`` when fewer than ``window`` tokens are available so callers
    do not treat a raw short-text TTR as a standardized measure.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if len(tokens) < window:
        return None
    scores: list[float] = []
    for start in range(0, len(tokens) - window + 1, window):
        chunk = tokens[start : start + window]
        scores.append(type_token_ratio(chunk))
    # Drop a trailing partial window (MSTTR standard).
    return statistics.mean(scores) if scores else None


def moving_average_ttr(tokens: list[str], window: int = DEFAULT_MATTR_WINDOW) -> float | None:
    """Moving-average TTR (MATTR) over all windows of size ``window``.

    Preferred over raw TTR when comparing texts of unequal length.
    Returns ``None`` when the text is shorter than ``window``.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if len(tokens) < window:
        return None
    scores = [
        type_token_ratio(tokens[i : i + window]) for i in range(0, len(tokens) - window + 1)
    ]
    return statistics.mean(scores) if scores else None


def lexical_diversity(
    tokenized: list[list[str]],
    *,
    mattr_window: int = DEFAULT_MATTR_WINDOW,
    msttr_window: int = DEFAULT_MSTTR_WINDOW,
) -> dict[str, Any]:
    """Corpus- and unit-aware lexical diversity summary.

    Reports raw TTR plus length-robust alternatives. When standardized measures
    cannot be computed, values are ``null`` with an explicit reason — never
    silently substituted with raw TTR.
    """
    pooled = [tok for tokens in tokenized for tok in tokens]
    per_unit_ttr = [type_token_ratio(tokens) for tokens in tokenized if tokens]
    mattr = moving_average_ttr(pooled, window=mattr_window)
    msttr = mean_segmental_ttr(pooled, window=msttr_window)

    return {
        "ttr": type_token_ratio(pooled),
        "mean_unit_ttr": statistics.mean(per_unit_ttr) if per_unit_ttr else 0.0,
        "median_unit_ttr": statistics.median(per_unit_ttr) if per_unit_ttr else 0.0,
        "guiraud_r": guiraud_r(pooled),
        "herdan_c": herdan_c(pooled),
        "mattr": mattr,
        "mattr_window": mattr_window,
        "mattr_available": mattr is not None,
        "mattr_note": None
        if mattr is not None
        else f"Need at least {mattr_window} tokens for MATTR; do not use raw TTR as a substitute.",
        "standardized_ttr": msttr,
        "msttr_window": msttr_window,
        "standardized_ttr_available": msttr is not None,
        "standardized_ttr_note": None
        if msttr is not None
        else f"Need at least {msttr_window} tokens for MSTTR/standardized TTR.",
        "length_caution": (
            "Raw TTR shrinks as texts get longer; prefer MATTR/MSTTR/Guiraud when "
            "comparing units or corpora of unequal length."
        ),
    }


def corpus_statistics(
    texts: list[str],
    tokenized: list[list[str]],
    *,
    document_ids: list[str] | None = None,
    mattr_window: int = DEFAULT_MATTR_WINDOW,
    msttr_window: int = DEFAULT_MSTTR_WINDOW,
) -> dict[str, Any]:
    """Corpus-level descriptive statistics for a set of text units.

    Args:
        texts: raw (or preprocessed) unit texts, one per unit.
        tokenized: token lists aligned 1:1 with ``texts``.
        document_ids: optional parent document ids aligned 1:1 with units;
            when provided, ``document_count`` is the number of unique documents
            and ``text_unit_count`` is the unit count.
    """
    if len(texts) != len(tokenized):
        raise ValueError("texts and tokenized must have the same length")
    if document_ids is not None and len(document_ids) != len(texts):
        raise ValueError("document_ids must align 1:1 with texts")

    n_units = len(texts)
    lengths = [len(tokens) for tokens in tokenized]
    total_tokens = sum(lengths)

    vocabulary: set[str] = set()
    for tokens in tokenized:
        vocabulary.update(tokens)

    if document_ids is not None:
        n_documents = len({doc_id for doc_id in document_ids if doc_id})
    else:
        n_documents = n_units

    diversity = lexical_diversity(
        tokenized, mattr_window=mattr_window, msttr_window=msttr_window
    )

    return {
        # ``document_count`` kept for backward compatibility (= text units when
        # document_ids omitted). Prefer text_unit_count / documents_unique.
        "document_count": n_units,
        "text_unit_count": n_units,
        "documents_unique": n_documents,
        "token_count": total_tokens,
        "vocabulary_size": len(vocabulary),
        "mean_length": _safe_div(total_tokens, n_units),
        "median_length": statistics.median(lengths) if lengths else 0.0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "length_stdev": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
        "lexical_diversity": diversity,
        # Flatten common diversity fields for simple consumers / dashboards.
        "ttr": diversity["ttr"],
        "mattr": diversity["mattr"],
        "standardized_ttr": diversity["standardized_ttr"],
        "guiraud_r": diversity["guiraud_r"],
        "pipeline": {
            "workflow": list(PIPELINE_STAGES),
            "stage_completed": "statistical_analysis",
            "framework_analogue": "quanteda",
        },
    }


def trim_token_vocabulary(
    tokenized: list[list[str]],
    *,
    min_term_frequency: int = 1,
    max_term_frequency: int | None = None,
    min_document_frequency: int = 1,
    max_document_frequency: int | float | None = None,
) -> tuple[list[list[str]], dict[str, Any]]:
    """Pre-DFM feature trimming on token lists (explicit, recorded).

    Full sparse-DFM trimming lives in the DFM trimming API; this helper supports
    the standardized pipeline stage with the same conceptual controls.
    """
    n_units = len(tokenized)
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    for tokens in tokenized:
        tf.update(tokens)
        df.update(set(tokens))

    max_df_abs: int | None
    if max_document_frequency is None:
        max_df_abs = None
    elif isinstance(max_document_frequency, float) and max_document_frequency <= 1.0:
        max_df_abs = int(max_document_frequency * n_units)
    else:
        max_df_abs = int(max_document_frequency)

    kept: set[str] = set()
    for term, count in tf.items():
        if count < min_term_frequency:
            continue
        if max_term_frequency is not None and count > max_term_frequency:
            continue
        doc_freq = df[term]
        if doc_freq < min_document_frequency:
            continue
        if max_df_abs is not None and doc_freq > max_df_abs:
            continue
        kept.add(term)

    trimmed = [[tok for tok in tokens if tok in kept] for tokens in tokenized]
    meta = {
        "min_term_frequency": min_term_frequency,
        "max_term_frequency": max_term_frequency,
        "min_document_frequency": min_document_frequency,
        "max_document_frequency": max_document_frequency,
        "features_before": len(tf),
        "features_after": len(kept),
        "features_removed": len(tf) - len(kept),
    }
    return trimmed, meta


def transform_token_weights(
    tokenized: list[list[str]],
    *,
    weighting: str | WeightingScheme = "count",
    k1: float | None = None,
    b: float | None = None,
    smooth_idf: bool | None = None,
) -> dict[str, Any]:
    """Record a feature-transformation choice applied at DFM construction time.

    Actual matrix values are produced by :func:`apply_weighting` on a count DFM —
    tokenization stays independent of the weighting scheme.
    """
    _ = tokenized  # scheme is matrix-level; tokens only define the count DFM.
    scheme = resolve_weighting_scheme(weighting, k1=k1, b=b, smooth_idf=smooth_idf)
    return describe_weighting(scheme)


@dataclass
class PipelineResult:
    """Result of :func:`run_standardized_pipeline`."""

    stages: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    tokenized: list[list[str]] = field(default_factory=list)
    dfm: dict[str, Any] | None = None
    workflow: list[str] = field(default_factory=lambda: list(PIPELINE_STAGES))

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "stages": self.stages,
            "statistics": self.statistics,
            "dfm": self.dfm,
            "unit_count": len(self.tokenized),
        }


def run_standardized_pipeline(
    texts: list[str],
    config: PreprocessingConfig | dict[str, Any] | None = None,
    *,
    document_ids: list[str] | None = None,
    unit_ids: list[str] | None = None,
    build_dfm: bool = True,
    dfm_mode: str = "count",
    trim: dict[str, Any] | None = None,
    transform_weighting: str | None = None,
    k1: float | None = None,
    b: float | None = None,
    smooth_idf: bool | None = None,
    cache_key: str | None = None,
) -> PipelineResult:
    """Run the Corpus→…→analysis workflow and record each stage.

    Stages that are skipped are still listed with ``status="skipped"`` so
    exports remain explicit about what did and did not run.
    """
    result = PipelineResult()
    cfg = _config_dict(config)
    impl = describe_implementation(cfg) if cfg else describe_implementation({})

    result.stages["corpus"] = {
        "status": "completed",
        "text_unit_count": len(texts),
        "documents_unique": (
            len({d for d in document_ids if d}) if document_ids is not None else len(texts)
        ),
        "unit_ids_provided": unit_ids is not None,
    }

    tokenized = _tokenize_texts(texts, cfg, cache_key=cache_key)
    result.tokenized = tokenized
    result.stages["tokens"] = {
        "status": "completed",
        "token_count": sum(len(t) for t in tokenized),
    }
    result.stages["preprocessing"] = {
        "status": "completed",
        "config": cfg,
        "implementation": impl,
    }

    trim_meta = None
    if trim:
        tokenized, trim_meta = trim_token_vocabulary(tokenized, **trim)
        result.tokenized = tokenized
        result.stages["feature_trimming"] = {"status": "completed", **trim_meta}
    else:
        result.stages["feature_trimming"] = {
            "status": "skipped",
            "note": "Pass trim={min_term_frequency, min_document_frequency, ...} to enable.",
        }

    weighting = transform_weighting or dfm_mode
    canonical = normalize_dfm_weighting(weighting)
    result.stages["feature_transformation"] = {
        "status": "completed" if build_dfm else "deferred",
        **transform_token_weights(
            tokenized, weighting=canonical, k1=k1, b=b, smooth_idf=smooth_idf
        ),
        "requested_weighting": weighting,
        "dfm_mode": canonical,
    }

    if build_dfm:
        result.dfm = build_dfm_matrix(
            tokenized,
            mode=canonical,
            unit_ids=unit_ids,
            preprocessing_config=cfg,
            k1=k1,
            b=b,
            smooth_idf=smooth_idf,
        )
        result.stages["dfm"] = {
            "status": "completed",
            "summary": dfm_summary(result.dfm),
            "mode": canonical,
            "storage": result.dfm.get("storage"),
        }
    else:
        result.stages["dfm"] = {"status": "skipped"}

    result.statistics = corpus_statistics(
        texts, tokenized, document_ids=document_ids
    )
    result.stages["statistical_analysis"] = {
        "status": "completed",
        "metrics": {
            "token_count": result.statistics["token_count"],
            "vocabulary_size": result.statistics["vocabulary_size"],
            "ttr": result.statistics["ttr"],
            "mattr": result.statistics["mattr"],
            "standardized_ttr": result.statistics["standardized_ttr"],
        },
    }
    return result


def resolve_rate_per(rate_per: int | float | None = 1000) -> float:
    """Validate a tokens-per-N normalization denominator (e.g. 100, 1000, 10000)."""
    if rate_per is None:
        return 1000.0
    value = float(rate_per)
    if value <= 0:
        raise ValueError("rate_per must be a positive number (e.g. 100, 1000, 10000)")
    return value


def term_frequencies(
    tokenized: list[list[str]],
    *,
    rate_per: int | float = 1000,
) -> list[dict[str, Any]]:
    """Per-term frequency table for research units.

    Each row includes:

    * ``raw_count`` / ``raw_frequency`` — absolute term frequency
    * ``relative_frequency`` — share of all tokens
    * ``rate`` — per-``rate_per``-token rate (configurable; default per 1,000)
    * ``per_100`` / ``per_1000`` / ``per_10000`` — common normalizations
    * ``document_frequency`` — units containing the term
    * ``document_prevalence`` — document_frequency / unit_count
    * ``rank`` — 1-based rank by descending raw count (ties broken by term)
    * ``cumulative_share`` — running share of token mass after rank sort
    """
    rate_denom = resolve_rate_per(rate_per)
    total_tokens = sum(len(tokens) for tokens in tokenized)
    n_units = len(tokenized)

    raw_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    for tokens in tokenized:
        raw_counts.update(tokens)
        doc_counts.update(set(tokens))

    results: list[dict[str, Any]] = []
    for term, raw_count in raw_counts.items():
        relative_frequency = (raw_count / total_tokens) if total_tokens else 0.0
        doc_freq = doc_counts[term]
        results.append(
            {
                "term": term,
                "raw_count": raw_count,
                "raw_frequency": raw_count,
                "relative_frequency": relative_frequency,
                "rate": relative_frequency * rate_denom,
                "rate_per": rate_denom,
                "per_100": relative_frequency * 100.0,
                "per_1000": relative_frequency * 1000.0,
                "per_10000": relative_frequency * 10000.0,
                "document_frequency": doc_freq,
                "document_prevalence": (doc_freq / n_units) if n_units else 0.0,
            }
        )

    results.sort(key=lambda row: (-row["raw_count"], row["term"]))
    cumulative = 0.0
    for rank, row in enumerate(results, start=1):
        cumulative += row["relative_frequency"]
        row["rank"] = rank
        row["cumulative_share"] = cumulative
    return results


def term_frequency_report(
    tokenized: list[list[str]],
    *,
    rate_per: int | float = 1000,
    top_n: int | None = 50,
    unit_ids: list[str] | None = None,
    document_ids: list[str] | None = None,
    group_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Frequencies plus corpus metadata for grouping / downstream analysis."""
    rate_denom = resolve_rate_per(rate_per)
    rows = term_frequencies(tokenized, rate_per=rate_denom)
    total_tokens = sum(len(tokens) for tokens in tokenized)
    n_units = len(tokenized)
    limited = rows if top_n is None else rows[: max(0, int(top_n))]

    metadata: dict[str, Any] = {
        "unit_count": n_units,
        "token_count": total_tokens,
        "vocabulary_size": len(rows),
        "terms_returned": len(limited),
        "rate_per": rate_denom,
        "normalization": {
            "relative": "raw_count / token_count",
            "rate": f"relative_frequency * {rate_denom}",
            "presets": {"per_100": 100, "per_1000": 1000, "per_10000": 10000},
        },
        "fields": [
            "term",
            "raw_count",
            "raw_frequency",
            "relative_frequency",
            "rate",
            "rate_per",
            "per_100",
            "per_1000",
            "per_10000",
            "document_frequency",
            "document_prevalence",
            "rank",
            "cumulative_share",
        ],
    }
    if unit_ids is not None:
        metadata["unit_ids"] = list(unit_ids)
    if document_ids is not None:
        metadata["document_ids"] = list(document_ids)
        metadata["documents_unique"] = len({d for d in document_ids if d})
    if group_keys is not None:
        if len(group_keys) != n_units:
            raise ValueError("group_keys must align 1:1 with tokenized units")
        metadata["group_keys"] = list(group_keys)
        # Per-group token totals for downstream stratified analysis.
        by_group: dict[str, dict[str, int]] = {}
        for key, tokens in zip(group_keys, tokenized, strict=True):
            bucket = by_group.setdefault(str(key), {"units": 0, "tokens": 0})
            bucket["units"] += 1
            bucket["tokens"] += len(tokens)
        metadata["group_token_totals"] = by_group

    return {"frequencies": limited, "metadata": metadata}


MIN_NGRAM_N = 1
MAX_NGRAM_N = 10  # safe upper bound for arbitrary N


def validate_ngram_order(n: int, *, skip: int = 0) -> int:
    """Validate n-gram order ``n`` (and optional skip) within safe bounds."""
    order = int(n)
    if order < MIN_NGRAM_N or order > MAX_NGRAM_N:
        raise ValueError(
            f"n must be between {MIN_NGRAM_N} and {MAX_NGRAM_N} inclusive (got {n})"
        )
    if int(skip) != 0:
        # Contiguous n-grams only for now; skip-grams reserved without silent no-op.
        raise ValueError(
            "skip-grams are not enabled yet (pass skip=0). "
            "Contiguous unigram/bigram/trigram/N-gram analysis is supported."
        )
    return order


def _ngrams(tokens: list[str], n: int, *, skip: int = 0) -> list[str]:
    """Extract contiguous n-grams (skip must be 0 until skip-grams ship)."""
    order = validate_ngram_order(n, skip=skip)
    if len(tokens) < order:
        return []
    return [" ".join(tokens[i : i + order]) for i in range(len(tokens) - order + 1)]


def ngram_frequencies(
    tokenized: list[list[str]],
    n: int = 1,
    *,
    rate_per: int | float = 1000,
    skip: int = 0,
) -> list[dict[str, Any]]:
    """N-gram frequency table (unigram n=1, bigram n=2, trigram n=3, …).

    Each row includes raw frequency, document frequency, relative/normalized
    rates, prevalence, rank, and cumulative share.
    """
    order = validate_ngram_order(n, skip=skip)
    rate_denom = resolve_rate_per(rate_per)
    per_unit_ngrams = [_ngrams(tokens, order, skip=skip) for tokens in tokenized]
    total_ngrams = sum(len(grams) for grams in per_unit_ngrams)
    n_units = len(tokenized)

    raw_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    for grams in per_unit_ngrams:
        raw_counts.update(grams)
        doc_counts.update(set(grams))

    results: list[dict[str, Any]] = []
    for ngram, raw_count in raw_counts.items():
        relative_frequency = (raw_count / total_ngrams) if total_ngrams else 0.0
        doc_freq = doc_counts[ngram]
        results.append(
            {
                "ngram": ngram,
                "tokens": ngram.split(" "),
                "n": order,
                "skip": 0,
                "raw_count": raw_count,
                "raw_frequency": raw_count,
                "relative_frequency": relative_frequency,
                "normalized_frequency": relative_frequency,
                "rate": relative_frequency * rate_denom,
                "rate_per": rate_denom,
                "per_100": relative_frequency * 100.0,
                "per_1000": relative_frequency * 1000.0,
                "per_10000": relative_frequency * 10000.0,
                "document_frequency": doc_freq,
                "document_prevalence": (doc_freq / n_units) if n_units else 0.0,
                "prevalence": (doc_freq / n_units) if n_units else 0.0,
            }
        )

    results.sort(key=lambda row: (-row["raw_count"], row["ngram"]))
    cumulative = 0.0
    for rank, row in enumerate(results, start=1):
        cumulative += row["relative_frequency"]
        row["rank"] = rank
        row["cumulative_share"] = cumulative
    return results


def ngram_frequency_report(
    tokenized: list[list[str]],
    n: int = 2,
    *,
    rate_per: int | float = 1000,
    top_n: int | None = 50,
    skip: int = 0,
    unit_ids: list[str] | None = None,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    """N-gram frequencies plus metadata for downstream analysis."""
    order = validate_ngram_order(n, skip=skip)
    rate_denom = resolve_rate_per(rate_per)
    rows = ngram_frequencies(tokenized, order, rate_per=rate_denom, skip=skip)
    total_ngrams = sum(len(_ngrams(tokens, order)) for tokens in tokenized)
    limited = rows if top_n is None else rows[: max(0, int(top_n))]

    label = {1: "unigram", 2: "bigram", 3: "trigram"}.get(order, f"{order}-gram")
    metadata: dict[str, Any] = {
        "n": order,
        "n_label": label,
        "skip": 0,
        "skip_grams_supported": False,
        "min_n": MIN_NGRAM_N,
        "max_n": MAX_NGRAM_N,
        "unit_count": len(tokenized),
        "ngram_token_count": total_ngrams,
        "vocabulary_size": len(rows),
        "ngrams_returned": len(limited),
        "rate_per": rate_denom,
        "normalization": {
            "relative": "raw_count / total_ngram_tokens",
            "normalized_frequency": "alias of relative_frequency",
            "rate": f"relative_frequency * {rate_denom}",
        },
        "fields": [
            "ngram",
            "tokens",
            "n",
            "raw_count",
            "raw_frequency",
            "relative_frequency",
            "normalized_frequency",
            "rate",
            "document_frequency",
            "document_prevalence",
            "prevalence",
            "rank",
            "cumulative_share",
        ],
    }
    if unit_ids is not None:
        metadata["unit_ids"] = list(unit_ids)
    if document_ids is not None:
        metadata["document_ids"] = list(document_ids)
        metadata["documents_unique"] = len({d for d in document_ids if d})
    return {"ngrams": limited, "metadata": metadata}


def kwic(
    texts: list[str],
    metadata: list[dict[str, Any]],
    keyword: str,
    window: int = 5,
    *,
    case_sensitive: bool = False,
    query_mode: str | None = "auto",
    language: str | None = None,
    token_attribute: str | None = None,
    max_matches: int | None = None,
) -> list[dict[str, Any]]:
    """Keyword-in-context search (delegates to :mod:`kwic`)."""
    from backend.modules.text_research.infrastructure.kwic import concordance

    return concordance(
        texts,
        metadata,
        keyword,
        window=window,
        case_sensitive=case_sensitive,
        query_mode=query_mode,
        language=language,
        token_attribute=token_attribute,
        max_matches=max_matches,
    )

def dictionary_hits(tokenized: list[list[str]], terms: list[str]) -> dict[str, Any]:
    """Corpus-level dictionary hit counts for a flat term list (legacy helper)."""
    from backend.modules.text_research.infrastructure.dictionary_matcher import (
        match_dictionary,
        parse_dictionary_payload,
    )

    spec = parse_dictionary_payload(terms)
    result = match_dictionary(tokenized, spec)
    return {
        "hits": result["total_hits"],
        "per_1000": result["hits_per_1000_tokens"],
        "unit_prevalence": result["unit_prevalence"],
        "per_unit_hits": [row["hits"] for row in result["per_unit"]],
        "n_units": result["n_units"],
        "terms": spec.flattened_terms(),
    }


def keyness(
    tokenized_a: list[list[str]],
    tokenized_b: list[list[str]],
    top_n: int = 50,
    *,
    method: str = "log_likelihood",
    min_frequency: int = 1,
    correction: str | None = "bh",
    group_a_label: str | None = None,
    group_b_label: str | None = None,
    group_field: str | None = None,
) -> list[dict[str, Any]]:
    """Keyness / group comparison (delegates to :mod:`keyness`)."""
    from backend.modules.text_research.infrastructure.keyness import keyness as _keyness

    return _keyness(
        tokenized_a,
        tokenized_b,
        top_n=top_n,
        method=method,
        min_frequency=min_frequency,
        correction=correction,
        group_a_label=group_a_label,
        group_b_label=group_b_label,
        group_field=group_field,
    )


def cooccurrence(
    tokenized: list[list[str]],
    window: int = 5,
    top_n: int = 100,
    *,
    association_method: str = "pmi",
    directional: bool | str | None = False,
    min_frequency: int = 1,
    min_count: int = 1,
) -> list[dict[str, Any]]:
    """Term co-occurrence / collocation (delegates to :mod:`collocation`)."""
    from backend.modules.text_research.infrastructure.collocation import cooccurrence as _cooc

    return _cooc(
        tokenized,
        window=window,
        top_n=top_n,
        association_method=association_method,
        directional=directional,
        min_frequency=min_frequency,
        min_count=min_count,
    )


DFM_WEIGHTINGS: frozenset[str] = frozenset(
    {
        "count",
        "binary",
        "tf",
        "term_frequency",
        "tfidf",
        "tf-idf",
        "sublinear_tf",
        "sublinear",
        "log_count",
        "bm25",
    }
)


def normalize_dfm_weighting(weighting: str) -> str:
    """Map aliases to canonical DFM weighting mode."""
    return resolve_weighting_scheme(weighting).name


def build_dfm_matrix(
    tokenized: list[list[str]],
    mode: str = "count",
    *,
    unit_ids: list[str] | None = None,
    preprocessing_config: dict[str, Any] | PreprocessingConfig | None = None,
    max_preview: int = 10,
    dense_export_limit: int = 5000,
    force_sparse_only: bool = False,
    k1: float | None = None,
    b: float | None = None,
    smooth_idf: bool | None = None,
    **vectorizer_kwargs: Any,
) -> dict[str, Any]:
    """Build a sparse document/text-unit × feature matrix (DFM).

    Always fits a **count** vocabulary first; the selected weighting scheme
    (count, binary, tf, tfidf, sublinear_tf, log_count, bm25) is applied by
    :func:`apply_weighting` so tokenization stays independent of weighting.
    """
    scheme = resolve_weighting_scheme(mode, k1=k1, b=b, smooth_idf=smooth_idf)
    weighting = scheme.name
    if unit_ids is not None and len(unit_ids) != len(tokenized):
        raise ValueError("unit_ids must align 1:1 with tokenized units")

    documents = [" ".join(tokens) for tokens in tokenized]
    base_kwargs: dict[str, Any] = {
        "tokenizer": str.split,
        "preprocessor": lambda doc: doc,
        "lowercase": False,
        "token_pattern": None,
    }
    trim_config = vectorizer_kwargs.pop("trim", None)
    for key in (
        "unit_ids",
        "preprocessing_config",
        "max_preview",
        "dense_export_limit",
        "force_sparse_only",
        "trim",
        "k1",
        "b",
        "smooth_idf",
    ):
        vectorizer_kwargs.pop(key, None)
    base_kwargs.update(vectorizer_kwargs)

    vectorizer = build_count_vectorizer(config=None, **base_kwargs)
    count_matrix = vectorizer.fit_transform(documents)
    matrix = apply_weighting(count_matrix, scheme)

    n_units, n_features = matrix.shape
    nnz = int(matrix.nnz)
    density = (nnz / (n_units * n_features)) if (n_units and n_features) else 0.0
    feature_names = list(vectorizer.get_feature_names_out())

    preview_rows = min(max_preview, n_units)
    preview_cols = min(max_preview, n_features)
    preview_values = (
        matrix[:preview_rows, :preview_cols].toarray().tolist() if preview_rows and preview_cols else []
    )

    coo = matrix.tocoo()
    sparse_payload = {
        "format": "coo",
        "row": coo.row.tolist(),
        "col": coo.col.tolist(),
        "data": [float(v) for v in coo.data.tolist()],
        "shape": [n_units, n_features],
        "nnz": nnz,
    }

    cells = n_units * n_features
    include_dense = not force_sparse_only and cells <= dense_export_limit and cells > 0
    cfg_payload = _config_dict(preprocessing_config) if preprocessing_config is not None else None
    weighting_meta = describe_weighting(scheme)

    result: dict[str, Any] = {
        "dimensions": {"units": n_units, "features": n_features},
        "density": density,
        "nnz": nnz,
        "feature_names": feature_names,
        "unit_ids": list(unit_ids) if unit_ids is not None else [str(i) for i in range(n_units)],
        "preprocessing_config": cfg_payload,
        "mode": weighting,
        "weighting": weighting,
        "weighting_scheme": weighting_meta,
        "sublinear_tf": weighting == "sublinear_tf",
        "sparse": sparse_payload,
        "storage": "sparse+dense" if include_dense else "sparse",
        "dense_export_limit": dense_export_limit,
        "preview": {
            "rows": preview_rows,
            "cols": preview_cols,
            "feature_names": feature_names[:preview_cols],
            "unit_ids": (list(unit_ids) if unit_ids is not None else [str(i) for i in range(n_units)])[
                :preview_rows
            ],
            "values": preview_values,
        },
    }
    if include_dense:
        result["dense_matrix"] = matrix.toarray().tolist()
    if trim_config:
        result = dfm_trim(
            result,
            max_preview=max_preview,
            dense_export_limit=dense_export_limit,
            force_sparse_only=force_sparse_only,
            **trim_config,
        )
        result["weighting_scheme"] = weighting_meta
    return result



def _config_dict(config: PreprocessingConfig | dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, PreprocessingConfig):
        return config.to_dict()
    return dict(config)


def _tokenize_texts(
    texts: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    cache_key: str | None = None,
) -> list[list[str]]:
    if cache_key:
        from backend.modules.text_research.infrastructure.feature_cache import get_cached, set_cached

        cached = get_cached(cache_key)
        if cached is not None:
            return cached
    cfg = _config_dict(config)
    tokenized = [tokenize(text, cfg) for text in texts]
    if cache_key:
        from backend.modules.text_research.infrastructure.feature_cache import set_cached

        set_cached(cache_key, tokenized)
    return tokenized


def corpus_stats(
    texts: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    cache_key: str | None = None,
    document_ids: list[str] | None = None,
    mattr_window: int = DEFAULT_MATTR_WINDOW,
    msttr_window: int = DEFAULT_MSTTR_WINDOW,
) -> dict[str, Any]:
    tokenized = _tokenize_texts(texts, config, cache_key=cache_key)
    return corpus_statistics(
        texts,
        tokenized,
        document_ids=document_ids,
        mattr_window=mattr_window,
        msttr_window=msttr_window,
    )


def compute_frequencies(
    texts: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    top_n: int = 50,
    rate_per: int | float = 1000,
    cache_key: str | None = None,
    unit_ids: list[str] | None = None,
    document_ids: list[str] | None = None,
    group_keys: list[str] | None = None,
) -> dict[str, Any]:
    tokenized = _tokenize_texts(texts, config, cache_key=cache_key)
    return term_frequency_report(
        tokenized,
        rate_per=rate_per,
        top_n=top_n,
        unit_ids=unit_ids,
        document_ids=document_ids,
        group_keys=group_keys,
    )


def compute_ngrams(
    texts: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    n: int = 2,
    top_n: int = 50,
    rate_per: int | float = 1000,
    skip: int = 0,
    cache_key: str | None = None,
    unit_ids: list[str] | None = None,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    tokenized = _tokenize_texts(texts, config, cache_key=cache_key)
    return ngram_frequency_report(
        tokenized,
        n=n,
        rate_per=rate_per,
        top_n=top_n,
        skip=skip,
        unit_ids=unit_ids,
        document_ids=document_ids,
    )


def build_dfm(
    texts_or_tokenized: list[Any],
    *args: Any,
    weighting: str = "count",
    mode: str | None = None,
    cache_key: str | None = None,
    unit_ids: list[str] | None = None,
    preprocessing_config: dict[str, Any] | PreprocessingConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a DFM from token lists or raw texts plus a preprocessing config.

    Preferred call::

        build_dfm(texts, config, weighting="tfidf", unit_ids=[...])

    Legacy call used by the analysis service::

        build_dfm(texts, unit_ids, config, weighting="count")
    """
    use_mode = mode if mode is not None else weighting
    config = preprocessing_config
    resolved_unit_ids = unit_ids

    if args:
        # Legacy: (unit_ids, config) or (config,)
        if len(args) >= 2 and (
            isinstance(args[1], PreprocessingConfig)
            or (isinstance(args[1], dict) and ("lowercase" in args[1] or "language" in args[1]))
        ):
            resolved_unit_ids = args[0] if resolved_unit_ids is None else resolved_unit_ids
            config = args[1]
        elif len(args) >= 1 and (
            isinstance(args[0], PreprocessingConfig)
            or (isinstance(args[0], dict) and ("lowercase" in args[0] or "language" in args[0]))
        ):
            config = args[0]

    if texts_or_tokenized and isinstance(texts_or_tokenized[0], list):
        tokenized = texts_or_tokenized  # already tokenized
        cfg_payload = _config_dict(config) if config is not None else None
    else:
        if config is None:
            raise ValueError("preprocessing config is required when building a DFM from raw texts")
        tokenized = _tokenize_texts(texts_or_tokenized, config, cache_key=cache_key)
        cfg_payload = _config_dict(config)

    return build_dfm_matrix(
        tokenized,
        mode=use_mode,
        unit_ids=resolved_unit_ids,
        preprocessing_config=cfg_payload,
        **kwargs,
    )


def dfm_summary(result: dict[str, Any]) -> dict[str, Any]:
    dimensions = result.get("dimensions", {})
    sparse = result.get("sparse") or {}
    summary = {
        "unit_count": dimensions.get("units", 0),
        "feature_count": dimensions.get("features", 0),
        "density": result.get("density", 0.0),
        "nnz": result.get("nnz", sparse.get("nnz", 0)),
        "mode": result.get("mode") or result.get("weighting"),
        "weighting": result.get("weighting") or result.get("mode"),
        "storage": result.get("storage", "sparse"),
        "sublinear_tf": bool(result.get("sublinear_tf")),
        "has_unit_ids": bool(result.get("unit_ids")),
        "has_feature_names": bool(result.get("feature_names")),
        "has_preprocessing_config": result.get("preprocessing_config") is not None,
        "has_sparse": "sparse" in result,
        "has_dense": "dense_matrix" in result,
    }
    if result.get("weighting_scheme"):
        summary["weighting_scheme"] = result["weighting_scheme"]
    if result.get("trim"):
        summary["trim"] = result["trim"]
    return summary


def _dfm_column_stats(dfm: dict[str, Any]) -> tuple[list[float], list[int]]:
    """Return per-feature term totals and document frequencies from sparse COO."""
    sparse = dfm.get("sparse")
    if not sparse:
        raise ValueError("DFM has no sparse payload to trim")
    n_features = int(sparse["shape"][1])
    tf = [0.0] * n_features
    seen: list[set[int]] = [set() for _ in range(n_features)]
    for row, col, value in zip(sparse["row"], sparse["col"], sparse["data"], strict=True):
        if value == 0:
            continue
        tf[col] += abs(float(value))
        seen[col].add(int(row))
    df = [len(s) for s in seen]
    return tf, df


def _quantile_threshold(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _resolve_trim_bounds(
    values: list[float],
    *,
    min_value: float | int | None,
    max_value: float | int | None,
    value_type: str,
    n_reference: int,
    label: str,
) -> tuple[float | None, float | None, dict[str, Any]]:
    """Map count/prop/rank/quantile thresholds onto absolute value bounds."""
    kind = (value_type or "count").strip().lower()
    if kind not in {"count", "prop", "rank", "quantile"}:
        raise ValueError(
            f"Unsupported {label}_type {value_type!r}; expected count|prop|rank|quantile"
        )
    meta: dict[str, Any] = {"type": kind, "min": min_value, "max": max_value}
    if not values:
        return None, None, meta

    if kind == "count":
        return (
            float(min_value) if min_value is not None else None,
            float(max_value) if max_value is not None else None,
            meta,
        )

    if kind == "prop":
        total = sum(values) if label == "term_frequency" else float(n_reference)
        if total <= 0:
            return None, None, meta
        abs_min = float(min_value) * total if min_value is not None else None
        abs_max = float(max_value) * total if max_value is not None else None
        meta["absolute_min"] = abs_min
        meta["absolute_max"] = abs_max
        return abs_min, abs_max, meta

    if kind == "rank":
        # Rank 1 = highest value. Keep features with min_rank <= rank <= max_rank.
        order = sorted(range(len(values)), key=lambda i: (-values[i], i))
        rank_of = {idx: rank for rank, idx in enumerate(order, start=1)}
        min_rank = int(min_value) if min_value is not None else 1
        max_rank = int(max_value) if max_value is not None else len(values)
        meta["min_rank"] = min_rank
        meta["max_rank"] = max_rank
        # Encode as a mask via sentinel bounds using ranks stored externally.
        meta["_rank_of"] = rank_of
        return None, None, meta

    # quantile: threshold is a probability in [0, 1]
    abs_min = _quantile_threshold(values, float(min_value)) if min_value is not None else None
    abs_max = _quantile_threshold(values, float(max_value)) if max_value is not None else None
    meta["absolute_min"] = abs_min
    meta["absolute_max"] = abs_max
    return abs_min, abs_max, meta


def dfm_trim(
    dfm: dict[str, Any],
    *,
    min_term_frequency: float | int | None = None,
    max_term_frequency: float | int | None = None,
    term_frequency_type: str = "count",
    min_document_frequency: float | int | None = None,
    max_document_frequency: float | int | None = None,
    document_frequency_type: str = "count",
    top_n: int | None = None,
    max_preview: int = 10,
    dense_export_limit: int | None = None,
    force_sparse_only: bool = False,
) -> dict[str, Any]:
    """Trim features from a DFM (Python-native analogue of quanteda ``dfm_trim``).

    Threshold types (``term_frequency_type`` / ``document_frequency_type``):

    * ``count`` — absolute totals / document counts
    * ``prop`` — proportion of total term mass / of unit count
    * ``rank`` — keep features by descending frequency rank (1 = most frequent)
    * ``quantile`` — threshold at the given probability mass of the frequency distribution

    ``top_n`` is a convenience alias for ``term_frequency_type="rank"`` with
    ``max_term_frequency=top_n``.
    """
    if "sparse" not in dfm:
        raise ValueError("dfm_trim requires a DFM with a sparse COO payload")

    if top_n is not None:
        if max_term_frequency is not None and term_frequency_type != "rank":
            raise ValueError("Pass either top_n or term-frequency bounds, not both")
        term_frequency_type = "rank"
        min_term_frequency = 1 if min_term_frequency is None else min_term_frequency
        max_term_frequency = top_n

    tf, df_counts = _dfm_column_stats(dfm)
    n_units = int(dfm["sparse"]["shape"][0])
    n_features = len(tf)
    feature_names = list(dfm.get("feature_names") or [str(i) for i in range(n_features)])

    tf_min, tf_max, tf_meta = _resolve_trim_bounds(
        tf,
        min_value=min_term_frequency,
        max_value=max_term_frequency,
        value_type=term_frequency_type,
        n_reference=n_units,
        label="term_frequency",
    )
    df_min, df_max, df_meta = _resolve_trim_bounds(
        [float(v) for v in df_counts],
        min_value=min_document_frequency,
        max_value=max_document_frequency,
        value_type=document_frequency_type,
        n_reference=n_units,
        label="document_frequency",
    )

    keep: list[int] = []
    tf_ranks = tf_meta.get("_rank_of")
    df_ranks = df_meta.get("_rank_of")
    for idx in range(n_features):
        if tf_ranks is not None:
            rank = tf_ranks[idx]
            min_rank = int(tf_meta.get("min_rank", 1))
            max_rank = int(tf_meta.get("max_rank", n_features))
            if rank < min_rank or rank > max_rank:
                continue
        else:
            if tf_min is not None and tf[idx] < tf_min:
                continue
            if tf_max is not None and tf[idx] > tf_max:
                continue

        if df_ranks is not None:
            rank = df_ranks[idx]
            min_rank = int(df_meta.get("min_rank", 1))
            max_rank = int(df_meta.get("max_rank", n_features))
            if rank < min_rank or rank > max_rank:
                continue
        else:
            if df_min is not None and df_counts[idx] < df_min:
                continue
            if df_max is not None and df_counts[idx] > df_max:
                continue
        keep.append(idx)

    old_to_new = {old: new for new, old in enumerate(keep)}
    sparse = dfm["sparse"]
    new_rows: list[int] = []
    new_cols: list[int] = []
    new_data: list[float] = []
    for row, col, value in zip(sparse["row"], sparse["col"], sparse["data"], strict=True):
        new_col = old_to_new.get(col)
        if new_col is None or value == 0:
            continue
        new_rows.append(int(row))
        new_cols.append(new_col)
        new_data.append(float(value))

    new_feature_names = [feature_names[i] for i in keep]
    n_new = len(keep)
    nnz = len(new_data)
    density = (nnz / (n_units * n_new)) if (n_units and n_new) else 0.0
    limit = (
        dense_export_limit
        if dense_export_limit is not None
        else int(dfm.get("dense_export_limit") or 5000)
    )
    include_dense = not force_sparse_only and n_units * n_new <= limit and n_new > 0

    # Drop internal rank maps from recorded metadata.
    tf_meta = {k: v for k, v in tf_meta.items() if not k.startswith("_")}
    df_meta = {k: v for k, v in df_meta.items() if not k.startswith("_")}

    trimmed: dict[str, Any] = {
        "dimensions": {"units": n_units, "features": n_new},
        "density": density,
        "nnz": nnz,
        "feature_names": new_feature_names,
        "unit_ids": list(dfm.get("unit_ids") or [str(i) for i in range(n_units)]),
        "preprocessing_config": dfm.get("preprocessing_config"),
        "mode": dfm.get("mode"),
        "weighting": dfm.get("weighting") or dfm.get("mode"),
        "weighting_scheme": dfm.get("weighting_scheme"),
        "sublinear_tf": bool(dfm.get("sublinear_tf")),
        "sparse": {
            "format": "coo",
            "row": new_rows,
            "col": new_cols,
            "data": new_data,
            "shape": [n_units, n_new],
            "nnz": nnz,
        },
        "storage": "sparse+dense" if include_dense else "sparse",
        "dense_export_limit": limit,
        "trim": {
            "features_before": n_features,
            "features_after": n_new,
            "features_removed": n_features - n_new,
            "term_frequency": tf_meta,
            "document_frequency": df_meta,
            "top_n": top_n,
        },
    }

    preview_rows = min(max_preview, n_units)
    preview_cols = min(max_preview, n_new)
    if include_dense or preview_rows and preview_cols:
        # Build dense only for preview / optional export.
        dense = [[0.0 for _ in range(n_new)] for _ in range(n_units)]
        for row, col, value in zip(new_rows, new_cols, new_data, strict=True):
            dense[row][col] = value
        trimmed["preview"] = {
            "rows": preview_rows,
            "cols": preview_cols,
            "feature_names": new_feature_names[:preview_cols],
            "unit_ids": trimmed["unit_ids"][:preview_rows],
            "values": [r[:preview_cols] for r in dense[:preview_rows]],
        }
        if include_dense:
            trimmed["dense_matrix"] = dense
    else:
        trimmed["preview"] = {
            "rows": 0,
            "cols": 0,
            "feature_names": [],
            "unit_ids": [],
            "values": [],
        }
    return trimmed


def kwic_search(
    payload: list[dict[str, Any]],
    keyword: str,
    *,
    window_size: int = 5,
    case_sensitive: bool = False,
    query_mode: str | None = "auto",
    language: str | None = None,
    token_attribute: str | None = None,
    max_matches: int | None = None,
) -> list[dict[str, Any]]:
    from backend.modules.text_research.infrastructure.kwic import kwic_search as _kwic_search

    return _kwic_search(
        payload,
        keyword,
        window_size=window_size,
        case_sensitive=case_sensitive,
        query_mode=query_mode,
        language=language,
        token_attribute=token_attribute,
        max_matches=max_matches,
    )

def dictionary_analysis(
    texts: list[str],
    unit_ids: list[str],
    dictionary_terms_or_spec: Any,
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    group_keys: list[str] | None = None,
    cache_key: str | None = None,
    metadata: list[dict[str, Any]] | None = None,
    case_sensitive: bool = False,
    rate_per: float = 1000.0,
    dictionary_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from backend.modules.text_research.infrastructure.dictionary_matcher import (
        DictionarySpec,
        match_dictionary,
        parse_dictionary_payload,
    )

    if isinstance(dictionary_terms_or_spec, DictionarySpec):
        spec = dictionary_terms_or_spec
    else:
        spec = parse_dictionary_payload(dictionary_terms_or_spec)

    tokenized = _tokenize_texts(texts, config, cache_key=cache_key)
    result = match_dictionary(
        tokenized,
        spec,
        unit_ids=unit_ids,
        metadata=metadata,
        case_sensitive=case_sensitive,
        rate_per=rate_per,
    )
    if dictionary_meta:
        result["dictionary"] = {**result.get("dictionary", {}), **dictionary_meta}

    if group_keys is not None:
        grouped: dict[str, dict[str, float | int]] = {}
        for group, row in zip(group_keys, result["per_unit"], strict=True):
            bucket = grouped.setdefault(group, {"hits": 0, "units": 0})
            bucket["hits"] = int(bucket["hits"]) + int(row["hits"])
            bucket["units"] = int(bucket["units"]) + 1
        result["by_group"] = grouped
    return result


def keyness_for_texts(
    texts_a: list[str],
    texts_b: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    top_n: int = 50,
    method: str = "log_likelihood",
    min_frequency: int = 1,
    correction: str | None = "bh",
    group_a_label: str | None = None,
    group_b_label: str | None = None,
    group_field: str | None = None,
    cache_key_a: str | None = None,
    cache_key_b: str | None = None,
    as_report: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    from backend.modules.text_research.infrastructure.keyness import keyness_report

    tokenized_a = _tokenize_texts(texts_a, config, cache_key=cache_key_a)
    tokenized_b = _tokenize_texts(texts_b, config, cache_key=cache_key_b)
    report = keyness_report(
        tokenized_a,
        tokenized_b,
        method=method,
        top_n=top_n,
        min_frequency=min_frequency,
        correction=correction,
        group_a_label=group_a_label,
        group_b_label=group_b_label,
        group_field=group_field,
    )
    return report if as_report else report["features"]


def similarity_for_texts(
    texts: list[str],
    ids: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    method: str = "tfidf_cosine",
    mode: str = "pairwise",
    group_keys: list[str] | None = None,
    centroid_target: str = "between_groups",
    query_text: str | None = None,
    query_id: str | None = None,
    embeddings: dict[str, list[float]] | None = None,
    query_embedding: list[float] | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    cache_key: str | None = None,
) -> dict[str, Any]:
    """Tokenize ``texts`` and dispatch to :mod:`similarity` for the requested mode.

    ``method='embedding_cosine'`` requires explicit ``embeddings`` (id -> vector);
    this module never computes embeddings itself and never substitutes them
    into the lexical (tfidf_cosine / jaccard) path.
    """
    from backend.modules.text_research.infrastructure import similarity as sim

    canonical_method = sim.normalize_similarity_method(method)
    canonical_mode = sim.normalize_similarity_mode(mode)

    embed_vectors: list[list[float]] | None = None
    query_vector = query_embedding
    if canonical_method == "embedding_cosine":
        if not embeddings:
            raise ValueError(
                "method='embedding_cosine' requires an explicit 'embeddings' mapping "
                "(id -> vector); this platform does not auto-generate embeddings for "
                "the standard similarity path. Use method='tfidf_cosine' or "
                "method='jaccard' instead, or supply embeddings explicitly."
            )
        missing = [item_id for item_id in ids if item_id not in embeddings]
        if missing:
            raise ValueError(
                f"Missing embeddings for {len(missing)} item(s), e.g. {missing[:5]!r}"
            )
        embed_vectors = [embeddings[item_id] for item_id in ids]
        tokenized = None
    else:
        tokenized = _tokenize_texts(texts, config, cache_key=cache_key)

    query_tokens = None
    if canonical_mode == "query":
        if canonical_method == "embedding_cosine":
            if query_vector is None:
                raise ValueError("mode='query' with embedding_cosine requires 'query_embedding'")
        else:
            if not query_text:
                raise ValueError("mode='query' requires 'query_text'")
            query_tokens = tokenize(query_text, _config_dict(config))

    if canonical_mode == "pairwise":
        return sim.pairwise_similarity(
            ids,
            method=canonical_method,
            tokenized=tokenized,
            embeddings=embed_vectors,
            top_k=top_k,
            min_score=min_score,
        )
    if canonical_mode == "query":
        return sim.query_similarity(
            query_id or "query",
            ids,
            method=canonical_method,
            query_tokens=query_tokens,
            tokenized=tokenized,
            query_embedding=query_vector,
            embeddings=embed_vectors,
            top_k=top_k,
            min_score=min_score,
        )
    # group_centroid
    if not group_keys:
        raise ValueError("mode='group_centroid' requires 'group_keys' aligned 1:1 with ids")
    return sim.group_centroid_similarity(
        ids,
        group_keys,
        method=canonical_method,
        tokenized=tokenized,
        embeddings=embed_vectors,
        target=centroid_target,
        top_k=top_k,
        min_score=min_score,
    )


def cooccurrence_for_texts(
    texts: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    window_size: int = 5,
    top_n: int = 50,
    association_method: str = "pmi",
    directional: bool = False,
    min_frequency: int = 1,
    min_count: int = 1,
    include_network: bool = True,
    cache_key: str | None = None,
    as_report: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    from backend.modules.text_research.infrastructure.collocation import collocation_report

    tokenized = _tokenize_texts(texts, config, cache_key=cache_key)
    report = collocation_report(
        tokenized,
        window=window_size,
        top_n=top_n,
        association_method=association_method,
        directional=directional,
        min_frequency=min_frequency,
        min_count=min_count,
        include_network=include_network,
    )
    return report if as_report else report["pairs"]
