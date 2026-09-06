"""Quantitative text analysis engines for Policy Text Lab.

Corpus statistics, term/n-gram frequencies, KWIC, dictionary analysis,
keyness (log-likelihood / G-test), co-occurrence (PMI), and document-feature
matrix (DFM) construction. All numbers are computed directly from the
provided (already segmented + tokenized) text; nothing here is fabricated.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any

from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    build_count_vectorizer,
    tokenize,
)

_WORD_RE = re.compile(r"\S+")
_NON_WORD_RE = re.compile(r"[^\w']")


def corpus_statistics(texts: list[str], tokenized: list[list[str]]) -> dict[str, Any]:
    """Basic corpus-level descriptive statistics.

    Args:
        texts: raw (or preprocessed) unit texts, one per unit.
        tokenized: token lists aligned 1:1 with ``texts``.
    """
    if len(texts) != len(tokenized):
        raise ValueError("texts and tokenized must have the same length")

    n_units = len(texts)
    lengths = [len(tokens) for tokens in tokenized]
    total_tokens = sum(lengths)

    vocabulary: set[str] = set()
    for tokens in tokenized:
        vocabulary.update(tokens)

    return {
        "document_count": n_units,
        "token_count": total_tokens,
        "vocabulary_size": len(vocabulary),
        "mean_length": (total_tokens / n_units) if n_units else 0.0,
        "median_length": statistics.median(lengths) if lengths else 0.0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def term_frequencies(tokenized: list[list[str]]) -> list[dict[str, Any]]:
    """Per-term raw counts, relative frequency, per-1000-token rate, and
    document prevalence (fraction of units containing the term at least
    once). Sorted by descending raw count.
    """
    total_tokens = sum(len(tokens) for tokens in tokenized)
    n_units = len(tokenized)

    raw_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    for tokens in tokenized:
        raw_counts.update(tokens)
        doc_counts.update(set(tokens))

    results = []
    for term, raw_count in raw_counts.items():
        relative_frequency = (raw_count / total_tokens) if total_tokens else 0.0
        results.append(
            {
                "term": term,
                "raw_count": raw_count,
                "relative_frequency": relative_frequency,
                "per_1000": relative_frequency * 1000,
                "document_prevalence": (doc_counts[term] / n_units) if n_units else 0.0,
            }
        )

    results.sort(key=lambda row: (-row["raw_count"], row["term"]))
    return results


def _ngrams(tokens: list[str], n: int) -> list[str]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_frequencies(tokenized: list[list[str]], n: int = 1) -> list[dict[str, Any]]:
    """N-gram frequencies (unigrams when n=1, bigrams when n=2, etc.)."""
    per_unit_ngrams = [_ngrams(tokens, n) for tokens in tokenized]
    total_ngrams = sum(len(grams) for grams in per_unit_ngrams)
    n_units = len(tokenized)

    raw_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    for grams in per_unit_ngrams:
        raw_counts.update(grams)
        doc_counts.update(set(grams))

    results = []
    for ngram, raw_count in raw_counts.items():
        relative_frequency = (raw_count / total_ngrams) if total_ngrams else 0.0
        results.append(
            {
                "ngram": ngram,
                "n": n,
                "raw_count": raw_count,
                "relative_frequency": relative_frequency,
                "document_prevalence": (doc_counts[ngram] / n_units) if n_units else 0.0,
            }
        )

    results.sort(key=lambda row: (-row["raw_count"], row["ngram"]))
    return results


def kwic(
    texts: list[str],
    metadata: list[dict[str, Any]],
    keyword: str,
    window: int = 5,
) -> list[dict[str, Any]]:
    """Keyword-in-context search.

    Args:
        texts: raw unit texts.
        metadata: dicts aligned 1:1 with ``texts`` (e.g. document, organization,
            year, text_unit_id); each field is copied into every match.
        keyword: case-insensitive whole-token match (punctuation-insensitive).
        window: number of surrounding tokens to include on each side.
    """
    if len(texts) != len(metadata):
        raise ValueError("texts and metadata must have the same length")

    keyword_normalized = _NON_WORD_RE.sub("", keyword).lower()
    results: list[dict[str, Any]] = []

    for text, meta in zip(texts, metadata, strict=True):
        raw_tokens = _WORD_RE.findall(text)
        normalized_tokens = [_NON_WORD_RE.sub("", tok).lower() for tok in raw_tokens]

        for idx, normalized in enumerate(normalized_tokens):
            if normalized != keyword_normalized:
                continue
            left = raw_tokens[max(0, idx - window) : idx]
            right = raw_tokens[idx + 1 : idx + 1 + window]
            entry = {
                "left_context": " ".join(left),
                "keyword": raw_tokens[idx],
                "right_context": " ".join(right),
            }
            entry.update(meta)
            results.append(entry)

    return results


def dictionary_hits(tokenized: list[list[str]], terms: list[str]) -> dict[str, Any]:
    """Corpus-level dictionary hit counts.

    Returns aggregate ``hits``, ``per_1000`` (tokens), ``unit_prevalence``
    (fraction of units with >=1 hit) plus a ``per_unit_hits`` breakdown.
    """
    term_set = {t.lower() for t in terms}
    total_tokens = sum(len(tokens) for tokens in tokenized)
    n_units = len(tokenized)

    per_unit_hits: list[int] = []
    units_with_hit = 0
    total_hits = 0

    for tokens in tokenized:
        hits = sum(1 for t in tokens if t.lower() in term_set)
        per_unit_hits.append(hits)
        total_hits += hits
        if hits > 0:
            units_with_hit += 1

    return {
        "hits": total_hits,
        "per_1000": (total_hits / total_tokens * 1000) if total_tokens else 0.0,
        "unit_prevalence": (units_with_hit / n_units) if n_units else 0.0,
        "per_unit_hits": per_unit_hits,
        "n_units": n_units,
        "terms": sorted(term_set),
    }


def keyness(
    tokenized_a: list[list[str]],
    tokenized_b: list[list[str]],
    top_n: int = 50,
) -> list[dict[str, Any]]:
    """Keyness comparison of two token corpora using log-likelihood (G2 / Dunning).

    Returns the ``top_n`` terms by |G2|, each with ``feature``, ``freq_a``,
    ``freq_b``, ``keyness_statistic`` (G2, always >= 0), and
    ``effect_direction`` (``"a"``, ``"b"``, or ``"neutral"``).
    """
    counts_a: Counter[str] = Counter(t for tokens in tokenized_a for t in tokens)
    counts_b: Counter[str] = Counter(t for tokens in tokenized_b for t in tokens)
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    grand_total = total_a + total_b

    vocabulary = set(counts_a) | set(counts_b)
    results = []

    for term in vocabulary:
        a = counts_a.get(term, 0)
        b = counts_b.get(term, 0)
        if a == 0 and b == 0:
            continue

        expected_a = total_a * (a + b) / grand_total if grand_total else 0.0
        expected_b = total_b * (a + b) / grand_total if grand_total else 0.0

        g2 = 0.0
        if a > 0 and expected_a > 0:
            g2 += a * math.log(a / expected_a)
        if b > 0 and expected_b > 0:
            g2 += b * math.log(b / expected_b)
        g2 *= 2

        rate_a = (a / total_a) if total_a else 0.0
        rate_b = (b / total_b) if total_b else 0.0
        if rate_a > rate_b:
            direction = "a"
        elif rate_b > rate_a:
            direction = "b"
        else:
            direction = "neutral"

        results.append(
            {
                "feature": term,
                "freq_a": a,
                "freq_b": b,
                "keyness_statistic": g2,
                "effect_direction": direction,
            }
        )

    results.sort(key=lambda row: -row["keyness_statistic"])
    return results[:top_n]


def cooccurrence(
    tokenized: list[list[str]],
    window: int = 5,
    top_n: int = 100,
) -> list[dict[str, Any]]:
    """Term co-occurrence within a sliding window, ranked by count.

    ``association_score`` is pointwise mutual information (PMI) computed
    from empirical unigram and co-occurrence probabilities within the
    corpus's token stream.
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    term_counts: Counter[str] = Counter()
    total_tokens = 0

    for tokens in tokenized:
        total_tokens += len(tokens)
        term_counts.update(tokens)
        n = len(tokens)
        for i in range(n):
            for j in range(i + 1, min(i + 1 + window, n)):
                a, b = tokens[i], tokens[j]
                if a == b:
                    continue
                pair_counts[tuple(sorted((a, b)))] += 1

    results = []
    for (term_a, term_b), count in pair_counts.items():
        p_a = (term_counts[term_a] / total_tokens) if total_tokens else 0.0
        p_b = (term_counts[term_b] / total_tokens) if total_tokens else 0.0
        p_ab = (count / total_tokens) if total_tokens else 0.0
        if p_a > 0 and p_b > 0 and p_ab > 0:
            association_score = math.log(p_ab / (p_a * p_b))
        else:
            association_score = 0.0

        results.append(
            {
                "term_a": term_a,
                "term_b": term_b,
                "count": count,
                "association_score": association_score,
            }
        )

    results.sort(key=lambda row: (-row["count"], row["term_a"], row["term_b"]))
    return results[:top_n]


def build_dfm_matrix(
    tokenized: list[list[str]],
    mode: str = "count",
    max_preview: int = 10,
    dense_export_limit: int = 5000,
    **vectorizer_kwargs: Any,
) -> dict[str, Any]:
    """Build a document/text-unit x feature matrix (document-feature matrix).

    Args:
        tokenized: token lists, one per document/unit.
        mode: ``"count"``, ``"binary"``, or ``"tfidf"``.
        max_preview: max rows/cols returned in the small dense ``preview``.
        dense_export_limit: if ``units * features`` is at or below this,
            the full dense matrix is included as ``dense_matrix``; otherwise
            a sparse (COO) export is included as ``sparse``.
        **vectorizer_kwargs: forwarded to the underlying vectorizer
            (e.g. ``ngram_range``, ``min_df``, ``max_df``, ``max_features``).
    """
    documents = [" ".join(tokens) for tokens in tokenized]

    base_kwargs: dict[str, Any] = {
        "tokenizer": str.split,
        "preprocessor": lambda doc: doc,
        "lowercase": False,
        "token_pattern": None,
    }
    base_kwargs.update(vectorizer_kwargs)

    if mode == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(**base_kwargs)
    elif mode == "binary":
        vectorizer = build_count_vectorizer(config=None, binary=True, **base_kwargs)
    elif mode == "count":
        vectorizer = build_count_vectorizer(config=None, **base_kwargs)
    else:
        raise ValueError(f"Unsupported DFM mode: {mode!r}")

    matrix = vectorizer.fit_transform(documents)
    n_units, n_features = matrix.shape
    density = (matrix.nnz / (n_units * n_features)) if (n_units and n_features) else 0.0
    feature_names = list(vectorizer.get_feature_names_out())

    preview_rows = min(max_preview, n_units)
    preview_cols = min(max_preview, n_features)
    preview_values = matrix[:preview_rows, :preview_cols].toarray().tolist()

    result: dict[str, Any] = {
        "dimensions": {"units": n_units, "features": n_features},
        "density": density,
        "feature_names": feature_names,
        "mode": mode,
        "preview": {
            "rows": preview_rows,
            "cols": preview_cols,
            "feature_names": feature_names[:preview_cols],
            "values": preview_values,
        },
    }

    if n_units * n_features <= dense_export_limit:
        result["dense_matrix"] = matrix.toarray().tolist()
    else:
        coo = matrix.tocoo()
        result["sparse"] = {
            "row": coo.row.tolist(),
            "col": coo.col.tolist(),
            "data": coo.data.tolist(),
            "shape": [n_units, n_features],
        }

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
) -> dict[str, Any]:
    tokenized = _tokenize_texts(texts, config, cache_key=cache_key)
    return corpus_statistics(texts, tokenized)


def compute_frequencies(
    texts: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    top_n: int = 50,
    cache_key: str | None = None,
) -> list[dict[str, Any]]:
    tokenized = _tokenize_texts(texts, config, cache_key=cache_key)
    return term_frequencies(tokenized)[:top_n]


def compute_ngrams(
    texts: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    n: int = 2,
    top_n: int = 50,
    cache_key: str | None = None,
) -> list[dict[str, Any]]:
    tokenized = _tokenize_texts(texts, config, cache_key=cache_key)
    return ngram_frequencies(tokenized, n=n)[:top_n]


def build_dfm(
    texts_or_tokenized: list[Any],
    *args: Any,
    weighting: str = "count",
    mode: str | None = None,
    cache_key: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a DFM from token lists or raw texts plus a preprocessing config."""
    if len(args) >= 2 and (
        isinstance(args[1], PreprocessingConfig)
        or (isinstance(args[1], dict) and "lowercase" in args[1])
    ):
        tokenized = _tokenize_texts(texts_or_tokenized, args[1], cache_key=cache_key)
        return build_dfm_matrix(tokenized, mode=weighting)
    use_mode = mode if mode is not None else weighting
    return build_dfm_matrix(texts_or_tokenized, mode=use_mode, **kwargs)


def dfm_summary(result: dict[str, Any]) -> dict[str, Any]:
    dimensions = result.get("dimensions", {})
    return {
        "unit_count": dimensions.get("units", 0),
        "feature_count": dimensions.get("features", 0),
        "density": result.get("density", 0.0),
    }


def kwic_search(
    payload: list[dict[str, Any]],
    keyword: str,
    *,
    window_size: int = 5,
    case_sensitive: bool = False,
) -> list[dict[str, Any]]:
    texts = [row["text"] for row in payload]
    metadata = [{key: value for key, value in row.items() if key != "text"} for row in payload]
    if case_sensitive:
        results: list[dict[str, Any]] = []
        for text, meta in zip(texts, metadata, strict=True):
            start = 0
            while True:
                idx = text.find(keyword, start)
                if idx == -1:
                    break
                left = text[max(0, idx - window_size * 8) : idx].split()[-window_size:]
                right = text[idx + len(keyword) :].split()[:window_size]
                results.append(
                    {
                        **meta,
                        "left_context": " ".join(left),
                        "keyword": keyword,
                        "right_context": " ".join(right),
                    }
                )
                start = idx + len(keyword)
        return results
    return kwic(texts, metadata, keyword, window=window_size)


def dictionary_analysis(
    texts: list[str],
    unit_ids: list[str],
    dictionary_terms: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    group_keys: list[str] | None = None,
    cache_key: str | None = None,
) -> dict[str, Any]:
    tokenized = _tokenize_texts(texts, config, cache_key=cache_key)
    base = dictionary_hits(tokenized, dictionary_terms)
    per_unit = base.pop("per_unit_hits")
    result: dict[str, Any] = {
        "total_hits": base["hits"],
        "hits_per_1000_tokens": base["per_1000"],
        "document_prevalence": base["unit_prevalence"],
        "per_unit": [
            {"text_unit_id": unit_id, "hits": hits}
            for unit_id, hits in zip(unit_ids, per_unit, strict=True)
        ],
        "terms": base["terms"],
    }
    if group_keys is not None:
        grouped: dict[str, dict[str, float | int]] = {}
        for group, hits in zip(group_keys, per_unit, strict=True):
            bucket = grouped.setdefault(group, {"hits": 0, "units": 0})
            bucket["hits"] = int(bucket["hits"]) + hits
            bucket["units"] = int(bucket["units"]) + 1
        result["by_group"] = grouped
    return result


def keyness_for_texts(
    texts_a: list[str],
    texts_b: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    top_n: int = 50,
    cache_key_a: str | None = None,
    cache_key_b: str | None = None,
) -> list[dict[str, Any]]:
    tokenized_a = _tokenize_texts(texts_a, config, cache_key=cache_key_a)
    tokenized_b = _tokenize_texts(texts_b, config, cache_key=cache_key_b)
    return keyness(tokenized_a, tokenized_b, top_n=top_n)


def cooccurrence_for_texts(
    texts: list[str],
    config: PreprocessingConfig | dict[str, Any] | None,
    *,
    window_size: int = 5,
    top_n: int = 50,
    cache_key: str | None = None,
) -> list[dict[str, Any]]:
    tokenized = _tokenize_texts(texts, config, cache_key=cache_key)
    return cooccurrence(tokenized, window=window_size, top_n=top_n)
