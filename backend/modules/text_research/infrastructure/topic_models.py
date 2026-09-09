"""Topic modeling engines (LDA / NMF) for text research (§40-42).

Given fixed input texts, a fixed algorithm/config, and a fixed
``random_seed``, training is fully reproducible (same top terms, same
document-topic distribution) — required for the "seed stability" robustness
checks described in the research workflow.

Implemented here (see module docstrings on each function for detail):

* LDA / NMF baselines (unchanged, no heavy optional dependencies required).
* NPMI-based topic coherence, computed directly from the fitted
  document-term matrix (no ``gensim`` dependency; gensim is not installed in
  this project, see :func:`topic_coherence_npmi`).
* :func:`k_sweep` — fit multiple ``n_topics`` values and return a comparison
  table (coherence/diversity/overlap per K); no automatic "best K" is picked
  (§42 — model selection stays a human decision).
* :func:`seed_stability` — multi-seed stability via pairwise Jaccard overlap
  on top terms, with optional greedy cross-seed topic matching.
* :class:`TopicModelEngine` — a minimal plug-in protocol so a future engine
  (e.g. BERTopic) could be added without changing callers (§41).

Limitation (documented, not hidden): BERTopic/transformer-based topic models
are **not implemented**. Adding them would pull in ``transformers``/
``sentence-transformers``/``torch`` as hard dependencies, which the platform
explicitly avoids for core functionality (§41). LDA/NMF remain the
dependency-light, reproducible baselines.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Protocol

import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation

from backend.modules.text_research.infrastructure.preprocessing import (
    build_count_vectorizer,
    build_tfidf_vectorizer,
)

ALGORITHMS = ("lda", "nmf")


def _to_native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_native(v) for v in value]
    return value


def _topic_diversity(topics: list[dict[str, Any]], top_n: int = 10) -> float:
    """Fraction of unique terms among the union of all topics' top terms.

    1.0 means every topic's top terms are completely distinct; values closer
    to 0 indicate topics share a lot of vocabulary (possible redundancy).
    """
    all_terms: list[str] = []
    for topic in topics:
        all_terms.extend(term["term"] for term in topic["top_terms"][:top_n])
    if not all_terms:
        return 0.0
    return len(set(all_terms)) / len(all_terms)


def _top_term_overlap(topics: list[dict[str, Any]], top_n: int = 10) -> float:
    """Mean pairwise Jaccard overlap between topics' top-term sets."""
    if len(topics) < 2:
        return 0.0
    term_sets = [{term["term"] for term in topic["top_terms"][:top_n]} for topic in topics]
    overlaps = []
    for i in range(len(term_sets)):
        for j in range(i + 1, len(term_sets)):
            union = term_sets[i] | term_sets[j]
            if not union:
                continue
            overlaps.append(len(term_sets[i] & term_sets[j]) / len(union))
    return float(np.mean(overlaps)) if overlaps else 0.0


def _pairwise_npmi(
    binary_dtm: Any,
    term_indices: list[int],
    n_docs: int,
    *,
    eps: float = 1e-12,
) -> float:
    """Mean pairwise NPMI over ``term_indices`` (Bouma NPMI / Röder C_NPMI style).

    Uses document co-occurrence (does the term appear anywhere in the
    document?), computed from the already-fitted document-term matrix — no
    external corpus or extra vectorization pass needed.
    """
    if n_docs == 0 or len(term_indices) < 2:
        return 0.0
    doc_freq = {idx: float(binary_dtm[:, idx].sum()) for idx in term_indices}
    scores: list[float] = []
    for i, j in combinations(term_indices, 2):
        di, dj = doc_freq[i], doc_freq[j]
        if di == 0 or dj == 0:
            continue
        co_doc = float(binary_dtm[:, i].multiply(binary_dtm[:, j]).sum())
        if co_doc == 0:
            # Standard NPMI convention: never co-occurring pairs score the
            # minimum (-1), not 0 — 0 would understate incoherence.
            scores.append(-1.0)
            continue
        p_i = di / n_docs
        p_j = dj / n_docs
        p_ij = co_doc / n_docs
        numerator = math.log((p_ij + eps) / (p_i * p_j))
        denominator = -math.log(p_ij + eps)
        scores.append(numerator / denominator if denominator else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def topic_coherence_npmi(
    dtm: Any,
    topics: list[dict[str, Any]],
    feature_names: list[str],
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    """NPMI-based topic coherence (§40), computed without ``gensim``.

    ``gensim`` (the usual home of ``CoherenceModel``) is not a project
    dependency; this reimplements the well-known NPMI coherence measure
    (Bouma 2009 / Röder et al. 2015's C_NPMI) directly against the
    ``CountVectorizer``/``TfidfVectorizer`` document-term matrix already
    produced by :func:`train_topic_model`, binarized to document
    presence/absence. Interpretation: higher (closer to 1) means a topic's
    top terms tend to co-occur in the same documents more than chance;
    values near/below 0 suggest the top terms are only loosely related.

    This is a coherence *proxy*, not a validated construct — never claim it
    corresponds to a real-world topic quality judgment on its own.
    """
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    binary_dtm = (dtm > 0).astype(np.float64) if hasattr(dtm, "astype") else dtm
    n_docs = dtm.shape[0]

    per_topic: list[dict[str, Any]] = []
    for topic in topics:
        indices = [
            name_to_index[t["term"]]
            for t in topic["top_terms"][:top_n]
            if t["term"] in name_to_index
        ]
        score = _pairwise_npmi(binary_dtm, indices, n_docs)
        per_topic.append({"topic_id": topic["topic_id"], "coherence_npmi": score})

    mean_score = float(np.mean([t["coherence_npmi"] for t in per_topic])) if per_topic else None
    return {
        "method": "npmi",
        "top_n": top_n,
        "per_topic": per_topic,
        "mean_coherence_npmi": mean_score,
        "note": (
            "Computed via document co-occurrence NPMI on the fitted vectorizer's "
            "document-term matrix (gensim not required). Higher is more coherent; "
            "treat as a diagnostic proxy, not ground truth for topic quality."
        ),
    }


def train_topic_model(
    texts: list[str],
    algorithm: str = "lda",
    n_topics: int = 5,
    config: dict[str, Any] | None = None,
    random_seed: int = 42,
    max_iter: int = 25,
    top_n_terms: int = 10,
) -> dict[str, Any]:
    """Train an LDA or NMF topic model on ``texts``.

    Args:
        texts: unit/document texts to vectorize and model.
        algorithm: ``"lda"`` (CountVectorizer + LatentDirichletAllocation) or
            ``"nmf"`` (TfidfVectorizer + NMF).
        n_topics: requested number of topics (capped to vocabulary size).
        config: preprocessing config forwarded to the vectorizer factory.
        random_seed: fixes vectorizer-independent randomness for
            reproducibility (LDA's variational init; NMF's ``nndsvda`` init
            is itself deterministic but the seed is still recorded).
        max_iter: solver iteration cap.
        top_n_terms: number of top terms to report per topic.

    Returns:
        Dict with ``topics`` (id + top_terms with weights),
        ``doc_topic_distribution``, ``dominant_topics``, ``diagnostics``
        (``topic_diversity``, ``top_term_overlap``, and ``perplexity`` for
        LDA), plus the fitted ``vectorizer``/``model`` for persistence.
    """
    algorithm = algorithm.lower()
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Unsupported topic model algorithm: {algorithm!r}; expected one of {ALGORITHMS}"
        )

    if not texts:
        raise ValueError("texts must be non-empty to train a topic model")

    if algorithm == "lda":
        vectorizer = build_count_vectorizer(config)
    else:
        vectorizer = build_tfidf_vectorizer(config)

    dtm = vectorizer.fit_transform(texts)
    feature_names = list(vectorizer.get_feature_names_out())

    if dtm.shape[1] == 0:
        raise ValueError("Vocabulary is empty after preprocessing; cannot train a topic model")

    effective_n_topics = max(1, min(n_topics, dtm.shape[1], dtm.shape[0]))

    perplexity: float | None = None
    if algorithm == "lda":
        model = LatentDirichletAllocation(
            n_components=effective_n_topics,
            max_iter=max_iter,
            random_state=random_seed,
            learning_method="batch",
        )
        doc_topic = model.fit_transform(dtm)
        perplexity = float(model.perplexity(dtm))
    else:
        model = NMF(
            n_components=effective_n_topics,
            max_iter=max(max_iter, 200),
            random_state=random_seed,
            init="nndsvda",
        )
        doc_topic = model.fit_transform(dtm)

    components = model.components_

    topics: list[dict[str, Any]] = []
    for topic_id in range(effective_n_topics):
        weights = components[topic_id]
        top_indices = np.argsort(-weights)[:top_n_terms]
        top_terms = [
            {"term": feature_names[idx], "weight": float(weights[idx])} for idx in top_indices
        ]
        topics.append({"topic_id": topic_id, "top_terms": top_terms})

    dominant_topics = np.argmax(doc_topic, axis=1).tolist()

    coherence = topic_coherence_npmi(dtm, topics, feature_names, top_n=top_n_terms)
    diagnostics: dict[str, Any] = {
        "topic_diversity": _topic_diversity(topics, top_n=top_n_terms),
        "top_term_overlap": _top_term_overlap(topics, top_n=top_n_terms),
        "coherence_npmi": coherence["mean_coherence_npmi"],
        "coherence_per_topic": coherence["per_topic"],
        "coherence_method": coherence["method"],
    }
    if perplexity is not None:
        diagnostics["perplexity"] = perplexity

    return {
        "algorithm": algorithm,
        "n_topics": effective_n_topics,
        "random_seed": random_seed,
        "topics": topics,
        "doc_topic_distribution": _to_native(doc_topic),
        "dominant_topics": dominant_topics,
        "diagnostics": diagnostics,
        "feature_names": feature_names,
        "vectorizer": vectorizer,
        "model": model,
    }


def k_sweep(
    texts: list[str],
    k_values: list[int],
    *,
    algorithm: str = "lda",
    config: dict[str, Any] | None = None,
    random_seed: int = 42,
    max_iter: int = 25,
    top_n_terms: int = 10,
) -> list[dict[str, Any]]:
    """Fit multiple ``n_topics`` values and return a diagnostics table (§40/§42).

    Each row reports the requested/effective K, coherence, diversity,
    top-term overlap, perplexity (LDA only), and a short preview of each
    topic's top terms — enough for a human to *compare* configurations.

    Deliberately does **not** select a "best" K automatically (§42: model
    comparison stays a human decision; a single statistic like coherence is
    not a reliable stand-in for what the researcher actually needs).
    """
    rows: list[dict[str, Any]] = []
    for k in k_values:
        try:
            result = train_topic_model(
                texts,
                algorithm=algorithm,
                n_topics=k,
                config=config,
                random_seed=random_seed,
                max_iter=max_iter,
                top_n_terms=top_n_terms,
            )
        except ValueError as exc:
            rows.append({"requested_n_topics": k, "status": "not_evaluable", "reason": str(exc)})
            continue
        rows.append(
            {
                "requested_n_topics": k,
                "effective_n_topics": result["n_topics"],
                "status": "ok",
                **result["diagnostics"],
                "top_terms_preview": [
                    [t["term"] for t in topic["top_terms"][:5]] for topic in result["topics"]
                ],
            }
        )
    return rows


def _greedy_topic_match(
    term_sets_a: list[set[str]], term_sets_b: list[set[str]]
) -> tuple[list[dict[str, Any]], float]:
    """Greedy one-to-one topic matching across two runs by descending Jaccard.

    Not a global optimum (that would be a linear assignment problem), but a
    fast, auditable approximation adequate for comparing a handful of topics.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, a in enumerate(term_sets_a):
        for j, b in enumerate(term_sets_b):
            union = a | b
            jaccard = (len(a & b) / len(union)) if union else 0.0
            candidates.append((jaccard, i, j))
    candidates.sort(key=lambda item: -item[0])

    used_a: set[int] = set()
    used_b: set[int] = set()
    matching: list[dict[str, Any]] = []
    for jaccard, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matching.append({"topic_a": i, "topic_b": j, "jaccard": jaccard})
    mean_jaccard = float(np.mean([m["jaccard"] for m in matching])) if matching else 0.0
    return matching, mean_jaccard


def seed_stability(
    texts: list[str],
    seeds: list[int],
    *,
    algorithm: str = "lda",
    n_topics: int = 5,
    config: dict[str, Any] | None = None,
    max_iter: int = 25,
    top_n_terms: int = 10,
) -> dict[str, Any]:
    """Multi-seed topic stability (§40): pairwise Jaccard + greedy topic matching.

    Trains one model per seed (same texts/algorithm/K/config) and reports,
    for every seed pair, the best-match Jaccard overlap between each run's
    topics (via :func:`_greedy_topic_match`). A low mean stability score
    means the topic solution is sensitive to random initialization — an
    important caveat before treating any single run's topics as "the" model.
    """
    if len(seeds) < 2:
        raise ValueError("seed_stability requires at least 2 seeds to compare")

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        result = train_topic_model(
            texts,
            algorithm=algorithm,
            n_topics=n_topics,
            config=config,
            random_seed=seed,
            max_iter=max_iter,
            top_n_terms=top_n_terms,
        )
        term_sets = [{t["term"] for t in topic["top_terms"]} for topic in result["topics"]]
        runs.append({"seed": seed, "term_sets": term_sets, "diagnostics": result["diagnostics"]})

    pairwise: list[dict[str, Any]] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            matching, mean_jaccard = _greedy_topic_match(runs[i]["term_sets"], runs[j]["term_sets"])
            pairwise.append(
                {
                    "seed_a": runs[i]["seed"],
                    "seed_b": runs[j]["seed"],
                    "mean_best_match_jaccard": mean_jaccard,
                    "matching": matching,
                }
            )

    mean_stability = (
        float(np.mean([p["mean_best_match_jaccard"] for p in pairwise])) if pairwise else None
    )
    return {
        "algorithm": algorithm,
        "n_topics": n_topics,
        "seeds": seeds,
        "pairwise": pairwise,
        "mean_stability_jaccard": mean_stability,
        "per_seed_diagnostics": [{"seed": r["seed"], **r["diagnostics"]} for r in runs],
        "note": (
            "Stability measured as best-match top-term Jaccard overlap across seed "
            "pairs (0 = no seed produced comparable topics, 1 = identical top terms). "
            "This is a stability diagnostic, not evidence that a topic represents a "
            "real-world construct."
        ),
    }


class TopicModelEngine(Protocol):
    """Future-compatible plug-in interface for topic model engines (§41).

    ``ClassicalTopicModelEngine`` (LDA/NMF via scikit-learn) is the only
    implementation shipped. A BERTopic (or other embedding-based) engine
    could implement this same ``fit`` signature later WITHOUT requiring
    ``transformers``/``torch``/``sentence-transformers`` as core platform
    dependencies — it is intentionally not implemented here.
    """

    name: str

    def fit(
        self,
        texts: list[str],
        *,
        n_topics: int,
        config: dict[str, Any] | None,
        random_seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


class ClassicalTopicModelEngine:
    """LDA/NMF engine implementing :class:`TopicModelEngine` (scikit-learn only)."""

    def __init__(self, algorithm: str = "lda"):
        if algorithm.lower() not in ALGORITHMS:
            raise ValueError(f"Unsupported algorithm {algorithm!r}; expected one of {ALGORITHMS}")
        self.algorithm = algorithm.lower()
        self.name = self.algorithm

    def fit(
        self,
        texts: list[str],
        *,
        n_topics: int = 5,
        config: dict[str, Any] | None = None,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return train_topic_model(
            texts,
            algorithm=self.algorithm,
            n_topics=n_topics,
            config=config,
            random_seed=random_seed,
            **kwargs,
        )
