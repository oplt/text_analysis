"""Topic modeling engines (LDA / NMF) for Policy Text Lab.

Given fixed input texts, a fixed algorithm/config, and a fixed
``random_seed``, training is fully reproducible (same top terms, same
document-topic distribution) — required for the "seed stability" robustness
checks described in the research workflow.
"""

from __future__ import annotations

from typing import Any

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

    diagnostics: dict[str, Any] = {
        "topic_diversity": _topic_diversity(topics, top_n=top_n_terms),
        "top_term_overlap": _top_term_overlap(topics, top_n=top_n_terms),
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
