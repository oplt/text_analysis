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

Limitation (documented, not hidden): optional semantic stacks (embedding →
UMAP/HDBSCAN → c-TF-IDF) and BERTopic are available via ``algorithm=
"semantic_stack"`` / ``"bertopic"`` when optional packages are installed.
Classical LDA/NMF remain the dependency-light defaults and are never replaced.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Protocol

import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation

from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact
from backend.modules.text_research.infrastructure.preprocessing import (
    build_count_vectorizer,
    build_tfidf_vectorizer,
)

ALGORITHMS = ("lda", "nmf")
SEMANTIC_ALGORITHMS = ("semantic_stack", "semantic", "bertopic")
ALL_ALGORITHMS = ALGORITHMS + SEMANTIC_ALGORITHMS


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


def _modeling_texts(
    texts: list[str],
    prepared: PreparedCorpusArtifact | None,
) -> list[str]:
    """Return space-joined prepared texts or raw texts in caller order."""
    if prepared is not None:
        return list(prepared.texts_joined)
    return texts


def _resolve_modeling_config(
    config: dict[str, Any] | None,
    prepared: PreparedCorpusArtifact | None,
) -> dict[str, Any] | None:
    if prepared is not None:
        return dict(prepared.preprocessing_profile)
    return config


def transform_topic_model(
    vectorizer: Any,
    model: Any,
    texts_or_prepared: list[str] | PreparedCorpusArtifact,
    algorithm: str,
) -> dict[str, Any]:
    """Infer document-topic distributions for unseen documents."""
    algorithm = algorithm.lower()
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Unsupported topic model algorithm: {algorithm!r}; expected one of {ALGORITHMS}"
        )
    if isinstance(texts_or_prepared, PreparedCorpusArtifact):
        texts = list(texts_or_prepared.texts_joined)
    else:
        texts = texts_or_prepared
    if not texts:
        raise ValueError("texts must be non-empty to transform a topic model")
    dtm = vectorizer.transform(texts)
    doc_topic = model.transform(dtm)
    dominant_topics = np.argmax(doc_topic, axis=1).tolist()
    return {
        "doc_topic_distribution": _to_native(doc_topic),
        "dominant_topics": dominant_topics,
    }


def train_topic_model(
    texts: list[str],
    algorithm: str = "lda",
    n_topics: int = 5,
    config: dict[str, Any] | None = None,
    random_seed: int = 42,
    max_iter: int = 25,
    top_n_terms: int = 10,
    *,
    prepared: PreparedCorpusArtifact | None = None,
    holdout_texts: list[str] | None = None,
    embedding_provider: str = "hashing",
    embedding_model_name: str | None = None,
    persist_embedding_artifacts: bool = True,
) -> dict[str, Any]:
    """Train LDA/NMF or an optional semantic topic family on ``texts``.

    Classical ``lda`` / ``nmf`` remain the default path. ``semantic_stack`` uses
    the decomposed embedding → reduction → clustering → c-TF-IDF pipeline.
    ``bertopic`` requires the optional ``bertopic`` package.
    """
    algorithm = algorithm.lower()
    if algorithm in SEMANTIC_ALGORITHMS:
        return _train_semantic_topic_model(
            texts,
            algorithm=algorithm,
            n_topics=n_topics,
            random_seed=random_seed,
            top_n_terms=top_n_terms,
            prepared=prepared,
            embedding_provider=embedding_provider,
            embedding_model_name=embedding_model_name,
            persist_embedding_artifacts=persist_embedding_artifacts,
        )
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Unsupported topic model algorithm: {algorithm!r}; expected one of {ALL_ALGORITHMS}"
        )

    modeling_texts = _modeling_texts(texts, prepared)
    if not modeling_texts:
        raise ValueError("texts must be non-empty to train a topic model")

    config = _resolve_modeling_config(config, prepared)

    if algorithm == "lda":
        vectorizer = build_count_vectorizer(config)
    else:
        vectorizer = build_tfidf_vectorizer(config)

    dtm = vectorizer.fit_transform(modeling_texts)
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
        topics.append(
            {
                "topic_id": topic_id,
                "top_terms": [
                    {"term": feature_names[idx], "weight": float(weights[idx])}
                    for idx in top_indices
                    if weights[idx] > 0
                ],
            }
        )

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
        if holdout_texts:
            holdout_dtm = vectorizer.transform(holdout_texts)
            diagnostics["holdout_perplexity"] = float(model.perplexity(holdout_dtm))

    return {
        "algorithm": algorithm,
        "family": "classical",
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


def _train_semantic_topic_model(
    texts: list[str],
    *,
    algorithm: str,
    n_topics: int,
    random_seed: int,
    top_n_terms: int,
    prepared: PreparedCorpusArtifact | None,
    embedding_provider: str,
    embedding_model_name: str | None,
    persist_embedding_artifacts: bool,
) -> dict[str, Any]:
    modeling_texts = _modeling_texts(texts, prepared)
    if not modeling_texts:
        raise ValueError("texts must be non-empty to train a topic model")

    from backend.modules.text_research.infrastructure.topic_engines import get_topic_engine

    engine_name = "bertopic" if algorithm == "bertopic" else "semantic_stack"
    engine = get_topic_engine(engine_name)
    fitted = engine.fit(
        modeling_texts,
        n_topics=n_topics,
        nr_topics=n_topics,
        random_seed=random_seed,
        top_n_terms=top_n_terms,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        persist_embedding_artifacts=persist_embedding_artifacts,
    )

    if engine_name == "bertopic":
        labels = [int(x) for x in fitted.get("topics") or []]
        # BERTopic topic info when available
        topics: list[dict[str, Any]] = []
        model = fitted.get("model")
        if model is not None and hasattr(model, "get_topics"):
            for topic_id, terms in model.get_topics().items():
                if int(topic_id) < 0:
                    continue
                topics.append(
                    {
                        "topic_id": int(topic_id),
                        "top_terms": [
                            {"term": str(term), "weight": float(weight)}
                            for term, weight in list(terms)[:top_n_terms]
                        ],
                    }
                )
        notes = list(fitted.get("notes") or [])
        components = {"engine": "bertopic"}
        availability = {"bertopic": True}
    else:
        labels = [int(x) for x in fitted.get("labels") or []]
        topics = list(fitted.get("topics") or [])
        notes = list(fitted.get("notes") or [])
        components = dict(fitted.get("components") or {})
        availability = dict(fitted.get("availability") or {})
        model = getattr(engine, "_pipeline", None)

    topic_ids = sorted({int(t.get("topic_id", i)) for i, t in enumerate(topics)})
    if not topic_ids and labels:
        topic_ids = sorted({lab for lab in labels if lab >= 0})
    if not topic_ids:
        topic_ids = list(range(max(1, n_topics)))
    id_to_col = {topic_id: index for index, topic_id in enumerate(topic_ids)}
    n_cols = len(topic_ids)
    doc_topic = np.zeros((len(modeling_texts), n_cols), dtype=float)
    dominant: list[int] = []
    for row, label in enumerate(labels):
        if label in id_to_col:
            col = id_to_col[label]
            doc_topic[row, col] = 1.0
            dominant.append(col)
        elif n_cols:
            doc_topic[row, :] = 1.0 / n_cols
            dominant.append(int(np.argmax(doc_topic[row])))
        else:
            dominant.append(0)

    # Remap topic_id to contiguous indices for UI consistency with LDA/NMF.
    remapped_topics = []
    for topic in topics:
        old_id = int(topic.get("topic_id", 0))
        remapped_topics.append(
            {
                "topic_id": id_to_col.get(old_id, old_id),
                "top_terms": topic.get("top_terms") or [],
            }
        )
    if not remapped_topics:
        remapped_topics = [{"topic_id": i, "top_terms": []} for i in range(n_cols)]

    diagnostics = {
        "topic_diversity": _topic_diversity(remapped_topics, top_n=top_n_terms),
        "top_term_overlap": _top_term_overlap(remapped_topics, top_n=top_n_terms),
        "family": "semantic",
        "components": components,
        "availability": availability,
        "notes": notes,
    }
    return {
        "algorithm": algorithm,
        "family": "semantic",
        "n_topics": n_cols,
        "topics": remapped_topics,
        "doc_topic_distribution": _to_native(doc_topic),
        "dominant_topics": dominant,
        "diagnostics": diagnostics,
        "feature_names": [],
        "vectorizer": {"type": "semantic_passthrough", "components": components},
        "model": model if model is not None else fitted,
        "notes": notes,
        "components": components,
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
    prepared: PreparedCorpusArtifact | None = None,
    holdout_texts: list[str] | None = None,
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
                prepared=prepared,
                holdout_texts=holdout_texts,
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


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    union = a | b
    return (len(a & b) / len(union)) if union else 0.0


def _greedy_topic_match(
    term_sets_a: list[set[str]], term_sets_b: list[set[str]]
) -> tuple[list[dict[str, Any]], float]:
    """Greedy one-to-one topic matching across two runs by descending Jaccard."""
    candidates: list[tuple[float, int, int]] = []
    for i, a in enumerate(term_sets_a):
        for j, b in enumerate(term_sets_b):
            candidates.append((_jaccard_similarity(a, b), i, j))
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


def _topic_word_vectors(
    topics: list[dict[str, Any]],
    *,
    top_n: int = 10,
) -> tuple[list[np.ndarray], list[str]]:
    """Build aligned sparse weight vectors over the union of top terms."""
    vocab: dict[str, int] = {}
    for topic in topics:
        for item in topic.get("top_terms", [])[:top_n]:
            term = item.get("term")
            if term and term not in vocab:
                vocab[term] = len(vocab)
    if not vocab:
        return [], []

    vectors: list[np.ndarray] = []
    for topic in topics:
        vec = np.zeros(len(vocab), dtype=float)
        for item in topic.get("top_terms", [])[:top_n]:
            term = item.get("term")
            if term in vocab:
                vec[vocab[term]] = float(item.get("weight", 1.0))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
    return vectors, list(vocab.keys())


def _cosine_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_topics_hungarian(
    terms_a: list[set[str]],
    terms_b: list[set[str]],
    *,
    topic_vectors_a: list[np.ndarray] | None = None,
    topic_vectors_b: list[np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """One-to-one topic matching via Hungarian assignment on Jaccard distance."""
    from scipy.optimize import linear_sum_assignment

    n_a = len(terms_a)
    n_b = len(terms_b)
    if n_a == 0 or n_b == 0:
        return [], 0.0

    n = max(n_a, n_b)
    cost = np.ones((n, n), dtype=float)
    for i in range(n_a):
        for j in range(n_b):
            cost[i, j] = 1.0 - _jaccard_similarity(terms_a[i], terms_b[j])

    row_ind, col_ind = linear_sum_assignment(cost)
    matching: list[dict[str, Any]] = []
    for i, j in zip(row_ind, col_ind, strict=False):
        if i < n_a and j < n_b:
            entry: dict[str, Any] = {
                "topic_a": int(i),
                "topic_b": int(j),
                "jaccard": float(1.0 - cost[i, j]),
            }
            if (
                topic_vectors_a is not None
                and topic_vectors_b is not None
                and i < len(topic_vectors_a)
                and j < len(topic_vectors_b)
            ):
                entry["topic_word_cosine"] = _cosine_similarity_vectors(
                    topic_vectors_a[i], topic_vectors_b[j]
                )
            matching.append(entry)
    mean_jaccard = float(np.mean([m["jaccard"] for m in matching])) if matching else 0.0
    return matching, mean_jaccard


def document_distribution_similarity(
    doc_topic_a: np.ndarray,
    doc_topic_b: np.ndarray,
    matching: list[dict[str, Any]],
) -> float | None:
    """Cosine similarity between mean doc-topic vectors after Hungarian alignment."""
    if doc_topic_a.size == 0 or doc_topic_b.size == 0 or not matching:
        return None

    aligned_a = np.zeros_like(doc_topic_a)
    for pair in matching:
        i = int(pair["topic_a"])
        j = int(pair["topic_b"])
        if i < doc_topic_a.shape[1] and j < doc_topic_b.shape[1]:
            aligned_a[:, i] = doc_topic_b[:, j]

    mean_a = doc_topic_a.mean(axis=0)
    mean_b = aligned_a.mean(axis=0)
    return _cosine_similarity_vectors(mean_a, mean_b)


def seed_stability(
    texts: list[str],
    seeds: list[int],
    *,
    algorithm: str = "lda",
    n_topics: int = 5,
    config: dict[str, Any] | None = None,
    max_iter: int = 25,
    top_n_terms: int = 10,
    prepared: PreparedCorpusArtifact | None = None,
) -> dict[str, Any]:
    """Multi-seed topic stability (§40): pairwise Jaccard + Hungarian matching.

    Trains one model per seed (same texts/algorithm/K/config) and reports,
    for every seed pair, the best-match Jaccard overlap between each run's
    topics (via :func:`match_topics_hungarian`). A low mean stability score
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
            prepared=prepared,
        )
        term_sets = [{t["term"] for t in topic["top_terms"]} for topic in result["topics"]]
        topic_vectors, _ = _topic_word_vectors(result["topics"], top_n=top_n_terms)
        runs.append(
            {
                "seed": seed,
                "term_sets": term_sets,
                "topic_vectors": topic_vectors,
                "doc_topic": np.asarray(result["doc_topic_distribution"], dtype=float),
                "diagnostics": result["diagnostics"],
            }
        )

    pairwise: list[dict[str, Any]] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            matching, mean_jaccard = match_topics_hungarian(
                runs[i]["term_sets"],
                runs[j]["term_sets"],
                topic_vectors_a=runs[i]["topic_vectors"] or None,
                topic_vectors_b=runs[j]["topic_vectors"] or None,
            )
            pair_entry: dict[str, Any] = {
                "seed_a": runs[i]["seed"],
                "seed_b": runs[j]["seed"],
                "mean_best_match_jaccard": mean_jaccard,
                "matching": matching,
                "matching_method": "hungarian",
            }
            if runs[i]["topic_vectors"] and runs[j]["topic_vectors"]:
                cosines = [m["topic_word_cosine"] for m in matching if "topic_word_cosine" in m]
                if cosines:
                    pair_entry["mean_topic_word_cosine"] = float(np.mean(cosines))
            doc_sim = document_distribution_similarity(
                runs[i]["doc_topic"], runs[j]["doc_topic"], matching
            )
            if doc_sim is not None:
                pair_entry["document_distribution_similarity"] = doc_sim
            pairwise.append(pair_entry)

    mean_stability = (
        float(np.mean([p["mean_best_match_jaccard"] for p in pairwise])) if pairwise else None
    )
    return {
        "algorithm": algorithm,
        "n_topics": n_topics,
        "seeds": seeds,
        "pairwise": pairwise,
        "mean_stability_jaccard": mean_stability,
        "matching_method": "hungarian",
        "per_seed_diagnostics": [{"seed": r["seed"], **r["diagnostics"]} for r in runs],
        "note": (
            "Stability measured as best-match top-term Jaccard overlap across seed "
            "pairs (0 = no seed produced comparable topics, 1 = identical top terms). "
            "Topic alignment uses Hungarian assignment on Jaccard distance. "
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
