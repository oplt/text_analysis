"""Decomposed BERTopic-style semantic topic stack without hard optional deps.

Pipeline::

    EmbeddingProvider → DimensionalityReducer → Clusterer → TopicRepresentation

Uses hashing embeddings by default. UMAP/HDBSCAN are selected when installed;
otherwise falls back to TruncatedSVD + KMeans with explicit availability notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class DimensionalityReducer(Protocol):
    name: str

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Clusterer(Protocol):
    name: str

    def fit_predict(self, reduced: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class TopicRepresentation(Protocol):
    name: str

    def represent(
        self,
        texts: list[str],
        labels: np.ndarray,
        *,
        top_n: int = 10,
    ) -> list[dict[str, Any]]: ...


def _umap_available() -> bool:
    try:
        import umap  # noqa: F401

        return True
    except ImportError:
        return False


def _hdbscan_available() -> bool:
    try:
        import hdbscan  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class CachedEmbeddingStage:
    """Embedding stage with optional artifact-backed cache."""

    provider_name: str = "hashing"
    n_features: int = 256
    model_name: str | None = None
    persist_artifacts: bool = True
    name: str = "embedding"
    _cache: dict[str, list[float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = self.provider_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        from backend.modules.text_research.infrastructure.embeddings import (
            embed_texts_cached,
            get_embedding_provider,
        )

        kwargs: dict[str, Any] = {}
        if self.provider_name == "hashing":
            kwargs["n_features"] = self.n_features
        if self.provider_name == "sentence_transformers" and self.model_name:
            kwargs["model_name"] = self.model_name
        provider = get_embedding_provider(self.provider_name, **kwargs)
        return embed_texts_cached(
            provider,
            texts,
            cache=self._cache,
            model_name=self.model_name or getattr(provider, "model_name", None),
            persist_artifacts=self.persist_artifacts,
        )



@dataclass
class HashingEmbeddingStage:
    """Wrap :func:`get_embedding_provider` hashing backend."""

    n_features: int = 256
    name: str = "hashing"
    persist_artifacts: bool = True
    _cache: dict[str, list[float]] = field(default_factory=dict, init=False, repr=False)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        stage = CachedEmbeddingStage(
            provider_name="hashing",
            n_features=self.n_features,
            persist_artifacts=self.persist_artifacts,
        )
        stage._cache = self._cache
        return stage.embed_texts(texts)


@dataclass
class TruncatedSVDReducer:
    n_components: int = 50
    name: str = "truncated_svd"
    _model: Any | None = field(default=None, init=False, repr=False)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        n = min(self.n_components, embeddings.shape[0], embeddings.shape[1])
        n = max(2, n)
        self._model = TruncatedSVD(n_components=n, random_state=42)
        return self._model.fit_transform(embeddings)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("TruncatedSVDReducer must be fitted before transform")
        return self._model.transform(embeddings)


@dataclass
class UMAPReducer:
    n_components: int = 5
    name: str = "umap"
    _model: Any | None = field(default=None, init=False, repr=False)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        import umap

        n = min(self.n_components, embeddings.shape[0] - 1, embeddings.shape[1])
        n = max(2, n)
        self._model = umap.UMAP(n_components=n, random_state=42)
        return self._model.fit_transform(embeddings)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("UMAPReducer must be fitted before transform")
        return self._model.transform(embeddings)


@dataclass
class PCAReducer:
    n_components: int = 50
    name: str = "pca"
    _model: Any | None = field(default=None, init=False, repr=False)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        n = min(self.n_components, embeddings.shape[0], embeddings.shape[1])
        n = max(2, n)
        self._model = PCA(n_components=n, random_state=42)
        return self._model.fit_transform(embeddings)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("PCAReducer must be fitted before transform")
        return self._model.transform(embeddings)


@dataclass
class KMeansClusterer:
    n_clusters: int = 5
    random_seed: int = 42
    name: str = "kmeans"
    _model: Any | None = field(default=None, init=False, repr=False)

    def fit_predict(self, reduced: np.ndarray) -> np.ndarray:
        k = max(1, min(self.n_clusters, len(reduced)))
        self._model = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
        return self._model.fit_predict(reduced)

    def predict(self, reduced: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("KMeansClusterer must be fitted before predict")
        return self._model.predict(reduced)


@dataclass
class HDBSCANClusterer:
    min_cluster_size: int = 5
    name: str = "hdbscan"
    _model: Any | None = field(default=None, init=False, repr=False)

    def fit_predict(self, reduced: np.ndarray) -> np.ndarray:
        import hdbscan

        self._model = hdbscan.HDBSCAN(
            min_cluster_size=max(2, self.min_cluster_size), prediction_data=True
        )
        labels = self._model.fit_predict(reduced)
        return np.asarray(labels)

    def predict(self, reduced: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("HDBSCANClusterer must be fitted before predict")
        import hdbscan

        labels, _strengths = hdbscan.approximate_predict(self._model, reduced)
        return np.asarray(labels)


@dataclass
class CTFIDFTopicRepresentation:
    name: str = "c_tf_idf"

    def represent(
        self,
        texts: list[str],
        labels: np.ndarray,
        *,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        labels = np.asarray(labels)
        unique = sorted({int(label) for label in labels if int(label) >= 0})
        if not unique:
            return []

        cluster_docs: dict[int, str] = {}
        for label in unique:
            cluster_texts = [texts[i] for i, item in enumerate(labels) if int(item) == label]
            cluster_docs[label] = " ".join(cluster_texts)

        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        matrix = vectorizer.fit_transform([cluster_docs[label] for label in unique])
        feature_names = list(vectorizer.get_feature_names_out())

        topics: list[dict[str, Any]] = []
        for row_index, label in enumerate(unique):
            weights = matrix[row_index].toarray().ravel()
            top_indices = np.argsort(-weights)[:top_n]
            top_terms = [
                {"term": feature_names[idx], "weight": float(weights[idx])}
                for idx in top_indices
                if weights[idx] > 0
            ]
            topics.append({"topic_id": int(label), "top_terms": top_terms})
        return topics


@dataclass
class SemanticTopicPipeline:
    embedder: Any
    reducer: Any
    clusterer: Any
    representation: Any = field(default_factory=CTFIDFTopicRepresentation)
    availability: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    is_fitted: bool = field(default=False, init=False)

    def fit(self, texts: list[str], *, top_n_terms: int = 10) -> dict[str, Any]:
        """Fit stages once so their learned state can infer unseen texts."""
        if not texts:
            raise ValueError("texts must be non-empty")

        embeddings = np.asarray(self.embedder.embed_texts(texts), dtype=float)
        reduced = self.reducer.fit_transform(embeddings)
        labels = np.asarray(self.clusterer.fit_predict(reduced))
        topics = self.representation.represent(texts, labels, top_n=top_n_terms)
        self.is_fitted = True
        return {
            "engine": "semantic_stack",
            "topics": topics,
            "labels": labels.tolist(),
            "components": {
                "embedding": getattr(self.embedder, "name", type(self.embedder).__name__),
                "reducer": self.reducer.name,
                "clusterer": self.clusterer.name,
                "representation": self.representation.name,
            },
            "availability": self.availability,
            "notes": self.notes,
            "n_documents": len(texts),
        }

    def transform(self, texts: list[str]) -> dict[str, Any]:
        """Assign unseen texts to the clusters learned during :meth:`fit`."""
        if not self.is_fitted:
            raise ValueError("SemanticTopicPipeline must be fitted before transform")
        if not texts:
            raise ValueError("texts must be non-empty")

        embeddings = np.asarray(self.embedder.embed_texts(texts), dtype=float)
        reduced = self.reducer.transform(embeddings)
        labels = self.clusterer.predict(reduced)
        return {
            "engine": "semantic_stack",
            "labels": np.asarray(labels).tolist(),
            "n_documents": len(texts),
            "components": {
                "embedding": getattr(self.embedder, "name", type(self.embedder).__name__),
                "reducer": self.reducer.name,
                "clusterer": self.clusterer.name,
                "representation": self.representation.name,
            },
            "notes": self.notes,
        }


def build_semantic_pipeline(
    *,
    n_topics: int = 5,
    n_components: int = 50,
    embedding_provider: str = "hashing",
    embedding_model_name: str | None = None,
    persist_embedding_artifacts: bool = True,
    random_seed: int = 42,
) -> SemanticTopicPipeline:
    """Construct a semantic topic stack with honest optional-component fallbacks."""
    notes: list[str] = []
    availability = {
        "umap": _umap_available(),
        "hdbscan": _hdbscan_available(),
        "sentence_transformers": False,
    }
    try:
        from backend.modules.text_research.infrastructure.embeddings import (
            SentenceTransformerEmbeddingProvider,
        )

        availability["sentence_transformers"] = SentenceTransformerEmbeddingProvider.available()
    except Exception:  # noqa: BLE001
        availability["sentence_transformers"] = False

    embedder: Any = CachedEmbeddingStage(
        provider_name=embedding_provider,
        model_name=embedding_model_name,
        persist_artifacts=persist_embedding_artifacts,
    )
    if embedding_provider == "sentence_transformers" and not availability["sentence_transformers"]:
        raise ValueError(
            "embedding_provider='sentence_transformers' requires the optional "
            "sentence_transformers package."
        )
    if embedding_provider == "hashing":
        notes.append(
            "Using hashing embeddings (lexical, not semantic). "
            "Install sentence_transformers for semantic embeddings."
        )

    if availability["umap"]:
        reducer: Any = UMAPReducer(n_components=min(n_components, 10))
    else:
        reducer = TruncatedSVDReducer(n_components=n_components)
        notes.append("umap-learn not installed; using TruncatedSVD for dimensionality reduction.")

    if availability["hdbscan"]:
        clusterer: Any = HDBSCANClusterer(min_cluster_size=max(2, n_topics))
    else:
        clusterer = KMeansClusterer(n_clusters=n_topics, random_seed=random_seed)
        notes.append("hdbscan not installed; using KMeans clustering.")

    return SemanticTopicPipeline(
        embedder=embedder,
        reducer=reducer,
        clusterer=clusterer,
        availability=availability,
        notes=notes,
    )


def fit_semantic_topics(texts: list[str], **opts: Any) -> dict[str, Any]:
    """Fit the decomposed semantic topic stack and return topics + labels."""
    if not texts:
        raise ValueError("texts must be non-empty")

    n_topics = int(opts.get("n_topics", 5))
    pipeline = build_semantic_pipeline(
        n_topics=n_topics,
        n_components=int(opts.get("n_components", 50)),
        embedding_provider=str(opts.get("embedding_provider", "hashing")),
        embedding_model_name=opts.get("embedding_model_name"),
        persist_embedding_artifacts=bool(opts.get("persist_embedding_artifacts", True)),
        random_seed=int(opts.get("random_seed", 42)),
    )

    return pipeline.fit(texts, top_n_terms=int(opts.get("top_n_terms", 10)))


class SemanticStackEngine:
    """Topic engine plugin wrapping :func:`fit_semantic_topics`."""

    name = "semantic_stack"

    def __init__(self) -> None:
        self._pipeline: SemanticTopicPipeline | None = None

    def fit(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        self._pipeline = build_semantic_pipeline(
            n_topics=int(kwargs.get("n_topics", 5)),
            n_components=int(kwargs.get("n_components", 50)),
            embedding_provider=str(kwargs.get("embedding_provider", "hashing")),
            embedding_model_name=kwargs.get("embedding_model_name"),
            persist_embedding_artifacts=bool(kwargs.get("persist_embedding_artifacts", True)),
            random_seed=int(kwargs.get("random_seed", 42)),
        )
        return self._pipeline.fit(texts, top_n_terms=int(kwargs.get("top_n_terms", 10)))

    def transform(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        if self._pipeline is None:
            raise ValueError("SemanticStackEngine must be fitted before transform")
        return self._pipeline.transform(texts)
