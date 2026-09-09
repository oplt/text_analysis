"""Optional embedding providers for research analyses (§45).

Embeddings stay an *optional* analysis family — never required for DFM,
frequencies, classification baselines, or classical LDA/NMF topic models.
Research code must not treat RAG retrieval chunks as canonical document text.

Providers:

* ``HashingEmbeddingProvider`` — always available lexical-hash baseline
  (deterministic feature hashing, **not** semantic).
* ``SentenceTransformerEmbeddingProvider`` — optional semantic embeddings when
  ``sentence_transformers`` is installed.
* ``UnavailableEmbeddingProvider`` — honest failure stub.

Embeddings can be cached in-memory and as content-addressable artifacts under
``RESEARCH_ARTIFACT_ROOT/embedding_cache/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.feature_extraction import FeatureHasher


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal embedding provider contract."""

    name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


class UnavailableEmbeddingProvider:
    """Default provider: honest failure until a real backend is configured."""

    name = "unavailable"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise ValueError(
            "Semantic embeddings are not configured for text research. "
            "Configure an EmbeddingProvider (optional analysis family) — "
            "embeddings are not part of the standard quantitative path."
        )


class HashingEmbeddingProvider:
    """Lexical-hash baseline via sklearn ``FeatureHasher`` (not semantic).

    Maps whitespace-tokenized text into a fixed dense vector using a
    deterministic hashing trick. Suitable for embedding + logistic-regression
    baselines without transformer dependencies. **Do not** treat outputs as
    semantic similarity vectors.
    """

    name = "hashing"

    def __init__(self, *, n_features: int = 256, alternate_sign: bool = False) -> None:
        self.n_features = n_features
        self.alternate_sign = alternate_sign

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        hasher = FeatureHasher(
            n_features=self.n_features,
            alternate_sign=self.alternate_sign,
            input_type="string",
        )
        tokenized = [(text or "").lower().split() for text in texts]
        matrix = hasher.transform(tokenized)
        return np.asarray(matrix.toarray(), dtype=float).tolist()


class SentenceTransformerEmbeddingProvider:
    """Optional semantic embeddings via ``sentence_transformers``."""

    name = "sentence_transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError(
                "SentenceTransformerEmbeddingProvider requires the optional "
                "sentence_transformers package; it is not installed in this environment."
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @staticmethod
    def available() -> bool:
        try:
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return np.asarray(vectors, dtype=float).tolist()


_PROVIDER_NAMES = frozenset({"hashing", "sentence_transformers", "unavailable"})


def get_embedding_provider(name: str = "hashing", **kwargs: Any) -> EmbeddingProvider:
    """Resolve an embedding provider by name (default: lexical-hash baseline)."""
    if name not in _PROVIDER_NAMES:
        raise ValueError(
            f"Unknown embedding provider {name!r}; expected one of {sorted(_PROVIDER_NAMES)}"
        )
    if name == "hashing":
        return HashingEmbeddingProvider(**kwargs)
    if name == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(**kwargs)
    return UnavailableEmbeddingProvider()


def get_default_embedding_provider() -> EmbeddingProvider:
    """Resolve the active provider (default: lightweight hashing baseline)."""
    return get_embedding_provider("hashing")


def _cache_key(text: str, provider_name: str, *, model_name: str | None = None) -> str:
    payload = f"{provider_name}|{model_name or ''}|{text}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return f"{provider_name}:{digest}"


def _artifact_cache_dir() -> Path:
    from backend.modules.text_research.infrastructure.model_storage import ARTIFACT_ROOT

    path = Path(ARTIFACT_ROOT) / "embedding_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_path_for_key(cache_key: str) -> Path:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return _artifact_cache_dir() / f"{digest}.json"


def load_embedding_artifact(cache_key: str) -> list[float] | None:
    path = _artifact_path_for_key(cache_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    vector = payload.get("vector")
    if not isinstance(vector, list):
        return None
    return [float(v) for v in vector]


def save_embedding_artifact(
    cache_key: str,
    vector: list[float],
    *,
    provider_name: str,
    model_name: str | None = None,
) -> str:
    path = _artifact_path_for_key(cache_key)
    payload = {
        "cache_key": cache_key,
        "provider": provider_name,
        "model_name": model_name,
        "vector": vector,
        "dim": len(vector),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return str(path)


def embed_texts_cached(
    provider: EmbeddingProvider,
    texts: list[str],
    cache: dict[str, list[float]] | None = None,
    *,
    model_name: str | None = None,
    persist_artifacts: bool = False,
) -> list[list[float]]:
    """Embed texts with optional in-memory and artifact caches.

    When ``persist_artifacts`` is true, vectors are also stored under
    ``RESEARCH_ARTIFACT_ROOT/embedding_cache/`` (content-addressable JSON).
    """
    if not texts:
        return []
    store = cache if cache is not None else {}
    results: list[list[float] | None] = [None] * len(texts)
    pending: list[tuple[int, str]] = []
    resolved_model = model_name or getattr(provider, "model_name", None)

    for index, text in enumerate(texts):
        key = _cache_key(text, provider.name, model_name=resolved_model)
        if key in store:
            results[index] = store[key]
            continue
        if persist_artifacts:
            artifact = load_embedding_artifact(key)
            if artifact is not None:
                store[key] = artifact
                results[index] = artifact
                continue
        pending.append((index, text))

    if pending:
        embedded = provider.embed_texts([text for _, text in pending])
        for (index, text), vector in zip(pending, embedded, strict=True):
            key = _cache_key(text, provider.name, model_name=resolved_model)
            store[key] = vector
            results[index] = vector
            if persist_artifacts:
                save_embedding_artifact(
                    key,
                    vector,
                    provider_name=provider.name,
                    model_name=resolved_model,
                )

    return [vector for vector in results if vector is not None]


def describe_embedding_capabilities() -> dict[str, Any]:
    default = get_default_embedding_provider()
    return {
        "available": True,
        "provider": default.name,
        "providers": {
            "hashing": {"available": True, "semantic": False},
            "sentence_transformers": {
                "available": SentenceTransformerEmbeddingProvider.available(),
                "semantic": True,
            },
            "unavailable": {"available": True, "semantic": False},
        },
        "artifact_cache": str(_artifact_cache_dir()),
        "notes": [
            "Default provider is hashing (lexical feature hash, not semantic).",
            "sentence_transformers is optional; enable persist_artifacts to cache vectors on disk.",
            "Embeddings are optional and not wired into the default DFM/classifier path.",
            "Do not use RAG chunk embeddings as research canonical text.",
        ],
    }
