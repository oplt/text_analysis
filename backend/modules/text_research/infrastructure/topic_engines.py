"""Pluggable topic-model engines (§4 gap analysis).

Classical LDA/NMF remain the default. BERTopic is an optional plugin that
fails honestly when heavy dependencies are absent.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TopicModelEngine(Protocol):
    name: str

    def fit(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        ...

    def transform(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        ...


class SklearnLDAEngine:
    name = "sklearn_lda"

    def __init__(self) -> None:
        self._fitted: dict[str, Any] | None = None

    def fit(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        from backend.modules.text_research.infrastructure.topic_models import train_topic_model

        self._fitted = train_topic_model(texts, algorithm="lda", **kwargs)
        return self._fitted

    def transform(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        from backend.modules.text_research.infrastructure.topic_models import transform_topic_model

        fitted = kwargs.get("fitted") or self._fitted
        if fitted is None:
            raise ValueError("SklearnLDAEngine must be fitted before transform")
        return transform_topic_model(fitted["vectorizer"], fitted["model"], texts, "lda")


class SklearnNMFEngine:
    name = "sklearn_nmf"

    def __init__(self) -> None:
        self._fitted: dict[str, Any] | None = None

    def fit(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        from backend.modules.text_research.infrastructure.topic_models import train_topic_model

        self._fitted = train_topic_model(texts, algorithm="nmf", **kwargs)
        return self._fitted

    def transform(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        from backend.modules.text_research.infrastructure.topic_models import transform_topic_model

        fitted = kwargs.get("fitted") or self._fitted
        if fitted is None:
            raise ValueError("SklearnNMFEngine must be fitted before transform")
        return transform_topic_model(fitted["vectorizer"], fitted["model"], texts, "nmf")


class BERTopicEngine:
    """Optional semantic topic engine — not part of the core install."""

    name = "bertopic"

    def __init__(self) -> None:
        try:
            import bertopic  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "BERTopic is optional. Install with: pip install bertopic "
                "(and a compatible embedding backend). Core LDA/NMF remain available."
            ) from exc
        self._model: Any | None = None

    def fit(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        from bertopic import BERTopic

        self._model = BERTopic(
            **{k: v for k, v in kwargs.items() if k in {"nr_topics", "min_topic_size"}}
        )
        topics, probs = self._model.fit_transform(texts)
        return {
            "engine": self.name,
            "model": self._model,
            "topics": topics,
            "probabilities": probs,
            "notes": [
                "BERTopic is an optional plugin. Decomposed embedding → reduction → "
                "clustering → representation steps can be swapped later."
            ],
        }

    def transform(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        model = kwargs.get("model") or self._model
        if model is None:
            raise ValueError("BERTopicEngine must be fitted before transform")
        topics, probs = model.transform(texts)
        return {"topics": topics, "probabilities": probs, "engine": self.name}


def get_topic_engine(name: str = "sklearn_lda") -> TopicModelEngine:
    key = (name or "sklearn_lda").strip().lower()
    if key in {"lda", "sklearn_lda"}:
        return SklearnLDAEngine()
    if key in {"nmf", "sklearn_nmf"}:
        return SklearnNMFEngine()
    if key in {"bertopic", "bertopic_engine"}:
        return BERTopicEngine()
    if key in {"semantic_stack", "semantic"}:
        from backend.modules.text_research.infrastructure.semantic_topic_pipeline import (
            SemanticStackEngine,
        )

        return SemanticStackEngine()
    raise ValueError(
        f"Unknown topic engine {name!r}; use sklearn_lda, sklearn_nmf, bertopic, or semantic_stack"
    )
