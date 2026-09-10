"""Pluggable registry for research analysis engines and transforms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

PluginFactory = Callable[..., Any]

_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "topic_engine": {},
    "embedding_provider": {},
    "classifier_engine": {},
    "text_transform": {},
    "analysis": {},
    "execution_engine": {},
}

_BUILTINS_REGISTERED = False


def register_plugin(
    kind: str,
    name: str,
    factory: PluginFactory,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Register a named plugin factory under ``kind``."""
    bucket = _REGISTRY.setdefault(kind, {})
    bucket[name] = {"factory": factory, "metadata": metadata or {}}


def get_plugin(kind: str, name: str) -> PluginFactory:
    """Return a plugin factory by kind and name."""
    bucket = _REGISTRY.get(kind)
    if bucket is None or name not in bucket:
        raise KeyError(f"unknown plugin {kind!r}:{name!r}")
    return bucket[name]["factory"]


def list_plugins(kind: str | None = None) -> dict[str, list[str]]:
    """List registered plugin names, optionally scoped to one kind."""
    if kind is not None:
        return {kind: sorted(_REGISTRY.get(kind, {}))}
    return {key: sorted(bucket) for key, bucket in _REGISTRY.items()}


def unregister_plugin(kind: str, name: str) -> None:
    """Remove a plugin registration (primarily for tests)."""
    bucket = _REGISTRY.get(kind)
    if bucket is not None:
        bucket.pop(name, None)


def register_builtins() -> None:
    """Register built-in topic, embedding, and analysis plugins."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from backend.modules.text_research.domain.analysis_specification import ANALYSIS_TYPES
    from backend.modules.text_research.infrastructure.embeddings import get_embedding_provider
    from backend.modules.text_research.infrastructure.engines.python_engine import (
        PythonAnalysisEngine,
    )
    from backend.modules.text_research.infrastructure.engines.r_engine import RAnalysisEngine
    from backend.modules.text_research.infrastructure.pipeline_compiler import compile_plan
    from backend.modules.text_research.infrastructure.topic_engines import get_topic_engine

    register_plugin("execution_engine", "python", PythonAnalysisEngine)
    register_plugin("execution_engine", "r", RAnalysisEngine)

    for engine_name in ("sklearn_lda", "sklearn_nmf", "bertopic", "semantic_stack"):
        register_plugin(
            "topic_engine",
            engine_name,
            lambda name=engine_name: get_topic_engine(name),
            metadata={"engine": engine_name},
        )

    from backend.modules.text_research.infrastructure.classifiers import (
        ALGORITHMS as CLASSIFIER_ALGORITHMS,
    )
    from backend.modules.text_research.infrastructure.embedding_classifier import (
        ALGORITHMS as EMBEDDING_ALGORITHMS,
    )

    for algo in CLASSIFIER_ALGORITHMS:
        register_plugin(
            "classifier_engine",
            algo,
            lambda algorithm=algo: algorithm,
            metadata={"algorithm": algo, "family": "sparse"},
        )
    embedding_aliases = {
        "embedding_logistic": "logistic_regression",
        "embedding_svm": "linear_svm",
    }
    for alias, algo in embedding_aliases.items():
        if algo in EMBEDDING_ALGORITHMS:
            register_plugin(
                "classifier_engine",
                alias,
                lambda algorithm=algo: algorithm,
                metadata={"algorithm": algo, "family": "embedding", "alias": alias},
            )

    register_plugin(
        "classifier_engine",
        "transformer",
        lambda: "transformer",
        metadata={"family": "transformer", "optional": True},
    )

    for provider_name in ("hashing", "unavailable", "sentence_transformers"):
        register_plugin(
            "embedding_provider",
            provider_name,
            lambda name=provider_name, **kwargs: get_embedding_provider(name, **kwargs),
            metadata={"provider": provider_name},
        )

    from backend.modules.text_research.infrastructure.pipeline_compiler import BASE_STAGES

    def _analysis_stage_factory(analysis_type: str):
        def _resolve(spec_dict: dict[str, Any]) -> str:
            from backend.modules.text_research.domain.analysis_specification import (
                AnalysisSpecification,
            )

            spec = AnalysisSpecification.model_validate(spec_dict)
            plan = compile_plan(spec)
            return plan.stages[len(BASE_STAGES)]

        return _resolve

    for analysis_type in sorted(ANALYSIS_TYPES):
        register_plugin(
            "analysis",
            analysis_type,
            _analysis_stage_factory(analysis_type),
            metadata={"analysis_type": analysis_type},
        )

    _BUILTINS_REGISTERED = True


def ensure_builtins_registered() -> None:
    """Lazy entry point used by package import."""
    register_builtins()
