"""Pluggable registry for research analysis engines and transforms.

Execution engines (Python / R) are resolved through
:func:`resolve_execution_engine` — the single authoritative source for runtime
name, implementation family, version, supported analyses, and factory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ExecutionEngineDescriptor:
    """Authoritative metadata + factory for one scientific execution runtime."""

    runtime: str
    implementation: str
    implementation_version: str
    supported_analyses: frozenset[str]
    factory: PluginFactory

    def create(self) -> Any:
        return self.factory()

    def supports(self, analysis_type: str) -> bool:
        return analysis_type in self.supported_analyses


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


def get_plugin_metadata(kind: str, name: str) -> dict[str, Any]:
    """Return registration metadata for a plugin (empty dict when unset)."""
    bucket = _REGISTRY.get(kind)
    if bucket is None or name not in bucket:
        raise KeyError(f"unknown plugin {kind!r}:{name!r}")
    return dict(bucket[name].get("metadata") or {})


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


def _descriptor_from_engine_class(engine_cls: type) -> ExecutionEngineDescriptor:
    analyses = getattr(engine_cls, "supported_analyses", None)
    if analyses is None:
        supported: frozenset[str] = frozenset()
    else:
        supported = frozenset(analyses)
    return ExecutionEngineDescriptor(
        runtime=str(engine_cls.name),
        implementation=str(engine_cls.implementation),
        implementation_version=str(engine_cls.implementation_version),
        supported_analyses=supported,
        factory=engine_cls,
    )


def resolve_execution_engine(runtime: str) -> ExecutionEngineDescriptor:
    """Resolve one execution engine through the registry (authoritative).

    Compiler, StageRunner, capability advertising, and provenance must use this
    (or :func:`list_execution_engines`) rather than hard-coding version strings
    or importing engine classes ad hoc.
    """
    ensure_builtins_registered()
    try:
        factory = get_plugin("execution_engine", runtime)
    except KeyError as exc:
        raise KeyError(f"unknown execution engine runtime: {runtime!r}") from exc
    meta = get_plugin_metadata("execution_engine", runtime)
    # Prefer live class attributes (single source on the engine) when present.
    if hasattr(factory, "name") and hasattr(factory, "implementation_version"):
        return _descriptor_from_engine_class(factory)
    analyses = meta.get("supported_analyses") or ()
    return ExecutionEngineDescriptor(
        runtime=str(meta.get("runtime") or runtime),
        implementation=str(meta.get("implementation") or runtime),
        implementation_version=str(meta.get("implementation_version") or ""),
        supported_analyses=frozenset(analyses),
        factory=factory,
    )


def list_execution_engines() -> list[ExecutionEngineDescriptor]:
    """Return descriptors for every registered execution runtime (stable order)."""
    ensure_builtins_registered()
    names = list_plugins("execution_engine").get("execution_engine") or []
    preferred = ("python", "r")
    ordered = [name for name in preferred if name in names]
    ordered.extend(sorted(name for name in names if name not in preferred))
    return [resolve_execution_engine(name) for name in ordered]


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

    for engine_cls in (PythonAnalysisEngine, RAnalysisEngine):
        descriptor = _descriptor_from_engine_class(engine_cls)
        register_plugin(
            "execution_engine",
            descriptor.runtime,
            descriptor.factory,
            metadata={
                "runtime": descriptor.runtime,
                "implementation": descriptor.implementation,
                "implementation_version": descriptor.implementation_version,
                "supported_analyses": sorted(descriptor.supported_analyses),
            },
        )

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
