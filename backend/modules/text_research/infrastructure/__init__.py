"""Pure-Python/scikit-learn computational engines for text research.

These modules contain no database or FastAPI dependencies. They implement the
statistical and machine-learning primitives used by the application/service
layer (segmentation, preprocessing, reliability statistics, quantitative text
analysis, classifiers, topic models, and artifact storage).

Kept intentionally free of heavy imports at package-import time; import the
specific submodule you need (e.g. ``backend.modules.text_research.infrastructure.segmentation``).
"""

__all__ = [
    "segmentation",
    "preprocessing",
    "reliability",
    "quantitative",
    "classifiers",
    "topic_models",
    "model_storage",
    "canonical_text",
    "ingestion_qa",
    "document_cleaning",
    "language_processing",
    "weighting",
    "kwic",
    "dictionary_matcher",
    "keyness",
    "collocation",
    "association_network",
    "similarity",
    "duplicate_detection",
    "clustering",
    "dimensionality",
    "readability",
    "embeddings",
    "ner",
    "nlp_preprocessing",
    "linguistic_features",
    "statistical_modeling",
    "measurement_validation",
    "sampling",
    "prepared_corpus_builder",
    "artifact_registry",
    "pipeline_compiler",
    "split_planner",
    "language_detection",
    "text_transforms",
    "error_analysis",
    "embedding_classifier",
    "topic_engines",
    "spacy_engine",
    "execution_policy",
    "plugin_registry",
    "stage_cache",
    "stage_runner",
    "out_of_core",
    "parquet_artifacts",
    "drift_monitoring",
]


def __getattr__(name: str):
    if name == "plugin_registry":
        from backend.modules.text_research.infrastructure import plugin_registry as mod

        mod.ensure_builtins_registered()
        return mod
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
