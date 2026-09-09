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
    "linguistic_features",
    "statistical_modeling",
    "measurement_validation",
    "sampling",
]
