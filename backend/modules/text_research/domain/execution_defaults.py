"""Pure execution defaults shared by research task planning and adapters."""

from __future__ import annotations

import hashlib

ENGINE_VERSION = "text_research.pipeline/1"

DEFAULT_RETRY_POLICIES: dict[str, dict[str, int]] = {
    "research_light": {"max_retries": 3, "countdown": 5},
    "research_cpu": {"max_retries": 2, "countdown": 30},
    "research_io": {"max_retries": 3, "countdown": 10},
    "research_nlp": {"max_retries": 2, "countdown": 60},
    "research_memory": {"max_retries": 1, "countdown": 60},
    "research_gpu": {"max_retries": 1, "countdown": 120},
}

DEFAULT_TIMEOUT_SECONDS: dict[str, int | None] = {
    "research_light": 600,
    "research_cpu": 3600,
    "research_io": 1800,
    "research_nlp": 7200,
    "research_memory": 7200,
    "research_gpu": 14400,
}


def retry_policy_for(resource_class: str) -> dict[str, int]:
    """Return Celery-compatible retry kwargs for a resource class."""
    return dict(DEFAULT_RETRY_POLICIES.get(resource_class, DEFAULT_RETRY_POLICIES["research_cpu"]))


def timeout_for(resource_class: str) -> int | None:
    """Return soft timeout seconds for a resource class, or None when unset."""
    return DEFAULT_TIMEOUT_SECONDS.get(resource_class)


def computation_identity(
    spec_hash: str,
    corpus_snapshot_hash: str,
    engine_version: str = ENGINE_VERSION,
) -> str:
    """Return the stable identity for a prepared-corpus computation artifact."""
    payload = f"{spec_hash}:{corpus_snapshot_hash}:{engine_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
