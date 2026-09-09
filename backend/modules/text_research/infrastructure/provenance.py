"""Unified provenance / reproducibility payloads for runs and artifacts (§23).

Every analysis run and exportable manifest should capture enough identity to
audit or re-execute the same computation: corpus/pipeline checksums, parent
artifacts, cleaning + preprocessing profiles, seeds, library versions, git
commit, optional container digest, NLP model identity, implementation
version, and the exact AnalysisSpecification.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from typing import Any

from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.infrastructure.pipeline_compiler import ENGINE_VERSION
from backend.modules.text_research.infrastructure.preprocessing import (
    PREPROCESSING_IMPLEMENTATION,
    PREPROCESSING_IMPLEMENTATION_VERSION,
    describe_implementation,
)

PROVENANCE_SCHEMA_VERSION = "1"


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in (
        "numpy",
        "scipy",
        "sklearn",
        "pandas",
        "pyarrow",
        "ftfy",
        "regex",
        "snowballstemmer",
        "simplemma",
        "statsmodels",
        "spacy",
        "joblib",
    ):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            versions[name] = getattr(mod, "__version__", None)
        except Exception:  # noqa: BLE001
            versions[name] = None
    return versions


def git_commit_sha() -> str | None:
    env_sha = (
        os.environ.get("GIT_COMMIT")
        or os.environ.get("SOURCE_COMMIT")
        or os.environ.get("GITHUB_SHA")
        or os.environ.get("COMMIT_SHA")
    )
    if env_sha:
        return env_sha.strip() or None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None
    return None


def container_image_digest() -> str | None:
    """Return container/image digest when operators set it in the environment."""
    try:
        from backend.core.config import settings

        configured = getattr(settings, "RESEARCH_IMAGE_DIGEST", "") or ""
        if configured.strip():
            return configured.strip()
    except Exception:  # noqa: BLE001
        pass
    for key in (
        "RESEARCH_IMAGE_DIGEST",
        "CONTAINER_IMAGE_DIGEST",
        "IMAGE_DIGEST",
        "OCI_IMAGE_DIGEST",
    ):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def runtime_environment() -> dict[str, Any]:
    return {
        "package_versions": library_versions(),
        "git_commit": git_commit_sha(),
        "container_image_digest": container_image_digest(),
        "engine_version": ENGINE_VERSION,
        "preprocessing_implementation": PREPROCESSING_IMPLEMENTATION,
        "preprocessing_implementation_version": PREPROCESSING_IMPLEMENTATION_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
    }


def _profile_snapshot(
    profile_id: str | None,
    config: dict[str, Any] | None,
    *,
    name: str | None = None,
    version: int | str | None = None,
) -> dict[str, Any] | None:
    if not profile_id and not config:
        return None
    return {
        "id": profile_id,
        "name": name,
        "version": version,
        "config": dict(config or {}),
    }


def build_run_provenance(
    *,
    spec: AnalysisSpecification | dict[str, Any] | None = None,
    corpus_checksum: str | None = None,
    pipeline_checksum: str | None = None,
    parent_artifact_checksums: list[str] | None = None,
    cleaning_profile: dict[str, Any] | None = None,
    preprocessing_profile: dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
    random_seed: int | None = None,
    nlp_model: dict[str, Any] | None = None,
    implementation_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the canonical provenance block for a run or artifact manifest."""
    spec_obj: AnalysisSpecification | None
    if isinstance(spec, AnalysisSpecification):
        spec_obj = spec.normalize()
    elif isinstance(spec, dict):
        spec_obj = AnalysisSpecification.model_validate(spec).normalize()
    else:
        spec_obj = None

    resolved_seed = random_seed
    if resolved_seed is None and spec_obj is not None:
        resolved_seed = spec_obj.random_seed

    preprocessing_impl = None
    if preprocessing_config is not None:
        preprocessing_impl = describe_implementation(preprocessing_config)
    elif preprocessing_profile and isinstance(preprocessing_profile.get("config"), dict):
        preprocessing_impl = describe_implementation(preprocessing_profile["config"])

    resolved_nlp = nlp_model
    if resolved_nlp is None and preprocessing_impl:
        resolved_nlp = {
            "spacy_model": preprocessing_impl.get("spacy_model"),
            "spacy_available": preprocessing_impl.get("spacy_available"),
            "lemmatizer": preprocessing_impl.get("lemmatizer"),
            "stemmer": preprocessing_impl.get("stemmer"),
            "package_versions": preprocessing_impl.get("package_versions"),
        }

    runtime = runtime_environment()
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "corpus_checksum": corpus_checksum,
        "pipeline_checksum": pipeline_checksum,
        "parent_artifact_checksums": list(parent_artifact_checksums or []),
        "cleaning_profile": cleaning_profile,
        "preprocessing_profile": preprocessing_profile
        or _profile_snapshot(
            spec_obj.preprocessing.preprocessing_profile_id if spec_obj else None,
            preprocessing_config,
        ),
        "random_seeds": {
            "analysis": resolved_seed,
            "validation": (
                spec_obj.validation.random_seed
                if spec_obj is not None and spec_obj.validation is not None
                else None
            ),
        },
        "package_versions": runtime["package_versions"],
        "git_commit": runtime["git_commit"],
        "container_image_digest": runtime["container_image_digest"],
        "nlp_model": resolved_nlp,
        "implementation_version": implementation_version
        or runtime["preprocessing_implementation_version"],
        "engine_version": runtime["engine_version"],
        "preprocessing_implementation": runtime["preprocessing_implementation"],
        "preprocessing_implementation_detail": preprocessing_impl,
        "analysis_specification": spec_obj.model_dump(mode="json") if spec_obj else None,
        "analysis_spec_hash": spec_obj.spec_hash() if spec_obj else None,
        "captured_at": runtime["captured_at"],
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def attach_provenance(
    parameters: dict[str, Any],
    *,
    spec: AnalysisSpecification | dict[str, Any] | None = None,
    corpus_checksum: str | None = None,
    pipeline_checksum: str | None = None,
    parent_artifact_checksums: list[str] | None = None,
    cleaning_profile: dict[str, Any] | None = None,
    preprocessing_profile: dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
    random_seed: int | None = None,
    nlp_model: dict[str, Any] | None = None,
    implementation_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return parameters with a full ``provenance`` block and identity fields."""
    payload = dict(parameters)
    resolved_corpus = corpus_checksum or payload.get("corpus_checksum")
    resolved_pipeline = pipeline_checksum or payload.get("pipeline_checksum")
    parents = parent_artifact_checksums
    if parents is None:
        existing = payload.get("parent_artifact_checksums")
        parents = list(existing) if isinstance(existing, list) else None

    provenance = build_run_provenance(
        spec=spec,
        corpus_checksum=resolved_corpus if isinstance(resolved_corpus, str) else None,
        pipeline_checksum=resolved_pipeline if isinstance(resolved_pipeline, str) else None,
        parent_artifact_checksums=parents,
        cleaning_profile=cleaning_profile or payload.get("cleaning_profile"),
        preprocessing_profile=preprocessing_profile or payload.get("preprocessing_profile"),
        preprocessing_config=preprocessing_config or payload.get("preprocessing_config"),
        random_seed=random_seed if random_seed is not None else payload.get("random_seed"),
        nlp_model=nlp_model,
        implementation_version=implementation_version,
        extra=extra,
    )
    payload["provenance"] = provenance
    if provenance.get("analysis_spec_hash"):
        payload["analysis_spec_hash"] = provenance["analysis_spec_hash"]
    payload["engine_version"] = provenance["engine_version"]
    if provenance.get("corpus_checksum"):
        payload["corpus_checksum"] = provenance["corpus_checksum"]
    if provenance.get("pipeline_checksum"):
        payload["pipeline_checksum"] = provenance["pipeline_checksum"]
    if provenance.get("analysis_specification"):
        payload["analysis_specification"] = provenance["analysis_specification"]
    return payload


def extract_reproduce_request(parameters: dict[str, Any], *, run_type: str, run_id: str) -> dict[str, Any]:
    """Shape a one-click reproduce payload from persisted run parameters."""
    provenance = parameters.get("provenance") if isinstance(parameters.get("provenance"), dict) else {}
    return {
        "source_run_id": run_id,
        "run_type": run_type,
        "parameters": parameters,
        "analysis_specification": provenance.get("analysis_specification")
        or parameters.get("analysis_specification"),
        "analysis_spec_hash": provenance.get("analysis_spec_hash")
        or parameters.get("analysis_spec_hash"),
        "corpus_checksum": provenance.get("corpus_checksum") or parameters.get("corpus_checksum"),
        "pipeline_checksum": provenance.get("pipeline_checksum")
        or parameters.get("pipeline_checksum"),
        "random_seeds": provenance.get("random_seeds")
        or {"analysis": parameters.get("random_seed")},
        "rerun_path": f"/api/v1/research/runs/{run_id}/rerun",
    }
