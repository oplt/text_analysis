"""Unified provenance / reproducibility payloads for runs and artifacts (§23 / Phase 13).

Every scientifically meaningful analysis run should capture enough identity to
audit or re-execute the same computation without relying on mutable names alone:
git SHA, application/Python versions, lockfile checksum, docker digest, corpus
snapshot id/hash, campaign/codebook identity, unit type + filters, preprocessing
profile id/config/hash, feature + selection config, seeds, algorithm/hyperparams,
validation strategy + partition hashes, model and artifact checksums, library
versions, and run lifecycle timestamps (surfaced via get_provenance).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.infrastructure.pipeline_compiler import ENGINE_VERSION
from backend.modules.text_research.infrastructure.preprocessing import (
    PREPROCESSING_IMPLEMENTATION,
    PREPROCESSING_IMPLEMENTATION_VERSION,
    describe_implementation,
)

PROVENANCE_SCHEMA_VERSION = "2"


def stable_content_hash(value: Any) -> str:
    """SHA-256 over canonical JSON (sorted keys, compact separators)."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def partition_hash(ids: list[Any] | tuple[Any, ...] | None) -> str | None:
    """Reproducible hash of a train/val/test partition (sorted string ids)."""
    if not ids:
        return None
    normalized = sorted(str(item) for item in ids)
    return stable_content_hash(normalized)


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


def application_version() -> str | None:
    env = os.environ.get("APP_VERSION") or os.environ.get("APPLICATION_VERSION")
    if env and env.strip():
        return env.strip()
    try:
        from importlib import metadata

        for dist_name in ("app-backend", "backend", "text-analysis"):
            try:
                return metadata.version(dist_name)
            except metadata.PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001
        pass
    try:
        from backend.core.config import settings

        configured = getattr(settings, "APP_VERSION", None) or getattr(
            settings, "APPLICATION_VERSION", None
        )
        if configured:
            return str(configured)
    except Exception:  # noqa: BLE001
        pass
    return None


def python_version() -> str:
    return (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )


def package_lock_checksum(*, search_roots: list[Path] | None = None) -> dict[str, Any] | None:
    """Hash the first dependency lockfile found (uv.lock preferred).

    Returns ``None`` when no lockfile exists. Phase 14 commits ``backend/uv.lock``;
    CI installs with ``uv sync --frozen``.
    """
    roots = list(search_roots or [])
    if not roots:
        here = Path(__file__).resolve()
        # …/backend/modules/text_research/infrastructure/provenance.py → repo root
        repo_root = here.parents[4] if len(here.parents) >= 5 else Path.cwd()
        roots = [repo_root, repo_root / "backend", Path.cwd(), Path.cwd() / "backend"]

    names = ("uv.lock", "poetry.lock", "requirements.lock")
    seen: set[Path] = set()
    for root in roots:
        try:
            root = root.resolve()
        except Exception:  # noqa: BLE001
            continue
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        for name in names:
            path = root / name
            try:
                if path.is_file():
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    return {
                        "path": name,
                        "sha256": digest,
                        "bytes": path.stat().st_size,
                    }
            except Exception:  # noqa: BLE001
                continue
    return None


def runtime_environment() -> dict[str, Any]:
    return {
        "package_versions": library_versions(),
        "library_versions": library_versions(),
        "git_commit": git_commit_sha(),
        "application_version": application_version(),
        "python_version": python_version(),
        "package_lock_checksum": package_lock_checksum(),
        "container_image_digest": container_image_digest(),
        "docker_image_digest": container_image_digest(),
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
    config_dict = dict(config or {})
    return {
        "id": profile_id,
        "name": name,
        "version": version,
        "config": config_dict,
        "config_hash": stable_content_hash(config_dict) if config_dict else None,
    }


def _scientific_fields_from_spec(
    spec_obj: AnalysisSpecification | None,
) -> dict[str, Any]:
    if spec_obj is None:
        return {}
    feature = spec_obj.feature_extraction.model_dump(mode="json")
    selection = feature.get("selection")
    model = spec_obj.model.model_dump(mode="json") if spec_obj.model else None
    validation = spec_obj.validation.model_dump(mode="json") if spec_obj.validation else None
    return {
        "corpus_snapshot_id": spec_obj.corpus.snapshot_id,
        "unit_type": spec_obj.corpus.unit_type,
        "filters": dict(spec_obj.corpus.filters or {}),
        "feature_configuration": feature,
        "feature_selection_configuration": selection,
        "algorithm": model.get("family") if model else None,
        "hyperparameters": model.get("hyperparameters") if model else None,
        "validation_strategy": validation,
        "random_seed": spec_obj.random_seed,
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
    corpus_snapshot_id: str | None = None,
    corpus_snapshot_hash: str | None = None,
    campaign_id: str | None = None,
    codebook_id: str | None = None,
    codebook_version: str | None = None,
    unit_type: str | None = None,
    filters: dict[str, Any] | None = None,
    feature_configuration: dict[str, Any] | None = None,
    feature_selection_configuration: dict[str, Any] | None = None,
    algorithm: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    validation_strategy: dict[str, Any] | str | None = None,
    split_hashes: dict[str, str | None] | None = None,
    model_artifact_checksum: str | None = None,
    input_artifact_checksums: list[str] | None = None,
    output_artifact_checksums: list[str] | None = None,
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

    from_spec = _scientific_fields_from_spec(spec_obj)

    resolved_seed = random_seed
    if resolved_seed is None:
        resolved_seed = from_spec.get("random_seed")

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

    prep_profile = preprocessing_profile or _profile_snapshot(
        spec_obj.preprocessing.preprocessing_profile_id if spec_obj else None,
        preprocessing_config,
    )
    if prep_profile and "config_hash" not in prep_profile and prep_profile.get("config"):
        prep_profile = {
            **prep_profile,
            "config_hash": stable_content_hash(prep_profile["config"]),
        }

    cleaning = cleaning_profile
    if cleaning and isinstance(cleaning.get("config"), dict) and "config_hash" not in cleaning:
        cleaning = {
            **cleaning,
            "config_hash": stable_content_hash(cleaning["config"]),
        }

    runtime = runtime_environment()
    parents = list(parent_artifact_checksums or [])
    if input_artifact_checksums:
        for checksum in input_artifact_checksums:
            if checksum and checksum not in parents:
                parents.append(checksum)

    snapshot_hash = corpus_snapshot_hash or corpus_checksum or pipeline_checksum

    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        # Runtime / environment
        "git_commit": runtime["git_commit"],
        "application_version": runtime["application_version"],
        "python_version": runtime["python_version"],
        "package_lock_checksum": runtime["package_lock_checksum"],
        "container_image_digest": runtime["container_image_digest"],
        "docker_image_digest": runtime["docker_image_digest"],
        "package_versions": runtime["package_versions"],
        "library_versions": runtime["library_versions"],
        "engine_version": runtime["engine_version"],
        # Corpus / annotation identity
        "corpus_checksum": corpus_checksum,
        "pipeline_checksum": pipeline_checksum,
        "corpus_snapshot_id": corpus_snapshot_id or from_spec.get("corpus_snapshot_id"),
        "corpus_snapshot_hash": snapshot_hash,
        "campaign_id": campaign_id,
        "codebook_id": codebook_id,
        "codebook_version": codebook_version,
        "unit_type": unit_type or from_spec.get("unit_type"),
        "filters": filters if filters is not None else from_spec.get("filters"),
        # Preprocessing
        "cleaning_profile": cleaning,
        "preprocessing_profile": prep_profile,
        "preprocessing_profile_id": (prep_profile or {}).get("id") if prep_profile else None,
        "preprocessing_config": (prep_profile or {}).get("config") if prep_profile else preprocessing_config,
        "preprocessing_config_hash": (prep_profile or {}).get("config_hash") if prep_profile else (
            stable_content_hash(preprocessing_config) if preprocessing_config else None
        ),
        "preprocessing_implementation": runtime["preprocessing_implementation"],
        "preprocessing_implementation_detail": preprocessing_impl,
        "implementation_version": implementation_version
        or runtime["preprocessing_implementation_version"],
        "nlp_model": resolved_nlp,
        # Features / model
        "feature_configuration": feature_configuration
        or from_spec.get("feature_configuration"),
        "feature_selection_configuration": feature_selection_configuration
        or from_spec.get("feature_selection_configuration"),
        "algorithm": algorithm or from_spec.get("algorithm"),
        "hyperparameters": hyperparameters
        if hyperparameters is not None
        else from_spec.get("hyperparameters"),
        "validation_strategy": validation_strategy
        if validation_strategy is not None
        else from_spec.get("validation_strategy"),
        "split_hashes": split_hashes,
        "model_artifact_checksum": model_artifact_checksum,
        "parent_artifact_checksums": parents,
        "input_artifact_checksums": list(input_artifact_checksums or parents),
        "output_artifact_checksums": list(output_artifact_checksums or []),
        "random_seeds": {
            "analysis": resolved_seed,
            "validation": (
                spec_obj.validation.random_seed
                if spec_obj is not None and spec_obj.validation is not None
                else None
            ),
        },
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
    corpus_snapshot_id: str | None = None,
    corpus_snapshot_hash: str | None = None,
    campaign_id: str | None = None,
    codebook_id: str | None = None,
    codebook_version: str | None = None,
    unit_type: str | None = None,
    filters: dict[str, Any] | None = None,
    feature_configuration: dict[str, Any] | None = None,
    feature_selection_configuration: dict[str, Any] | None = None,
    algorithm: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    validation_strategy: dict[str, Any] | str | None = None,
    split_hashes: dict[str, str | None] | None = None,
    model_artifact_checksum: str | None = None,
    input_artifact_checksums: list[str] | None = None,
    output_artifact_checksums: list[str] | None = None,
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
        corpus_snapshot_id=corpus_snapshot_id or payload.get("snapshot_id") or payload.get("corpus_snapshot_id"),
        corpus_snapshot_hash=corpus_snapshot_hash,
        campaign_id=campaign_id if campaign_id is not None else payload.get("campaign_id"),
        codebook_id=codebook_id if codebook_id is not None else payload.get("codebook_id"),
        codebook_version=codebook_version
        if codebook_version is not None
        else payload.get("codebook_version"),
        unit_type=unit_type if unit_type is not None else payload.get("unit_type"),
        filters=filters if filters is not None else payload.get("filters"),
        feature_configuration=feature_configuration,
        feature_selection_configuration=feature_selection_configuration,
        algorithm=algorithm if algorithm is not None else payload.get("algorithm"),
        hyperparameters=hyperparameters,
        validation_strategy=validation_strategy,
        split_hashes=split_hashes,
        model_artifact_checksum=model_artifact_checksum,
        input_artifact_checksums=input_artifact_checksums,
        output_artifact_checksums=output_artifact_checksums,
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


def merge_completion_provenance(
    parameters: dict[str, Any],
    **updates: Any,
) -> dict[str, Any]:
    """Deep-merge completion-time fields into an existing provenance block."""
    payload = dict(parameters)
    existing = (
        dict(payload["provenance"])
        if isinstance(payload.get("provenance"), dict)
        else {}
    )
    for key, value in updates.items():
        if value is None:
            continue
        if key == "extra" and isinstance(value, dict):
            prior_extra = existing.get("extra") if isinstance(existing.get("extra"), dict) else {}
            existing["extra"] = {**prior_extra, **value}
        elif key in {"output_artifact_checksums", "input_artifact_checksums", "parent_artifact_checksums"}:
            prior = existing.get(key) if isinstance(existing.get(key), list) else []
            merged = list(prior)
            for item in value if isinstance(value, list) else [value]:
                if item and item not in merged:
                    merged.append(item)
            existing[key] = merged
        else:
            existing[key] = value
    if "schema_version" not in existing:
        existing["schema_version"] = PROVENANCE_SCHEMA_VERSION
    payload["provenance"] = existing
    return payload


def enrich_provenance_response(
    *,
    run: Any,
    parameters: dict[str, Any],
    results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Augment stored provenance for API responses with row lifecycle + results."""
    provenance = (
        dict(parameters["provenance"])
        if isinstance(parameters.get("provenance"), dict)
        else {}
    )
    results = results or {}

    provenance.setdefault("created_by", getattr(run, "created_by", None))
    provenance.setdefault(
        "started_at",
        getattr(run, "started_at", None).isoformat()
        if getattr(run, "started_at", None) is not None
        else None,
    )
    provenance.setdefault(
        "completed_at",
        getattr(run, "completed_at", None).isoformat()
        if getattr(run, "completed_at", None) is not None
        else None,
    )
    provenance.setdefault("corpus_id", getattr(run, "corpus_id", None))
    provenance.setdefault("project_id", getattr(run, "project_id", None))
    provenance.setdefault("random_seed", getattr(run, "random_seed", None))

    artifact_meta = results.get("artifact_metadata")
    if isinstance(artifact_meta, dict) and not provenance.get("model_artifact_checksum"):
        model_meta = artifact_meta.get("model")
        if isinstance(model_meta, dict) and model_meta.get("sha256"):
            provenance["model_artifact_checksum"] = model_meta["sha256"]
        outputs = list(provenance.get("output_artifact_checksums") or [])
        for key in ("model", "vectorizer"):
            meta = artifact_meta.get(key)
            if isinstance(meta, dict) and meta.get("sha256") and meta["sha256"] not in outputs:
                outputs.append(meta["sha256"])
        if outputs:
            provenance["output_artifact_checksums"] = outputs

    if not provenance.get("split_hashes"):
        split_hashes = {
            "train": partition_hash(results.get("train_groups")),
            "validation": partition_hash(results.get("val_groups")),
            "test": partition_hash(results.get("test_groups")),
        }
        if any(split_hashes.values()):
            provenance["split_hashes"] = split_hashes

    for key in (
        "corpus_checksum",
        "pipeline_checksum",
        "analysis_spec_hash",
        "feature_selection",
        "feature_space",
    ):
        if key in results and not provenance.get(key):
            provenance[key] = results[key]

    return provenance


def extract_reproduce_request(
    parameters: dict[str, Any], *, run_type: str, run_id: str
) -> dict[str, Any]:
    """Shape a one-click reproduce payload from persisted run parameters."""
    provenance = (
        parameters.get("provenance") if isinstance(parameters.get("provenance"), dict) else {}
    )
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
        "corpus_snapshot_id": provenance.get("corpus_snapshot_id"),
        "corpus_snapshot_hash": provenance.get("corpus_snapshot_hash"),
        "random_seeds": provenance.get("random_seeds")
        or {"analysis": parameters.get("random_seed")},
        "git_commit": provenance.get("git_commit"),
        "application_version": provenance.get("application_version"),
        "package_lock_checksum": provenance.get("package_lock_checksum"),
        "rerun_path": f"/api/v1/research/runs/{run_id}/rerun",
    }
