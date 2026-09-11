"""Secure collection of R job artifacts into durable application storage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.core.config import settings
from backend.modules.text_research.domain.analysis_result import AnalysisResult
from backend.modules.text_research.infrastructure.artifact_store import (
    default_artifact_store,
)
from backend.modules.text_research.infrastructure.r_runtime.serializer import RJobBundle

logger = logging.getLogger(__name__)

_MIME_BY_SUFFIX = {
    ".parquet": "application/vnd.apache.parquet",
    ".json": "application/json",
    ".csv": "text/csv",
    ".rds": "application/octet-stream",
    ".txt": "text/plain",
}


class RArtifactSecurityError(ValueError):
    """Raised when an R job artifact violates collection security rules."""


def _max_artifact_bytes() -> int:
    return int(settings.RESEARCH_R_MAX_ARTIFACT_MB) * 1024 * 1024


def _max_artifacts_total_bytes() -> int:
    return int(settings.RESEARCH_R_MAX_ARTIFACTS_TOTAL_MB) * 1024 * 1024


def _max_artifact_count() -> int:
    return int(settings.RESEARCH_R_MAX_ARTIFACTS)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _assert_safe_artifact_file(path: Path, *, artifacts_root: Path) -> None:
    if path.is_symlink():
        raise RArtifactSecurityError(f"refusing symlink artifact {path.name!r}")
    for parent in path.parents:
        if parent == artifacts_root or parent == artifacts_root.parent:
            break
        if parent.is_symlink():
            raise RArtifactSecurityError(f"refusing artifact under symlink parent {path.name!r}")
    if not path.is_file():
        raise RArtifactSecurityError(f"artifact is not a regular file: {path.name!r}")
    if not _is_within(path, artifacts_root):
        raise RArtifactSecurityError(f"path traversal rejected for artifact {path.name!r}")


def _hint_from_result(result: AnalysisResult, name: str) -> dict[str, Any]:
    for item in result.artifacts:
        if isinstance(item, dict) and item.get("name") == name:
            return dict(item)
    return {}


def _content_type(name: str, hint: dict[str, Any]) -> str:
    fmt = str(hint.get("format") or "")
    if "parquet" in fmt:
        return "application/vnd.apache.parquet"
    suffix = Path(name).suffix.lower()
    return _MIME_BY_SUFFIX.get(suffix, "application/octet-stream")


def collect_and_persist_r_artifacts(
    bundle: RJobBundle,
    result: AnalysisResult,
    *,
    store=default_artifact_store,
) -> AnalysisResult:
    """Move R job files into durable storage; never expose worker absolute paths."""
    artifacts_dir = bundle.workdir / "artifacts"
    if not artifacts_dir.is_dir():
        return result

    artifacts_root = artifacts_dir.resolve()
    candidates: list[Path] = []
    for path in sorted(artifacts_dir.rglob("*")):
        if path.is_dir():
            continue
        _assert_safe_artifact_file(path, artifacts_root=artifacts_root)
        candidates.append(path)

    if len(candidates) > _max_artifact_count():
        raise RArtifactSecurityError(
            f"R job produced too many artifacts ({len(candidates)} > {_max_artifact_count()})"
        )

    max_file = _max_artifact_bytes()
    max_total = _max_artifacts_total_bytes()
    total_bytes = 0
    run_id = bundle.analysis_run_id
    persisted: list[dict[str, Any]] = []

    for path in candidates:
        size = path.stat().st_size
        if size > max_file:
            raise RArtifactSecurityError(
                f"R artifact {path.name!r} exceeds per-file limit ({size} > {max_file} bytes)"
            )
        total_bytes += size
        if total_bytes > max_total:
            raise RArtifactSecurityError(
                f"R artifacts exceed aggregate size limit ({total_bytes} > {max_total} bytes)"
            )

        relative = path.relative_to(artifacts_root)
        name = relative.as_posix()
        hint = _hint_from_result(result, Path(name).name)
        role = str(hint.get("kind") or Path(name).stem)
        content_type = _content_type(name, hint)
        descriptor = store.put_file(
            "r_artifact",
            path,
            filename=Path(name).name,
            producing_run_id=run_id,
            implementation_version=result.identity.engine_version,
            content_type=content_type,
            metadata={
                "role": role,
                "name": name,
                "format": hint.get("format") or Path(name).suffix.lstrip(".") or "bin",
                "run_id": run_id,
                "dimensions": hint.get("shape") or hint.get("dimensions"),
                "nnz": hint.get("nnz"),
                "row_count": hint.get("row_count"),
            },
        )
        entry = {
            "artifact_id": descriptor.artifact_id,
            "kind": "r_artifact",
            "role": role,
            "name": name,
            "format": descriptor.metadata.get("format"),
            "content_type": content_type,
            "sha256": descriptor.checksum,
            "bytes": descriptor.metadata.get("bytes"),
            "run_id": run_id,
            "storage_backend": descriptor.metadata.get("storage_backend"),
            "storage_key": descriptor.metadata.get("storage_key"),
            "object_key": descriptor.metadata.get("object_key"),
        }
        if descriptor.metadata.get("dimensions") is not None:
            entry["dimensions"] = descriptor.metadata.get("dimensions")
        if descriptor.metadata.get("nnz") is not None:
            entry["nnz"] = descriptor.metadata.get("nnz")
        # Never publish worker absolute paths.
        persisted.append(entry)

    enriched = _attach_dfm_matrix_checksum_from_job(result, artifacts_root)
    # Replace ephemeral R-emitted artifact stubs with durable descriptors.
    return enriched.model_copy(update={"artifacts": persisted})


def _attach_dfm_matrix_checksum_from_job(
    result: AnalysisResult,
    artifacts_root: Path,
) -> AnalysisResult:
    """Compute full-matrix checksum from R parquet before workdir cleanup."""
    coo_path = artifacts_root / "dfm_sparse_coo.parquet"
    features_path = artifacts_root / "dfm_features.parquet"
    units_path = artifacts_root / "dfm_units.parquet"
    if not (coo_path.is_file() and features_path.is_file() and units_path.is_file()):
        return result
    if not isinstance(result.results, dict):
        return result

    try:
        import pandas as pd

        from backend.modules.text_research.infrastructure.dfm_matrix_identity import (
            attach_matrix_checksum_fields,
            matrix_checksum_from_coo,
        )

        coo = pd.read_parquet(coo_path)
        features = pd.read_parquet(features_path).sort_values("feature_index")
        units = pd.read_parquet(units_path).sort_values("unit_index")
        feature_names = [str(name) for name in features["feature"].tolist()]
        unit_ids = [str(uid) for uid in units["unit_id"].tolist()]
        value_col = "value" if "value" in coo.columns else "data"
        triples = zip(
            (int(v) for v in coo["row"].tolist()),
            (int(v) for v in coo["col"].tolist()),
            (float(v) for v in coo[value_col].tolist()),
            strict=False,
        )
        checksum, cells = matrix_checksum_from_coo(
            unit_ids=unit_ids,
            feature_names=feature_names,
            triples=triples,
        )
        updated = attach_matrix_checksum_fields(
            dict(result.results), checksum=checksum, cells_compared=cells
        )
        return result.model_copy(update={"results": updated})
    except Exception:
        logger.exception("Failed to attach DFM matrix checksum from R artifacts")
        return result
