"""Serialize canonical prepared corpora into R's bounded job directory."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from backend.core.config import settings
from backend.modules.text_research.domain.analysis_result import (
    build_analysis_identity,
    build_scientific_inputs,
)
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact
from backend.modules.text_research.infrastructure.parquet_artifacts import (
    DEFAULT_PARQUET_ROW_BATCH,
    parquet_available,
    save_token_table,
    save_unit_table,
)
from backend.modules.text_research.infrastructure.r_runtime.manifest import (
    R_MANIFEST_SCHEMA_VERSION,
    write_manifest,
)


@dataclass(frozen=True)
class RJobBundle:
    workdir: Path
    manifest_path: Path
    result_path: Path
    analysis_run_id: str


def serialize_r_job(
    *,
    specification: AnalysisSpecification,
    prepared: PreparedCorpusArtifact,
    comparison_prepared: PreparedCorpusArtifact | None = None,
    run_id: str,
    engine_version: str,
    parquet_batch_size: int = DEFAULT_PARQUET_ROW_BATCH,
) -> RJobBundle:
    """Create tabular R inputs; no user field is ever interpreted as code."""
    if not parquet_available():
        raise RuntimeError("R execution requires the installed pyarrow parquet dependency")
    root = Path(settings.RESEARCH_R_WORK_DIR).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="job-", dir=root))
    (workdir / "artifacts").mkdir(parents=True, exist_ok=True)
    units_path = workdir / "units.parquet"
    tokens_path = workdir / "tokens.parquet"
    metadata_path = workdir / "metadata.parquet"
    save_unit_table(
        units_path,
        columns={
            "unit_id": list(prepared.unit_ids),
            "document_id": list(prepared.document_ids),
            "original_text": list(prepared.original_units),
            "cleaned_text": list(prepared.cleaned_units),
        },
        chunk_size=parquet_batch_size,
    )
    # Token rows can be millions — stream bounded batches, never full column lists.
    save_token_table(
        tokens_path,
        prepared.unit_ids,
        prepared.token_sequences,
        batch_size=parquet_batch_size,
    )
    save_unit_table(
        metadata_path,
        (
            {"unit_id": unit_id, **prepared.metadata_by_unit.get(unit_id, {})}
            for unit_id in prepared.unit_ids
        ),
        chunk_size=parquet_batch_size,
    )
    result_path = workdir / "result.json"
    manifest_path = workdir / "manifest.json"
    inputs = {
        "units": units_path.name,
        "tokens": tokens_path.name,
        "metadata": metadata_path.name,
    }
    if comparison_prepared is not None:
        units_b_path = workdir / "units_b.parquet"
        tokens_b_path = workdir / "tokens_b.parquet"
        save_unit_table(
            units_b_path,
            columns={
                "unit_id": list(comparison_prepared.unit_ids),
                "document_id": list(comparison_prepared.document_ids),
            },
            chunk_size=parquet_batch_size,
        )
        save_token_table(
            tokens_b_path,
            comparison_prepared.unit_ids,
            comparison_prepared.token_sequences,
            batch_size=parquet_batch_size,
        )
        inputs.update({"units_b": units_b_path.name, "tokens_b": tokens_b_path.name})
    write_manifest(
        manifest_path,
        {
            "schema_version": R_MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "analysis": {
                "type": specification.analysis.type,
                "parameters": specification.analysis.parameters,
            },
            "feature_extraction": specification.feature_extraction.model_dump(mode="json"),
            "model": specification.model.model_dump(mode="json") if specification.model else None,
            "validation": (
                specification.validation.model_dump(mode="json")
                if specification.validation
                else None
            ),
            "random_seed": specification.random_seed,
            "identity": build_analysis_identity(
                spec_hash=specification.spec_hash(),
                engine_name="r",
                engine_version=engine_version,
                inputs=build_scientific_inputs(
                    target_corpus_checksum=prepared.corpus_checksum,
                    target_pipeline_checksum=prepared.pipeline_checksum,
                    reference_corpus_checksum=(
                        comparison_prepared.corpus_checksum
                        if comparison_prepared is not None
                        else None
                    ),
                    reference_pipeline_checksum=(
                        comparison_prepared.pipeline_checksum
                        if comparison_prepared is not None
                        else None
                    ),
                ),
            ).model_dump(mode="json"),
            "inputs": inputs,
            "output": {"result": result_path.name, "artifacts_directory": "artifacts"},
        },
    )
    return RJobBundle(
        workdir=workdir,
        manifest_path=manifest_path,
        result_path=result_path,
        analysis_run_id=run_id,
    )
