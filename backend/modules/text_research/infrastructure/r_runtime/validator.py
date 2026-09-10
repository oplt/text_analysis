"""Strict validation for untrusted data crossing the R subprocess boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.modules.text_research.domain.analysis_result import AnalysisResult
from backend.modules.text_research.domain.exceptions import RArtifactTooLarge, RInvalidResult

MAX_RESULT_SCHEMA_VERSION = "1.0"


def validate_r_result(
    path: Path,
    *,
    expected_analysis_type: str,
    expected_identity: dict[str, Any],
    max_bytes: int,
) -> AnalysisResult:
    if not path.is_file():
        raise RInvalidResult("R engine did not produce a result")
    if path.stat().st_size > max_bytes:
        raise RArtifactTooLarge("R result exceeds configured size limit")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = AnalysisResult.model_validate(payload)
    except Exception as exc:  # validation errors must not leak R internals
        raise RInvalidResult("R engine returned an invalid result") from exc
    if result.schema_version != MAX_RESULT_SCHEMA_VERSION:
        raise RInvalidResult("unsupported R result schema version")
    if result.analysis_type != expected_analysis_type:
        raise RInvalidResult("R result analysis type does not match request")
    identity = result.identity.model_dump(mode="json")
    for key, value in expected_identity.items():
        if identity.get(key) != value:
            raise RInvalidResult("R result identity does not match request")
    if not isinstance(result.results, dict) or not isinstance(result.artifacts, list):
        raise RInvalidResult("R result has invalid result fields")
    return result
