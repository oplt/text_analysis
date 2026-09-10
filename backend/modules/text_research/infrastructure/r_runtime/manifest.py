"""Versioned, language-neutral R job manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

R_MANIFEST_SCHEMA_VERSION = "1.0"


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write only the fixed manifest schema owned by this repository."""
    if payload.get("schema_version") != R_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported R manifest schema version")
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
