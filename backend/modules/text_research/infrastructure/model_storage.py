"""Artifact storage for trained vectorizers/classifiers/topic models.

Artifacts (fitted vectorizers, classifiers, topic models) are persisted with
joblib under a project/run-scoped directory tree so that:

    - predictions can reload the exact fitted vectorizer used at train time
      (never refit at prediction time), and
    - runs remain reproducible/inspectable after the process exits.

The artifact root resolves in this order:
    1. ``RESEARCH_ARTIFACT_ROOT`` environment variable, if set.
    2. ``backend.core.config.settings.RESEARCH_ARTIFACT_ROOT``, if configured
       and importable (best-effort; failures fall through silently so this
       module has no hard dependency on the full application settings stack).
    3. ``backend/.research_artifacts`` (relative to the backend package root).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib


def _default_artifact_root() -> Path:
    env_override = os.environ.get("RESEARCH_ARTIFACT_ROOT")
    if env_override:
        return Path(env_override)

    try:
        from backend.core.config import settings

        configured = getattr(settings, "RESEARCH_ARTIFACT_ROOT", "") or ""
        if configured:
            return Path(configured)
    except Exception:
        # Settings may be unavailable/unconfigured (e.g. missing .env in a
        # test sandbox); fall back to the on-disk default below.
        pass

    # backend/modules/text_research/infrastructure/model_storage.py -> backend/
    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / ".research_artifacts"


#: Resolved once at import time; override via the RESEARCH_ARTIFACT_ROOT
#: environment variable if the process needs a different root after import
#: (tests may also monkeypatch this module attribute directly).
ARTIFACT_ROOT: Path = _default_artifact_root()


def ensure_artifact_dir(project_id: str, run_id: str) -> Path:
    """Create (if needed) and return the artifact directory for a run."""
    directory = ARTIFACT_ROOT / str(project_id) / str(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_joblib(obj: Any, path: str | Path) -> Path:
    """Persist ``obj`` to ``path`` with joblib, creating parent dirs as needed."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, resolved)
    return resolved


def load_joblib(path: str | Path) -> Any:
    """Load a joblib-serialized artifact from ``path``."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Artifact not found: {resolved}")
    return joblib.load(resolved)


def save_artifact(obj: Any, *, category: str) -> str:
    """Persist ``obj`` under ``ARTIFACT_ROOT/<category>/<uuid>.joblib`` and
    return the resulting path as a string (suitable for storing in a
    `*_artifact_path` column). Each call generates a fresh, unique path so
    retraining a model never clobbers a previously persisted artifact still
    referenced by an existing `TrainedModel`/`AnalysisRun` row."""
    directory = ARTIFACT_ROOT / category
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid4()}.joblib"
    save_joblib(obj, path)
    return str(path)


def load_artifact(path: str | Path) -> Any:
    """Load a previously `save_artifact`-persisted artifact."""
    return load_joblib(path)


def delete_artifact(path: str | Path) -> None:
    """Best-effort delete of a previously persisted artifact file."""
    resolved = Path(path)
    if resolved.exists():
        resolved.unlink()
