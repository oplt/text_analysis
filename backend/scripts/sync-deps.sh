#!/usr/bin/env bash
# Install backend deps from the committed uv.lock (deterministic).
# Usage (from repo root or backend/):
#   ./backend/scripts/sync-deps.sh
#   ./backend/scripts/sync-deps.sh --extra nlp
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -f uv.lock ]]; then
  echo "missing uv.lock next to pyproject.toml in ${ROOT}" >&2
  exit 1
fi

# --frozen: never resolve; fail if lock is out of date with pyproject.toml
# --extra dev: pytest/ruff/honcho used by CI and local checks
exec uv sync --frozen --extra dev "$@"
