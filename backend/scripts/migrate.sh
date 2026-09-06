#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing backend virtualenv at backend/.venv" >&2
  echo "Create it first, then rerun this script." >&2
  exit 1
fi

"$PYTHON" scripts/check_db.py
exec "$PYTHON" -m alembic "$@"
