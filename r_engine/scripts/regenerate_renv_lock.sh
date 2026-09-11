#!/usr/bin/env bash
# Regenerate r_engine/renv.lock inside rocker/r-ver:4.4.0 (same release as production).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${R_RENV_IMAGE:-rocker/r-ver:4.4.0}"

echo "Using image: ${IMAGE}"
rm -rf "${ROOT}/r_engine/renv/library" "${ROOT}/r_engine/renv/staging" "${ROOT}/r_engine/renv/cellar" 2>/dev/null || true

docker run --rm \
  -e LIBARROW_BINARY=true \
  -v "${ROOT}/r_engine:/opt/r_engine" \
  -w /opt/r_engine \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    apt-get update
    apt-get install -y --no-install-recommends \
      libuv1-dev cmake \
      libcurl4-openssl-dev libssl-dev libxml2-dev \
      libfontconfig1-dev libfreetype6-dev libharfbuzz-dev libfribidi-dev \
      libpng-dev libtiff5-dev libjpeg-dev pkg-config
    rm -rf /var/lib/apt/lists/*
    # Prove system libuv is visible to pkg-config before R builds fs.
    pkg-config --exists libuv && pkg-config --modversion libuv
    Rscript /opt/r_engine/scripts/regenerate_renv_lock.R
  '

echo "Done. Review r_engine/renv.lock and commit with activate scaffolding if changed."
