# R/quanteda execution engine

## Status

The application now has an optional, controlled R/quanteda execution engine.
Python continues to own FastAPI, authentication, corpus storage, snapshots,
preprocessing, `AnalysisRun` persistence, Celery orchestration, provenance and
artifacts. R is only a subprocess scientific runtime.

The supported R analyses are `frequencies`, count-weighted `dfm`, word-query
`kwic`, user-supplied `dictionary`, `keyness`, and `cooccurrence`. Python
remains the default runtime. Topic models and other analyses remain
Python-only until an equivalent R method and parity contract are added.

## Engine registry (authoritative resolution)

Internal consumers must resolve Python/R engines through
`plugin_registry.resolve_execution_engine` (or `list_execution_engines`), which
returns runtime name, implementation family, implementation version, supported
analyses, and the engine factory. Version strings such as `r-quanteda-2` live on
the engine class / `execution_defaults` once; compiler plans, StageRunner,
`/analysis-engines`, R worker heartbeats, and provenance all read them via the
resolver so they cannot drift.

## Data and execution boundary

```text
AnalysisSpecification(engine=r)
  -> canonical PreparedCorpusArtifact (Python preprocessing)
  -> versioned manifest + Parquet units/tokens/metadata
  -> Celery research_r queue
  -> controlled Rscript r_engine/run_analysis.R
  -> validated AnalysisResult
  -> existing AnalysisRun, provenance, artifacts, frontend
```

`preprocessing_mode=standardized` is the only implemented R mode. It consumes
the canonical Python token sequences, so differences in parity tests identify
analysis semantics rather than independent tokenization. Native R
preprocessing is deliberately not implemented.

## Configuration and operations

R is opt-in. With `RESEARCH_R_ENABLED=false` (the default), the application does
not offer R analyses. Set `RESEARCH_R_ENABLED=true` on the **API** to advertise
R in `GET /api/v1/research/analysis-engines` — the API does **not** require a
local `Rscript`. Only the dedicated `research_r` Celery worker must have
`Rscript`, `RESEARCH_R_ENGINE_ENTRYPOINT`, and related runtime settings.

That worker publishes a Redis heartbeat (`research:r_worker:capabilities`).
The capability endpoint exposes both `available` (feature flag) and `ready`
(heartbeat). Configure `RESEARCH_RSCRIPT_PATH`, `RESEARCH_R_ENGINE_ENTRYPOINT`,
`RESEARCH_R_TIMEOUT_SECONDS`, `RESEARCH_R_MAX_OUTPUT_MB`,
`RESEARCH_R_WORK_DIR`, and `RESEARCH_QUEUE_R` on the R worker.

R jobs always use a repository-owned entrypoint; API clients cannot provide R
code, a command, a script path, or package-install requests. The bridge uses
`--vanilla`, no shell, a unique temporary directory, timeout, bounded result
validation, and cleanup. Failures surface as safe R runtime errors rather than
raw environment data or stack traces.

Engine **version** strings (`r-quanteda-2`, Python `ENGINE_VERSION`) are always
resolved server-side from the registered engine. Clients may send
`engine.runtime` and an optional implementation *family* (`quanteda`), never a
version used as provenance/cache identity.

## Reproducibility

The manifest and canonical result contain specification hash, corpus checksum,
pipeline checksum, engine name/version, runtime version and package versions.
Engine selection participates in normalized specification and computation
identity, so Python and R cannot reuse each other's cached results.

`r_engine/renv.lock` is a full scientific lockfile: R 4.4.0 plus the resolved
transitive dependency graph for `quanteda`, `quanteda.textstats`, `arrow`,
`jsonlite`, and `testthat`. Do not hand-edit it. Regenerate with:

```bash
./r_engine/scripts/regenerate_renv_lock.sh
```

### CRAN / repository strategy

* Repository: CRAN via `https://cloud.r-project.org` (recorded in `renv.lock`).
* R release: **4.4.0** everywhere (Dockerfile `FROM rocker/r-ver:4.4.0`, CI
  container, lockfile `R.Version`).
* Arrow binaries: `LIBARROW_BINARY=true` at build time so CI/production avoid
  compiling Arrow from source when binaries exist.
* Restore happens **only at image build** (`renv::restore`). Runtime sets
  `RESEARCH_R_NO_INSTALL=1` and blocks `install.packages`.
* Deployment: pin published worker images by **immutable digest** when the
  registry provides one (for example
  `text-analysis-r-worker@sha256:…`), not only a mutable `:latest` tag.

Run provenance records `r_package_lock_checksum` (SHA-256 of `renv.lock`)
alongside Python `package_lock_checksum`, container digest, and R
`package_versions` from the analysis result runtime block.

## Parity

Parity is defined as same prepared corpus, comparable method, and normalized
result contract—not byte-identical output from unrelated statistical methods.
Use small deterministic fixtures for frequencies, DFM dimensions/vocabulary,
KWIC ordering, user-defined dictionary matches, keyness methods, and
co-occurrence scores. Count DFM and word KWIC deliberately reject unsupported
Python options instead of silently changing scientific methods. R keyness
implements `log_likelihood`, `chi_square`, and `fisher` with optional BH FDR;
R co-occurrence implements `count`, PMI, NPMI, Dice, logDice, and t-score.

## Deployment: dedicated R-capable worker

Build the combined Python+R Celery worker from the repository root:

```bash
cp infra/.env.example infra/.env
# Set JWT_SECRET and storage credentials in infra/.env; do not put them in the image.
docker build -f r_engine/Dockerfile -t text-analysis-r-worker:local .
# or via Compose profile:
docker compose -f infra/docker-compose.yml --profile r up --build worker-r
```

The image CMD starts Celery with `-Q research_r` (not `Rscript --version`).
Set the same database, Redis, artifact-storage, and application settings as the
Python worker, plus `RESEARCH_R_ENABLED=true`, `RESEARCH_RSCRIPT_PATH=Rscript`,
and `RESEARCH_R_ENGINE_ENTRYPOINT=/opt/r_engine/run_analysis.R`. Keep API and
ordinary Python workers with R execution disabled unless they also need to
run the dedicated queue. Local `Procfile.dev` includes a `worker-r` process
that consumes `research_r`.

The container restores R packages solely from `renv.lock` at build time;
production workers must not install R packages at runtime.

### CI: production image, not host-equivalent restore

`.github/workflows/quanteda-parity.yml` builds `r_engine/Dockerfile` (same image
as deployment) and runs smoke checks, `testthat`, and Python↔R parity **inside
that image**. Path filters cover `r_engine/**`, `backend/modules/text_research/**`,
`backend/workers/**`, `backend/core/config.py`, Compose/env, and R-facing
frontend analysis files. Buildx GHA layer cache is keyed by R 4.4.0 +
`renv.lock` + `uv.lock` digests so Arrow/`renv::restore` work is not repeated
from scratch on every PR when locks are unchanged.

The Compose `worker-r` service uses the shared backend environment contract:
database, Redis/Celery, JWT, and object-storage settings are supplied through
`infra/.env`. The service waits for healthy PostgreSQL and Redis dependencies;
it does not contain production secrets in the Dockerfile or image layers.

### Resource isolation (container cgroup)

Enforceable limits for the dedicated R worker are applied at the **container**
layer, not via unsafe `preexec_fn` rlimits inside the multi-threaded Celery
process:

| Control | Compose / env | Default |
| --- | --- | --- |
| Memory | `mem_limit` / `WORKER_R_MEMORY_LIMIT` | `4g` |
| CPU | `cpus` / `WORKER_R_CPUS` | `2.0` |
| PIDs | `pids_limit` / `WORKER_R_PIDS_LIMIT` | `256` |
| Celery concurrency | `--concurrency` / `CELERY_R_CONCURRENCY` | `1` |

Mirror the same values into `RESEARCH_R_WORKER_*` env vars (Compose does this)
so run provenance can stamp operator intent even when cgroup files are opaque.
`provenance.resource_limits` records cgroup observations when available, plus
`execution_spec_policy: advisory`.

`ExecutionSpec.cpu` and `ExecutionSpec.memory_mb` on an analysis specification
are **advisory hints only** — they do not configure cgroups and must not be
treated as guarantees. Subprocess timeout and process-group kill remain in
`r_runtime/runner.py`.

For Kubernetes (or similar), set equivalent pod `resources.limits` and
`CELERY_R_CONCURRENCY`, and pass the intended values through the
`RESEARCH_R_WORKER_*` settings so provenance stays accurate.

Local development is Python-only by default. Start the R worker explicitly
after installing the pinned local R environment:

```bash
make -f Makefile.local local-dev-r
```

This verifies `Rscript`, `renv::status(project = "r_engine")`, and the R
project files before starting both the API and the dedicated worker with
`RESEARCH_R_ENABLED=true`. Use `make -f Makefile.local local-dev` for the
normal Python-only development stack.

Before rollout, run `Rscript -e 'renv::status()'`,
`Rscript -e 'testthat::test_dir("/opt/r_engine/tests/testthat")'`, and the
cross-runtime pytest suite with `RESEARCH_R_ENABLED=true`. Confirm the
`analysis-engines` endpoint lists R as available, submit an R analysis from
the frontend, and verify the completed `AnalysisRun` includes the canonical
runtime, identity, and artifact provenance.

Optional Playwright R gate (local/CI with a live `research_r` worker):

```bash
E2E_R_ENABLED=1 npx playwright test e2e/research-flow.spec.ts -g "R frequencies"
```
