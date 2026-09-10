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

Engine **version** strings (`r-quanteda-1`, Python `ENGINE_VERSION`) are always
resolved server-side from the registered engine. Clients may send
`engine.runtime` and an optional implementation *family* (`quanteda`), never a
version used as provenance/cache identity.

## Reproducibility

The manifest and canonical result contain specification hash, corpus checksum,
pipeline checksum, engine name/version, runtime version and package versions.
Engine selection participates in normalized specification and computation
identity, so Python and R cannot reuse each other's cached results.

`r_engine/renv.lock` pins the intended R package set (`quanteda`,
`quanteda.textstats`, `arrow`, `jsonlite`, and `testthat`). The supplied
Dockerfile is for the R-capable Celery worker image; it is not an R HTTP API.

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

Before rollout, run `Rscript -e 'renv::status()'`,
`Rscript -e 'testthat::test_dir("/opt/r_engine/tests/testthat")'`, and the
cross-runtime pytest suite with `RESEARCH_R_ENABLED=true`. Confirm the
`analysis-engines` endpoint lists R as available, submit an R analysis from
the frontend, and verify the completed `AnalysisRun` includes the canonical
runtime, identity, and artifact provenance.
