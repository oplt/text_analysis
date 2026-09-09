# Text Research Module

Generic computational text analysis workflow:
corpus creation through annotation, reliability, supervised classification,
topic modeling, and reproducible export.

Project-specific labels, dictionaries, and comparison groups are always
user-defined — the platform does not ship substantive research constructs.

Engineering report for the research-grade upgrade: [`text-research-engineering-report.md`](text-research-engineering-report.md).

## Architecture

```text
backend/modules/text_research/
  domain/           SQLAlchemy entities + enums
  application/      Workflow services (corpus → export)
  infrastructure/   sklearn engines, segmentation, repositories
  api/              FastAPI routes at /api/v1/research
  workers.py        Celery dispatch for long jobs
```

The module is a **bounded context**. It does not duplicate RAG ingestion.

### Relationship to RAG

| Concern | Owner |
|---------|-------|
| Document upload, PDF/DOCX parsing, object storage | `backend/modules/rag/` |
| Corpus membership + research metadata | `text_research` (`CorpusDocument`) |
| Retrieval-optimized chunks | RAG |
| Deterministic research units | `text_research` (`TextUnit`) |

**RAG chunks ≠ research text units.** Segmentation builds `document` / `paragraph` /
`sentence` units with stable positions and `text_hash` for reproducibility.

Research analysis starts from an immutable **canonical research source**
(`research_canonical_sources`), built by re-parsing the original file or from
an explicit full-text extract. Overlapping retrieval chunks are never joined
to reconstruct analysis text.

## Workflow

```text
Project → Research Corpus → RAG documents + metadata
       → Segmentation → TextUnits
       → Quantitative analysis / Topic models
       → Human annotation → Reliability → Adjudication
       → Frozen TrainingDatasetSnapshot
       → Classifier training (grouped split, TF-IDF fit on train only)
       → Evaluation → Corpus-wide prediction → Active learning
       → Comparative analysis / Dashboard / Export
```

## API

Base path: `/api/v1/research`

Key groups:

- **Corpora**: `/projects/{id}/corpora`, `/corpora/{id}/documents`, `/corpora/{id}/segment`
- **Annotation**: `/codebooks`, `/annotations`, `/corpora/{id}/reliability`, `/adjudication`
- **Analysis**: `/corpora/{id}/analysis/*` (stats, frequencies, ngrams, dfm, kwic, keyness, …)
- **Classification**: `/classifiers/dataset-preview`, `/classifiers/train`, `/classifiers/{id}/predict`
- **Topics**: `/corpora/{id}/topics/train`
- **Runs**: `/runs/{id}` with status polling; Celery for large jobs

All endpoints require authentication and project membership.

## Frontend

Routes under `/research/:projectId/`:

- `dashboard` — KPI summary from persisted data
- `corpus` — corpus CRUD, demo seed, segmentation
- `annotation` — multilabel workspace
- `reliability` — Cohen's κ, Krippendorff's α
- `analysis` — quantitative text analysis
- `classification` — dataset preview, training, metrics
- `topics` — LDA / NMF
- `explorer` — comparative discourse prevalence
- `exports` — CSV + reproducibility manifest

Open from a project detail page via **Open Text Research**.

## Statistical methods

Implemented in `infrastructure/` (scikit-learn, scipy, numpy):

- Term frequencies, n-grams, DFM (count/binary/TF-IDF), KWIC, dictionary hits, keyness, co-occurrence
- Cohen's kappa, Krippendorff's alpha (nominal), agreement matrices
- LDA, NMF with diagnostics (perplexity, topic diversity, overlap)
- Classifiers: Logistic Regression, Linear SVM, Multinomial NB, Complement NB, SGDClassifier
  (log_loss/hinge); OneVsRest wrapping for multilabel
- Task type (binary/multiclass/multilabel) is user/config-driven — inferred from
  label shape only when not explicitly requested, never silently forced to multilabel
- Configurable `FeatureConfig`: count or TF-IDF vectors, word n-grams and/or
  character n-grams (combined via `FeatureUnion`)
- Grouped train/validation/test split by source document (leakage prevention);
  validation is skipped with a note (not an error) when too few groups remain

## Preprocessing

Profiles stored in `research_preprocessing_profiles`. Default preserves negation
(`not`, `no`, `never`). Original unit text is never mutated.

## Training data & leakage prevention

1. Annotations → reliability → adjudication
2. `DatasetBuilderService` creates immutable `TrainingDatasetSnapshot`
3. Classifier vectorizer is **fit only on training partition**
4. Default split: **group by source document**

## Demo data

`POST /research/projects/{id}/demo-seed` creates a synthetic corpus (clearly labeled)
via existing RAG ingestion — four fictional policy documents with metadata.

Codebook labels are user-defined (no default theoretical constructs)
are marked `is_placeholder=True`.

## Background jobs

Celery tasks in `backend/workers/tasks.py`:

- `research_segmentation_task`
- `research_classifier_training_task`
- `research_topic_model_training_task`
- `research_robustness_sweep_task`

Triggered when corpora exceed `RESEARCH_LARGE_CORPUS_DOCUMENT_THRESHOLD` (default 50)
or when `run_async=true` on training endpoints.

### Nested parallelism (CPU workers)

Research CPU work must not nest Celery processes × sklearn/joblib workers ×
BLAS/OpenMP threads. Defaults keep **one thread per native pool inside each
worker process**; scale out with Celery `--concurrency` instead.

| Setting | Default | Effect |
|---------|---------|--------|
| `RESEARCH_APPLY_THREAD_LIMITS` | `true` | Master switch |
| `RESEARCH_WORKER_BLAS_THREADS` | `1` | Sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS` |
| `RESEARCH_SKLEARN_N_JOBS` | `1` | Default/clamp for sklearn `n_jobs` (negative/`-1` clamped) |
| `RESEARCH_JOBLIB_N_JOBS` | `1` | Default/clamp for joblib + `LOKY_MAX_CPU_COUNT` |
| `CELERY_CONCURRENCY` | `2` (Procfile.dev) | Worker process count |

Applied on Celery `worker_process_init` / `worker_ready` via
`backend/workers/parallelism.py`, and again at the start of research sync
job entrypoints (covers eager in-process runs).

Recommended for a dedicated `research_cpu` worker:

```bash
CELERY_CONCURRENCY=4          # ≈ physical cores (or slightly below)
RESEARCH_WORKER_BLAS_THREADS=1
RESEARCH_SKLEARN_N_JOBS=1
RESEARCH_JOBLIB_N_JOBS=1
```

Inspect the active policy:

```python
from backend.workers.parallelism import describe_parallelism_policy
print(describe_parallelism_policy())
```

## Migrations

`backend/alembic/versions/b4e8c2f1a903_add_text_research_tables.py`

```bash
cd backend && .venv/bin/alembic upgrade head
```

## Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 backend/.venv/bin/python -m pytest backend/modules/text_research/tests/ -q

cd frontend
npm test
npm run build
```

### End-to-end (Playwright)

Requires a running backend with migrated DB plus provisioned test user:

```bash
# terminal 1 — backend + worker + postgres/redis per README
cd backend && .venv/bin/alembic upgrade head

# terminal 2 — frontend dev (Playwright can auto-start this)
cd frontend
export E2E_TEST_EMAIL=you@example.com
export E2E_TEST_PASSWORD=your-password
export E2E_API_URL=http://localhost:8000
npm run test:e2e -- e2e/research-flow.spec.ts
```

CI runs the research Playwright workflow against the **current commit SHA** in
`.github/workflows/ci.yml` (API + Postgres + Redis + Vite). Quality-gate jobs
also cover backend lint/tests, frontend lint/tests/build, Alembic upgrade, and
1k/10k scale benchmarks (100k opt-in via workflow_dispatch).

Local mirrors:

```bash
make check          # ruff + eslint + tsc
make test-backend
make test-frontend
make bench-1k
make ci-local       # check + unit tests + 1k bench for $(git rev-parse HEAD)
```

`e2e/research-flow.spec.ts` exercises the full pipeline via API (demo seed → segment →
annotate → reliability → train → predict) and verifies the Text Research UI shows
documents, trained models, and exports.

## Limitations (MVP)

- No transformer models; sklearn TF-IDF + linear models only
- Western Bias Explorer shows patterns — no automated bias verdicts
- Contextual dataset entities exist in DB but no dedicated UI yet
- Large matrix views show preview/dimensions, not full sparse grids

## Recommended next steps

- Playwright E2E for full annotation → train → predict flow
- `@mui/x-charts` for dashboard visualizations
- Notification hooks when Celery jobs complete
- Stratified grouped CV UI warnings for unsafe splits
