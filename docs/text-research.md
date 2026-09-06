# Policy Text Lab — Text Research Module

Computational analysis workflow for global education policy discourse research:
corpus creation through annotation, reliability, supervised classification,
topic modeling, and reproducible export.

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

Open from a project detail page via **Open Policy Text Lab**.

## Statistical methods

Implemented in `infrastructure/` (scikit-learn, scipy, numpy):

- Term frequencies, n-grams, DFM (count/binary/TF-IDF), KWIC, dictionary hits, keyness, co-occurrence
- Cohen's kappa, Krippendorff's alpha (nominal), agreement matrices
- LDA, NMF with diagnostics (perplexity, topic diversity, overlap)
- TF-IDF + Logistic Regression / Linear SVM (OneVsRest for multilabel)
- Grouped train/test split by source document (leakage prevention)

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

Placeholder codebook labels (Liberalism, Universalism, Individualism, Multiculturalism)
are marked `is_placeholder=True`.

## Background jobs

Celery tasks in `backend/workers/tasks.py`:

- `research_segmentation_task`
- `research_classifier_training_task`
- `research_topic_model_training_task`
- `research_robustness_sweep_task`

Triggered when corpora exceed `RESEARCH_LARGE_CORPUS_DOCUMENT_THRESHOLD` (default 50)
or when `run_async=true` on training endpoints.

## Migrations

`backend/alembic/versions/b4e8c2f1a903_add_text_research_tables.py`

```bash
cd backend && .venv/bin/alembic upgrade head
```

## Testing

```bash
cd backend
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest modules/text_research/tests/ -q

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

`e2e/research-flow.spec.ts` exercises the full pipeline via API (demo seed → segment →
annotate → reliability → train → predict) and verifies the Policy Text Lab UI shows
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
