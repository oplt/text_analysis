# Text Analysis

Text Analysis is a full-stack platform for reproducible computational text research. It combines corpus construction, canonical text preparation, human annotation, inter-coder reliability, quantitative text analysis, supervised machine learning, topic modeling, evidence-grounded corpus interrogation, experiment/run tracking, model and prediction management, and research-oriented export in one project-based workspace.

The platform is designed for computational social scientists, political scientists, sociologists, digital-humanities researchers, NLP researchers, research software engineers, and interdisciplinary teams that need more than an isolated NLP notebook. Its central design goal is to keep research evidence, analytical units, human judgments, model outputs, and provenance explicit enough to audit and reproduce.

> **Research boundary:** retrieval chunks are optimized for search and evidence retrieval. They are not treated as canonical statistical observations. Quantitative research begins from immutable canonical source text and deterministic research units.

---

## Why Text Analysis?

Serious computational text research normally spans several disconnected workflows: document ingestion, corpus construction, metadata management, cleaning and preprocessing, annotation, coder reliability, document-feature matrices, statistical text analysis, classification, topic modeling, experiment tracking, prediction review, provenance, and export.

Separating those stages across unrelated scripts and tools creates methodological risks. Retrieval chunks can accidentally become units of analysis; sentences from the same source document can leak into both training and test data; feature selectors can be fitted outside the training partition; model predictions can be confused with human codes; and preprocessing settings can disappear between exploratory work and publication.

Text Analysis treats those concerns as part of the software architecture. Research state is project-scoped and persisted, expensive analyses are represented as runs, annotation is campaign-scoped, training data can be frozen into snapshots, model predictions are stored separately from human and adjudicated judgments, and exports can carry provenance rather than only final numbers.

The result is not a substitute for research design. It is infrastructure for implementing and documenting that design consistently.

---

## Key capabilities

| Area | Implemented capabilities |
| --- | --- |
| **Identity & projects** | Sign-up/sign-in, sessions, optional MFA, project workspaces, project roles/membership API, project tasks |
| **Document ingestion** | PDF, TXT, Markdown, DOCX and CSV ingestion; structured PDF parsing options; optional OCR; object-storage integration |
| **RAG & evidence** | pgvector-backed dense retrieval, PostgreSQL lexical retrieval, reciprocal-rank fusion, scoped retrieval, citation validation, retrieval traces |
| **Corpus management** | Research corpora, document membership, researcher-defined metadata, canonical source text, ingestion QA |
| **Preparation** | Cleaning profiles, preprocessing profiles, language-aware processing, deterministic document/paragraph/sentence segmentation |
| **Annotation** | Versioned codebooks, labels, campaigns, assignment strategies, blind reliability mode, AI-assisted mode, adjudication |
| **Reliability** | Raw agreement, Cohen's κ, Fleiss' κ, Krippendorff's α, bootstrap confidence intervals, disagreement review |
| **Quantitative analysis** | Corpus statistics, frequencies, n-grams, DFM, weighting, KWIC, dictionaries, keyness, co-occurrence, similarity, duplicates, readability |
| **Exploratory methods** | Clustering, dimensionality reduction, comparative/contextual analysis, measurement comparison, statistical-model routes |
| **Classification** | Binary/multiclass/multilabel tasks; Logistic Regression, Linear SVM, Multinomial NB, Complement NB, SGD; embedding-based classifier paths |
| **Validation** | Group-aware splitting, grouped/nested CV, threshold tuning, bootstrap evaluation, robustness and temporal/transfer analyses |
| **Topic modeling** | LDA and NMF core engines; semantic topic stack; optional BERTopic dependencies |
| **Predictions** | First-class staged/published prediction sets, prediction browsing, active-learning queues, drift analysis |
| **Research assistant** | Corpus-scoped **Ask Corpus**, cited answers, evidence scope/coverage, streaming progress, corpus synthesis, saved research memos |
| **Reproducibility** | Dataset snapshots, specification/checksum identity, run provenance, artifact checksums, manifests, quanteda-script export |
| **Execution** | FastAPI async I/O, Celery queues, Redis, SSE run events, resource-aware job routing |
| **Scale** | Sparse matrices, hashing paths, batched repository access, stage caching, asynchronous thresholds, bounded research artifacts |
| **Quality** | pytest, Vitest, Playwright, Ruff, ESLint/TypeScript, Alembic migration checks, benchmark and RAG evaluation workflows |

Substantive concepts are deliberately not hard-coded. Codebook labels, dictionaries, comparison groups, metadata fields, and theoretical constructs are supplied by the researcher.

---

## End-to-end research workflow

```text
Project
  │
  ▼
Document ingestion
  │
  ├──────────────► RAG index / retrieval chunks ──► Ask Corpus / cited evidence
  │
  ▼
Research corpus
  │
  ▼
Canonical research sources
  │
  ▼
Cleaning + preprocessing specification
  │
  ▼
Deterministic segmentation
  │
  ▼
Research TextUnits
  │
  ├──────────────► Quantitative / exploratory analysis
  │
  ▼
Codebook + AnnotationCampaign
  │
  ▼
Multiple human annotators
  │
  ▼
Inter-coder reliability
  │
  ▼
Adjudication / GOLD labels
  │
  ▼
Frozen TrainingDatasetSnapshot
  │
  ▼
Feature engineering
  │
  ▼
Grouped train / validation / test design
  │
  ▼
Model training + evaluation
  │
  ▼
Model Registry
  │
  ▼
PredictionSet
  │
  ├──────────────► Active learning
  ├──────────────► Drift review
  ├──────────────► Robustness analysis
  └──────────────► Comparative analysis
  │
  ▼
Reproducible export
```

The workflow is deliberately non-linear where research requires it: topic models, corpus synthesis, similarity, dictionary analysis, clustering, and comparative methods can be run without first building a supervised classifier.

---

## Research methodology and scientific safeguards

### Canonical research sources versus RAG chunks

Text Analysis contains both a research-analysis subsystem and a retrieval-augmented generation (RAG) subsystem, but they serve different purposes.

RAG chunks may overlap because overlap can improve retrieval context. Joining those chunks back together would duplicate text spans and corrupt frequencies, document-feature matrices, hashes, annotations, and classifier inputs. For that reason, the Text Research module maintains a separate `CanonicalResearchSource` built from parser output or an explicit full-text source.

```text
Original document
      │
      ├──► retrieval chunks ──► embeddings / lexical search / RAG
      │
      └──► canonical research source
                    │
                    ▼
           deterministic segmentation
                    │
                    ▼
                 TextUnits
                    │
                    ▼
        quantitative / annotation / ML
```

Research units support document, paragraph, and sentence granularity and carry stable positional/provenance information where available. The separation between retrieval evidence and research observations is a core validity constraint, not a UI convention.

### Annotation campaigns

Annotation is organized around an `AnnotationCampaign`. A campaign binds coding to a specific study wave: codebook/version, unit type, annotator set, assignment strategy, annotation mode, and lifecycle state.

This scope prevents a reliability calculation from silently combining, for example, pilot annotations with a later codebook version or sentence-level coding with paragraph-level coding.

Two campaign modes are explicitly represented:

- **`blind_reliability`** — intended for independent coding. Peer codes, adjudications and model suggestions are hidden according to backend policy while coding is incomplete.
- **`ai_assisted`** — may surface model assistance and is kept distinct from the blind reliability design.

Blindness is enforced in the API as well as in the frontend, so hidden layers are not merely concealed visually.

### HUMAN, GOLD and MODEL remain distinct

The platform keeps three conceptually different sources of labels separate:

| Layer | Meaning | Storage role |
| --- | --- | --- |
| **HUMAN** | Original annotator judgments | Annotation records |
| **GOLD** | Adjudicated/reference judgments | Adjudication records |
| **MODEL** | Predictions from a trained model | PredictionSet / prediction records |

A prediction never overwrites a human annotation. Adjudication does not erase the underlying coder decisions. This makes coder disagreement, model evaluation and active learning inspectable instead of conflating different sources of evidence.

### Inter-coder reliability

The implemented reliability layer is currently **nominal**.

| Design | Primary statistic |
| --- | --- |
| Exactly two coders with overlapping units | **Cohen's κ** |
| Three or more coders in a fixed-rater-count nominal design | **Fleiss' κ** |
| Two or more coders with missing/ragged coder×unit cells | **Krippendorff's α** |

Raw agreement is available alongside chance-corrected coefficients. Reliability services are campaign-aware and can attach percentile bootstrap confidence intervals; the underlying implementation defaults to a 95% confidence level and deterministic seed unless callers override the configuration.

Weighted ordinal κ and a full ordinal/interval reliability pipeline are not implemented.

### Leakage prevention

Supervised learning is designed around an explicit invariant:

```text
vectorizer.fit        → training partition only
feature_selector.fit  → training partition only
hyperparameter tuning → inner/training folds only
threshold tuning      → designated validation/tuning data
test partition        → transform/evaluate only
```

The default research logic groups units by their source document so paragraphs or sentences from one document are not randomly distributed across training and test partitions. Nested grouped CV and related robustness workflows refit leakage-sensitive stages inside the appropriate training fold.

`TrainingDatasetSnapshot` provides a frozen training-data boundary so a model run is tied to a specific annotation/corpus state rather than whatever data happens to exist later.

Prediction execution additionally uses staged prediction sets: newly generated rows remain unpublished until the prediction set is finalized, and failed/cancelled runs can discard unpublished staged results.

---

## Quantitative text analysis

The quantitative path uses a Python implementation with terminology familiar to research packages such as quanteda:

```text
Canonical text
   ↓
TextUnits
   ↓
Cleaning / preprocessing
   ↓
Tokens / n-grams
   ↓
Document-feature representation
   ↓
Trim / weighting
   ↓
Analysis
```

| Method | Research use |
| --- | --- |
| **Corpus statistics** | Inspect corpus/document/unit composition |
| **Frequencies** | Count feature prevalence, optionally by metadata grouping |
| **N-grams** | Examine recurring token sequences and skip/n-gram patterns |
| **DFM** | Build sparse document-feature matrices with configurable trimming/weighting |
| **KWIC** | Inspect words, phrases or patterns in textual context |
| **Dictionaries** | Apply researcher-defined dictionaries and hierarchical categories |
| **Keyness** | Compare lexical distributions between groups/filters |
| **Co-occurrence** | Measure lexical associations and optionally build association-network output |
| **Similarity** | Compare units/documents using supported lexical or supplied-embedding representations |
| **Duplicate detection** | Detect exact, normalized, lexical and optional MinHash near-duplicates |
| **Clustering** | Explore unsupervised groups of text representations |
| **Dimensionality reduction** | Produce lower-dimensional representations for exploration |
| **Readability** | Compute supported readability/style descriptors |
| **Measurement comparison** | Compare researcher-supplied measures without automatically declaring equivalence |
| **Statistical model** | Fit supported configured statistical models over supplied/derived research rows |

DFM infrastructure supports count/binary and TF/TF-IDF-style weighting plus additional weighting paths used by the research engine. The API exposes BM25 parameters, sparse-only execution, preprocessing-profile selection, and feature trimming where relevant.

Keyness, dictionary and comparative routines operate on researcher-selected categories rather than embedding substantive political or social-science concepts in source code.

---

## Feature engineering

For classical supervised models, the feature pipeline can combine:

- count or TF-IDF vectorization;
- word n-grams;
- character n-grams;
- combined word/character spaces via scikit-learn composition;
- configurable document-frequency/feature trimming;
- optional supervised feature selection.

The important distinction is whether a transformation learns from labels or from the full sample distribution. Leakage-sensitive fitting is performed inside the training partition/fold. Held-out data is transformed only after the fitted pipeline exists.

Large-corpus preparation also includes hashing-based sparse paths that avoid materializing a complete vocabulary where that execution strategy is selected.

---

## Supervised classification

The normal research-training pipeline supports the following core model families:

- Logistic Regression;
- Linear SVM;
- Multinomial Naive Bayes;
- Complement Naive Bayes;
- SGDClassifier (`log_loss` / `hinge` paths).

It supports binary, multiclass and multilabel task types; multilabel classification uses a one-vs-rest strategy where required. The frontend also exposes embedding + logistic and embedding + SVM configurations.

A typical training lifecycle is:

```text
GOLD / selected annotation source
          ↓
TrainingDatasetSnapshot
          ↓
group-aware split
          ↓
vectorization
          ↓
optional feature selection
          ↓
training
          ↓
validation / nested CV / threshold tuning
          ↓
held-out evaluation
          ↓
registered model
          ↓
published PredictionSet
```

Evaluation infrastructure includes common classification metrics such as precision, recall, F1, confusion-matrix-derived diagnostics and task-appropriate aggregate metrics. Binary threshold objectives include F1, precision, recall, balanced accuracy, Youden's J, expected cost and custom utility; configuration also supports abstention-oriented workflows.

A transformer-classifier helper exists behind the optional `transformers` dependency extra, but **transformer training is not integrated into the normal classifier-training run**. The standard API deliberately rejects `algorithm="transformer"` rather than pretending that the optional helper is a supported end-to-end training path.

---

## Topic modeling

The core topic-model families are:

- **Latent Dirichlet Allocation (LDA)**
- **Non-negative Matrix Factorization (NMF)**

Topic workflows include configurable topic counts and stability/diagnostic operations such as topic-count sweeps and repeated-seed analysis.

The repository also contains a decomposed semantic topic stack:

```text
EmbeddingProvider
      ↓
DimensionalityReducer
      ↓
Clusterer
      ↓
TopicRepresentation
```

Classical engines remain the lightweight default. BERTopic, UMAP/HDBSCAN and related heavy dependencies are optional and should fail explicitly when their extras are not installed.

```bash
cd backend
uv sync --frozen --extra bertopic
```

Optional spaCy and transformer dependencies are similarly isolated:

```bash
uv sync --frozen --extra nlp
uv sync --frozen --extra transformers
```

---

## Ask Corpus, evidence and research memos

Text Analysis includes a corpus-scoped research assistant rather than treating RAG as an unbounded chatbot.

**Ask Corpus** resolves an authoritative evidence scope from the selected research corpus and delegates retrieval to the RAG module. The research facade uses project access control and an explicit document allow-list. Empty scope stays empty; it is not silently widened to all user documents.

The assistant stores and returns evidence-oriented metadata including retrieval trace identifiers, citations, claims, evidence revision hashes, coverage information, degradation state and the selected scope. Streaming endpoints emit turn progress, while the final persisted/validated answer remains the durable output.

A separate corpus-synthesis path supports broader evidence synthesis over the selected corpus. Assistant answers and synthesis results can be saved as **Research Memos**, which preserve their source/provenance relationship. Memos are editable as research notes without changing the underlying scientific analyses.

Assistant output does not mutate quantitative results, human annotations, GOLD adjudications or model predictions.

For RAG architecture, evaluation and parser details, see [`backend/modules/rag/README.md`](backend/modules/rag/README.md).

---

## System architecture

Text Analysis is a **modular monolith**: one deployable backend application composed from bounded feature modules rather than a collection of independently deployed microservices.

```text
┌────────────────────────────────────┐
│ React / TypeScript / MUI           │
│ TanStack Query                     │
└─────────────────┬──────────────────┘
                  │ REST / SSE
                  ▼
┌────────────────────────────────────┐
│ FastAPI / /api/v1                  │
└─────────────────┬──────────────────┘
                  │
        router → application/service
                  │
                  ▼
        repository / infrastructure
          │                  │
          ▼                  ▼
 PostgreSQL + pgvector      Redis
          │             cache / events
          │                  │
          └──────┬───────────┘
                 ▼
            Celery workers
                 │
                 ▼
        artifacts / object storage
```

Shared configuration, security, caching, database/session management and observability live in common backend packages; feature behavior lives under `backend/modules/*`. The Text Research domain layer is kept separate from infrastructure engines, and HTTP route modules remain orchestration/adaptation boundaries rather than the home of analytical logic.

The Text Research HTTP layer itself is split into corpus, annotation, quantitative, classification, topic, run, export, contextual, assistant and memo routers while preserving a common `/api/v1/research` surface.

---

## Repository structure

```text
text_analysis/
├── .github/
│   └── workflows/                 # current-HEAD CI, RAG nightly, quanteda parity
├── backend/
│   ├── api/                       # FastAPI composition and middleware
│   ├── core/                      # config, security, cache, storage
│   ├── db/                        # async SQLAlchemy sessions/pool policy
│   ├── lib/                       # shared adapters and utilities
│   ├── modules/
│   │   ├── identity_access/
│   │   ├── projects/
│   │   ├── rag/
│   │   ├── ai/
│   │   ├── memory/
│   │   └── text_research/
│   │       ├── api/
│   │       ├── application/
│   │       ├── domain/
│   │       ├── infrastructure/
│   │       └── tests/
│   ├── workers/                   # Celery app/tasks/parallelism
│   ├── alembic/                   # migrations
│   ├── pyproject.toml
│   └── uv.lock
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── app/
│   │   ├── components/
│   │   └── features/
│   │       └── text-research/
│   └── e2e/                       # Playwright workflows
├── infra/                         # Compose services and Nginx config
├── observability/                 # Prometheus/Grafana/Tempo local config/docs
├── docs/
│   └── frontend-ux-redesign-plan.md
├── DESIGN.md
├── Makefile
├── Makefile.local
└── Procfile.dev
```

---

## Technology stack

| Layer | Technologies |
| --- | --- |
| Backend runtime | Python 3.12+ |
| API | FastAPI, Uvicorn, Pydantic Settings |
| Persistence | SQLAlchemy async, asyncpg, Alembic |
| Database | PostgreSQL, pgvector |
| Background work | Celery, Redis |
| NLP / ML | scikit-learn, NumPy, SciPy, pandas, statsmodels, joblib |
| Text processing | regex, ftfy, Snowball stemmer, simplemma |
| Optional NLP | spaCy |
| Optional topic stack | BERTopic, UMAP, HDBSCAN |
| Optional transformer stack | PyTorch, Transformers |
| Document parsing | pypdf, python-docx; optional PyMuPDF/OCR |
| Object storage | boto3-compatible S3 / MinIO |
| Frontend | React 19, TypeScript, Vite |
| UI | MUI 7, MUI X Charts |
| Client data | TanStack Query |
| Forms/validation | React Hook Form, Zod |
| Backend tests | pytest |
| Frontend tests | Vitest, Testing Library |
| E2E | Playwright |
| Quality | Ruff, ESLint, TypeScript |
| Observability | Prometheus, OpenTelemetry, optional Sentry |

CI currently uses Node.js 22 and PostgreSQL 16 with the pgvector image.

---

## Quick start

### Prerequisites

For the standard local workflow:

- Python **3.12+**
- Node.js **22 recommended** (matches CI)
- PostgreSQL **16 recommended**, with the `vector` extension available for full RAG functionality
- Redis
- [`uv`](https://docs.astral.sh/uv/) for the locked backend environment
- npm

### 1. Clone

```bash
git clone https://github.com/oplt/text_analysis.git
cd text_analysis
```

### 2. Create local environment files

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

The backend example expects, by default:

```env
DATABASE_URL=postgresql+asyncpg://text_analysis:text_analysis@localhost:5432/text_analysis
REDIS_URL=redis://127.0.0.1:6380/0
FRONTEND_URL=http://localhost:5173
REQUIRE_EMAIL_VERIFICATION=false
```

The frontend example uses:

```env
VITE_API_BASE=/api/v1
```

Development secrets in `.env.example` are placeholders. Replace them for any non-local environment.

### 3. Install backend dependencies

```bash
cd backend
uv sync --frozen --extra dev
cd ..
source backend/.venv/bin/activate
```

Optional parser/NLP extras can be added separately, for example `--extra pdf`, `--extra pdf-ocr`, `--extra nlp` or `--extra bertopic`.

### 4. Install frontend dependencies

```bash
cd frontend
npm ci
cd ..
```

### 5. Migrate the database

Ensure the database in `DATABASE_URL` exists and is reachable, then run:

```bash
make db-migrate
```

The migration target validates connectivity and applies Alembic migrations to `head`.

### 6. Start development

The simplest app-focused startup is:

```bash
make local-dev-no-observability
```

This starts the application process group defined in `Procfile.dev`: local Redis on port 6380, FastAPI/Uvicorn, a Celery worker and Vite.

| Service | Default URL |
| --- | --- |
| Frontend | `http://localhost:5173` |
| API | `http://localhost:8000/api/v1` |
| OpenAPI / Swagger | `http://localhost:8000/docs` |
| Prometheus endpoint | `http://localhost:8000/metrics` when enabled |

To include the configured local observability stack, use:

```bash
make local-dev
```

### Optional Compose services

`infra/docker-compose.yml` defines PostgreSQL 16, Redis 7, Mailpit and MinIO. The Compose PostgreSQL service currently uses the standard `postgres:16` image, while CI uses `pgvector/pgvector:pg16` for vector-enabled validation. Therefore the Compose file should not be assumed to provide full pgvector parity without adapting the database image/environment.

If you use the Compose Redis service, note that it exposes port 6379 while the default `make local-dev` Redis configuration uses port 6380; keep `REDIS_URL` consistent with the runtime you choose.

---

## First research project

A practical UI workflow is:

1. Sign in and create a project.
2. Open the project's **Text Research** workspace.
3. Create a corpus and add/upload research documents.
4. Inspect source/canonical text and ingestion QA where needed.
5. Create cleaning/preprocessing profiles and preview transformations.
6. Segment the corpus at document, paragraph or sentence level.
7. Run exploratory quantitative analyses if appropriate.
8. Create/version a codebook.
9. Create an annotation campaign and assign coders.
10. Annotate research units under the selected campaign design.
11. Compute inter-coder reliability and inspect disagreements.
12. Adjudicate to create GOLD/reference labels where the study requires it.
13. Preview and freeze a training-dataset snapshot.
14. Configure a classifier and group-aware validation strategy.
15. Train, evaluate and register the resulting model.
16. Predict uncoded units into a separate `PredictionSet`.
17. Review active-learning candidates, drift or robustness results.
18. Run topic, comparative/contextual or measurement analyses as needed.
19. Use **Ask Corpus** for cited evidence questions or corpus synthesis without changing analytical results.
20. Save evidence-backed findings as research memos and export results/provenance.

The Text Research UI is mounted under:

```text
/research/:projectId/*
```

---

## Frontend research workspace

Current nested routes include:

| View | Purpose |
| --- | --- |
| `dashboard` | Research/project summary |
| `corpus` | Corpus and document management |
| `prepare` | Cleaning, preprocessing and preparation |
| `codebook` | Codebook/label management |
| `annotation` | Campaign setup and annotation workspace |
| `reliability` | Agreement, disagreement and adjudication workflows |
| `analysis` | Quantitative analysis, including statistical/measurement tabs |
| `dictionaries` | Research dictionary management |
| `classification` | Dataset snapshots, training and evaluation |
| `active-learning` | Uncertain-prediction review/assignment |
| `models` | Model registry and lifecycle |
| `predictions` | Prediction-set browsing |
| `drift` | Distribution-drift review |
| `topics` | Topic-model configuration/results |
| `robustness` | Robustness/stability workflows |
| `comparative` | Metadata-grouped comparison |
| `explorer` | Research exploration views |
| `contextual` | Contextual/mixed-source datasets |
| `runs` | Run history, results, provenance, comparison, cancellation/rerun |
| `exports` | Research-oriented exports/manifests |

Ask Corpus, evidence inspection and research memos are integrated into the research layout rather than implemented as a separate top-level research route.

Project membership endpoints exist in the backend, but dedicated membership-management UI is not currently present.

---

## API surface

The application base path is:

```text
/api/v1
```

Text Research is exposed under:

```text
/api/v1/research
```

Representative route families are:

| Area | Representative API surface |
| --- | --- |
| Corpora/documents | project corpora, corpus documents, metadata, source text |
| Preparation | cleaning profiles, preprocessing profiles, segmentation, ingestion QA |
| Annotation | codebooks, labels, campaigns, queues, annotations, blind policy |
| Reliability | corpus/campaign reliability, disagreements, adjudications |
| Quantitative | `/corpora/{id}/analysis/*` |
| Classification | dataset snapshots, classifiers, model lifecycle, prediction sets |
| Active learning | uncertain predictions and assignment |
| Topics | training, K-sweep, seed stability, topic labels |
| Runs | run history, events, results, provenance, rerun/cancel/compare |
| Exports | JSON, CSV, manifest and quanteda-script export |
| Contextual data | contextual datasets/observations/import/linking |
| Ask Corpus | assistant conversations, retrieve, message/stream, scope, synthesize |
| Research memos | project memos and save-from-assistant/synthesis |

All research routes are authenticated and operate within project/corpus authorization boundaries.

---

## Background processing and concurrency

Computational text analysis mixes I/O-bound and CPU-heavy workloads. The runtime therefore uses different execution mechanisms deliberately:

| Runtime | Intended work |
| --- | --- |
| `asyncio` | PostgreSQL, Redis, HTTP, storage and other I/O |
| Celery worker processes | CPU-heavy analysis, training, prediction and background research jobs |
| Native/library threads | Kept constrained to avoid nested oversubscription |
| GPU-oriented queue | Resource class for engines configured to require GPU execution |

The development worker subscribes to queues including:

```text
default, email,
research_light, research_cpu, research_io,
research_nlp, research_memory, research_gpu
```

The default development worker concurrency is 2 processes. Inside each process the research defaults constrain BLAS/OpenMP, scikit-learn and joblib parallelism:

```env
RESEARCH_APPLY_THREAD_LIMITS=true
RESEARCH_WORKER_BLAS_THREADS=1
RESEARCH_SKLEARN_N_JOBS=1
RESEARCH_JOBLIB_N_JOBS=1
```

This avoids the common `Celery processes × n_jobs=-1 × BLAS threads` oversubscription pattern. Scale process concurrency intentionally rather than enabling unrestricted nested parallelism.

### Run events

PostgreSQL is the durable source of truth for analysis-run state. Redis is used for ephemeral progress transport. Research run events can be consumed through SSE; persisted run state allows reconciliation/polling when the event stream is unavailable.

---

## Large-corpus and performance architecture

The repository contains explicit scale-oriented mechanisms rather than assuming all corpora fit comfortably in one dense in-memory structure:

- sparse DFM representations;
- hashing-based preparation/vectorization paths;
- bounded batch iteration;
- paged database reads;
- chunked ID predicates;
- projected queries for reliability/prediction workflows;
- staged/batched prediction persistence;
- content-addressed research artifacts;
- reusable stage caching where scientifically safe;
- asynchronous dispatch based on estimated workload.

Default workload thresholds in `backend/.env.example` include:

```env
RESEARCH_LARGE_CORPUS_DOCUMENT_THRESHOLD=50
RESEARCH_ASYNC_UNIT_THRESHOLD=5000
RESEARCH_ASYNC_TOKEN_THRESHOLD=500000
RESEARCH_ASYNC_PAIR_THRESHOLD=2000000
```

Local benchmark targets are available:

```bash
make bench-1k
make bench-10k
make bench-workflows
make bench-all
```

These are engineering regression tools; they should not be presented as universal performance guarantees for arbitrary hardware, databases or corpora.

The RAG subsystem additionally contains fixture-based retrieval/evidence evaluation and a separate live pgvector benchmark harness. See [`backend/modules/rag/README.md`](backend/modules/rag/README.md) for the distinction between deterministic pipeline tests, live retrieval evaluation and synthetic load measurements.

---

## Database, vectors, cache and storage

### PostgreSQL + pgvector

PostgreSQL stores durable product and research state: projects, corpora, canonical sources, units, annotations, runs, snapshots, model metadata, prediction sets, assistant evidence metadata and related provenance.

pgvector supports dense RAG retrieval. PostgreSQL full-text search supplies the lexical branch used by hybrid retrieval. The presence of pgvector does not mean retrieval embeddings are substituted for canonical research text.

Schema evolution is managed through Alembic migrations in `backend/alembic/versions/`.

### Redis

Redis is used for Celery, cache coordination and ephemeral event/retrieval infrastructure. Durable research run state remains in PostgreSQL.

### Object and artifact storage

The platform supports S3-compatible object storage through boto3/MinIO configuration. Research artifacts can also use the configured `RESEARCH_ARTIFACT_DIR`.

### Connection pools

The backend defines separate SQLAlchemy pool limits for API and worker processes. Worker pool capacity must be multiplied mentally by Celery process concurrency when sizing PostgreSQL connections.

---

## Headless CLI

The backend package registers a `text-research` command:

```bash
source backend/.venv/bin/activate
text-research --help
```

It provides headless access to parts of the research specification/execution workflow for automation and reproducible non-UI usage.

---

## Testing and quality assurance

### Repository checks

With the backend virtual environment activated:

```bash
make check
```

This runs backend Ruff checks/format validation plus frontend lint and TypeScript validation.

### Backend research tests

```bash
make test-backend
```

### Frontend unit tests

```bash
make test-frontend
```

Production frontend build:

```bash
cd frontend
npm run build
```

### End-to-end tests

```bash
cd frontend
npm run test:e2e
```

The Playwright suite includes the research workflow and dedicated Ask Corpus/UX coverage. Service-backed tests require the corresponding database/API configuration and test credentials.

### CI

`.github/workflows/ci.yml` defines current-commit gates for:

- dependency-lock consistency;
- backend Ruff checks;
- Text Research tests;
- RAG/failure-matrix tests, including optional PDF/OCR dependencies in CI;
- memory/agent tests;
- scale/workflow benchmarks;
- Alembic migrations against pgvector PostgreSQL;
- frontend lint, unit tests and build;
- Playwright research workflows.

Additional workflows include [`quanteda-parity.yml`](.github/workflows/quanteda-parity.yml) for optional R/quanteda parity and [`rag-nightly.yml`](.github/workflows/rag-nightly.yml) for heavier RAG evaluation.

Do not infer current-HEAD health from an older green workflow run; consult the Actions page for the exact commit being evaluated.

---

## Observability

The backend includes structured logging, Prometheus instrumentation, OpenTelemetry tracing and optional Sentry configuration. The authenticated application also includes an Observability Hub.

To run the configured local observability stack with development:

```bash
make local-dev
```

Operational details and local Prometheus/Grafana/Tempo configuration are documented in [`observability/README.md`](observability/README.md).

---

## Configuration

The exhaustive backend example is [`backend/.env.example`](backend/.env.example). Important configuration groups include:

- application/runtime and logging;
- PostgreSQL and API/worker pool limits;
- Redis and Celery;
- authentication, cookies, CSRF and rate limits;
- email;
- S3/MinIO storage;
- AI/memory providers;
- RAG parsing, chunking, retrieval, evidence and evaluation controls;
- research artifact paths and asynchronous thresholds;
- worker thread limits;
- observability.

The frontend environment example is [`frontend/.env.example`](frontend/.env.example).

---

## Security

Research endpoints require authentication and project-aware authorization. The wider application includes session controls, optional MFA, CSRF/cookie configuration, rate limits and administrator-only surfaces.

For deployments:

- never commit `.env` files, tokens or credentials;
- replace all development secrets;
- use secure cookie/TLS settings appropriate to the deployment;
- scope storage and model-provider credentials narrowly;
- keep project membership/authorization checks in service/API paths rather than relying on frontend visibility.

See [`SECURITY.md`](SECURITY.md) for reporting guidance.

---

## Reproducibility and research integrity

The intended provenance chain is:

```text
source document
  → canonical research source + checksum
  → cleaning / preprocessing specification
  → deterministic segmentation
  → TextUnit + provenance/hash
  → codebook version + annotation campaign
  → HUMAN annotations
  → adjudicated GOLD labels
  → TrainingDatasetSnapshot
  → analysis / feature / validation specification
  → AnalysisRun / TrainedModel
  → staged then published MODEL PredictionSet
  → export + provenance manifest
```

Research runs and artifacts preserve identifiers/checksums/specification metadata needed to distinguish one analytical configuration from another. The backend uses a committed `uv.lock`, and CI installs backend dependencies with frozen resolution.

For evidence-grounded assistant work, corpus scope, retrieval trace information, evidence revisions, citations and memo provenance provide a separate audit trail. This assistant provenance complements scientific-analysis provenance; it does not replace it.

---

## Current limitations

The project intentionally exposes several boundaries rather than hiding them:

1. **Reliability is nominal.** Weighted ordinal κ and a complete ordinal/interval agreement workflow are not implemented.
2. **R/quanteda is not an execution engine.** The repository contains Python parity fixtures, an optional R parity workflow and quanteda-script export, but research analyses are not executed through an embedded R/quanteda runtime.
3. **Transformer training is not part of the normal classifier run.** An optional helper exists, but the standard training API rejects `algorithm="transformer"`; classical and supported embedding paths remain the integrated workflow.
4. **Heavy NLP engines are optional.** spaCy, BERTopic, UMAP/HDBSCAN, OCR and transformer dependencies require explicit extras/native dependencies where applicable.
5. **Project membership management is API-only.** Membership endpoints exist, but no dedicated membership-management frontend was found in the current application routes/client.
6. **RAG quality is provider/data dependent.** Fixture and synthetic evaluation can verify retrieval/citation pipeline integrity, but they are not evidence that every embedding/generation provider will achieve the same semantic quality on a new corpus.
7. **The repository Compose PostgreSQL image is not full pgvector parity.** CI validates migrations against `pgvector/pgvector:pg16`; local infrastructure should use a vector-capable PostgreSQL when RAG is enabled.

---

## Roadmap candidates

The following items follow directly from the current implementation boundaries; they are not presented as already delivered:

- [ ] Integrate transformer classification into the normal training/run lifecycle if that becomes a supported core workflow.
- [ ] Add ordinal/weighted reliability once label-scale semantics are first-class end to end.
- [ ] Add an executable R/quanteda engine if cross-runtime execution becomes a project requirement.
- [ ] Add project-membership management to the frontend.
- [ ] Provide a vector-enabled local Compose database for closer CI/development parity.
- [ ] Add curated, version-controlled application screenshots once stable documentation screenshots are maintained independently of test-only visual baselines.

Open feature work should preserve the existing distinction between optional heavy dependencies and the lightweight classical research core.

---

## Documentation

Current repository documentation includes:

| Document | Purpose |
| --- | --- |
| [`backend/modules/rag/README.md`](backend/modules/rag/README.md) | RAG, Ask Corpus evidence boundary, parsing, retrieval, evaluation and scale testing |
| [`DESIGN.md`](DESIGN.md) | Implemented application design system and interaction principles |
| [`docs/frontend-ux-redesign-plan.md`](docs/frontend-ux-redesign-plan.md) | Frontend UX architecture/redesign record |
| [`observability/README.md`](observability/README.md) | Local metrics/tracing stack and Observability Hub |
| [`AGENTS.md`](AGENTS.md) | Repository conventions for AI-assisted development |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution workflow and architecture conventions |
| [`SECURITY.md`](SECURITY.md) | Security reporting guidance |
| [`CHANGELOG.md`](CHANGELOG.md) | Notable project changes |
| [`CITATION.cff`](CITATION.cff) | Software citation metadata |

The FastAPI OpenAPI UI at `/docs` is the most current endpoint-level reference for a running checkout.

---

## Contributing

Contributions should preserve the repository's architectural and methodological boundaries.

Typical validation before a pull request:

```bash
source backend/.venv/bin/activate
make check
make test-backend
make test-frontend
```

Database schema changes require Alembic migrations. New research behavior should include focused tests, and changes to supervised pipelines should demonstrate that held-out data cannot influence fitted preprocessing/selection stages.

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Citation

Software citation metadata is provided in [`CITATION.cff`](CITATION.cff). GitHub's **Cite this repository** interface can render that metadata where supported.

The repository does not declare a DOI, so a DOI should not be inferred or fabricated.

---

## License

Text Analysis is released under the [MIT License](LICENSE).

---

## Research-use note

Text Analysis provides infrastructure for making computational text workflows more explicit and auditable; it does not make substantive research decisions automatically. Researchers remain responsible for sampling, construct validity, codebook design, reliability interpretation, preprocessing choices, validation design, statistical assumptions, model selection and the claims drawn from results.