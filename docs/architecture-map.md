# Architecture Map

Verified against the Phase 8 working tree. The existing
`.understand-anything` graph is from an older commit and is used only as
discovery input; the mappings below are source-verified.

## Composition

```text
React route/view
  -> frontend/src/api/* typed client + TanStack Query key
  -> /api/v1 FastAPI router
  -> module application/service
  -> repository/infrastructure + async SQLAlchemy/PostgreSQL
  -> optional Celery task + Redis events
  -> durable AnalysisRun / model / prediction / artifact metadata
```

`backend/api/router.py` composes the modular monolith. Shared configuration,
auth, storage, cache, database pooling, telemetry, and pagination are in
`backend/core`, `backend/db`, and `backend/lib`; module code must not import a
peer router and Text Research domain code must not import infrastructure.

## Module map

| Area | API owner | Frontend entry point | Durable / async path | Test evidence |
| --- | --- | --- | --- | --- |
| Identity and users | `identity_access/router.py`, `users/router.py` | auth hooks/pages, profile/settings | user, session, verification and MFA persistence; email task | backend identity tests; protected-route tests |
| Projects and calendar | `projects/router.py`, `calendar/router.py` | project pages, calendar page | project/task and calendar repositories | project/calendar unit tests |
| Profile and notifications | `profile/router.py`, `notifications/router.py` | profile feature, notification page | profile/notification repositories; object storage for avatars | profile serializer and notification tests |
| AI studio | `ai/router.py` and subroute modules | `features/ai/views/AiStudioView.tsx` | AI repositories; evaluation worker | AI module tests |
| Agent | `ai/agent_router.py` | `features/agent/AgentView.tsx` | persisted `AiRun` history/detail, agent service, RAG/memory ports | agent API/client tests |
| RAG | `rag/api/routes.py` | `features/rag/RagView.tsx` | RAG repositories, pgvector, object storage, ingestion/cleanup Celery tasks | RAG ingestion/retrieval/stream tests |
| Memory | `memory/api/routes.py` | `features/memory/MemoryView.tsx` | memory service/repository and extraction worker | memory tests |
| Platform and settings | platform route modules, `settings/router.py`, `admin/router.py` | platform, admin-platform, admin-users pages | shared config/cache and platform repositories | frontend mutation and backend module tests |
| Observability | `observability/router.py` | `features/observability` | Prometheus/telemetry links and health probes | URL-builder and page tests |
| Text Research | `text_research/api/routes.py`, `api/corpora.py` | nested `/research/:projectId/*` views | research repositories, analysis runs, artifacts, models, predictions, cache, events, workers | focused research unit/scale/workflow tests and research E2E |

## Text Research execution map

```text
ResearchPage + selected project/corpus/codebook context
  -> textResearch.ts + queryKeys.textResearch
  -> /api/v1/research routes
  -> application service (corpus, annotation, analysis, classification, topic,
     export, lifecycle, drift, contextual, or reliability)
  -> async SQLAlchemy repositories and persisted AnalysisRun state
  -> prepared-corpus builder / StageRunner / classifiers / topic engines
  -> artifact store and provenance
  -> Redis event publication and SSE reconciliation
  -> Celery when workload policy selects durable background execution
```

The core reproducibility boundary is `AnalysisSpecification` + normalized
specification hash + corpus snapshot/checksum + prepared-corpus artifact.
Feature selection, validation splitting, and classifier fitting stay in the
training partition. Human annotations, adjudications, and model predictions
remain in distinct stores.

## Worker routing

| Resource class / purpose | Queue | Entrypoints |
| --- | --- | --- |
| I/O research work | `RESEARCH_QUEUE_IO` | segmentation |
| CPU research work | `RESEARCH_QUEUE_CPU` | classifier training, prediction, robustness, quantitative analysis |
| GPU research work | `RESEARCH_QUEUE_GPU` | topic model, k sweep, seed stability |
| General tasks | `CELERY_TASK_DEFAULT_QUEUE` | RAG indexing/cleanup, AI evaluation |
| Email | `CELERY_EMAIL_QUEUE` | verification and reset mail |
| Memory | default Celery queue | turn-memory extraction |

`backend/workers/parallelism.py` limits BLAS/OpenMP/joblib threads per worker
process. `execution_defaults.py` supplies pure retry/timeout policy; adapters
retain the public compatibility exports used by workers.

## Cache, artifact, and event inventory

| Layer | Key / identity | Producer and consumer | Scope / expiry |
| --- | --- | --- | --- |
| Core typed cache | named keys in `core/cache.py` | profile, project, settings, platform, calendar, directory, observability | local and Redis according to key policy; TTLs in `core/config.py` |
| Embedding cache | provider + model + normalized text hash | RAG/AI embedding callers via `lib/embedding_cache.py` | batched Redis MGET/MSET, bounded single-flight; embedding TTL configuration |
| Retrieval/memory cache | retrieval and memory query keys | RAG/memory query paths | authorization-scoped keys; short configured TTLs |
| Research stage cache | engine version + stage + input checksum + spec/params hash | prepared corpus and StageRunner | L1 process, L2 Redis locks/meta, L3 artifact directory/object store; seven-day default TTL |
| Feature cache | corpus/text/preprocessing identity | quantitative operations | deterministic reusable token/feature output |
| Artifact registry/store | kind + checksum | prepared corpus, DFM, model, topic, export, manifest stages | content-addressed metadata; persistent artifact path is configured separately |
| Run events | `research:run:{run_id}:events` | persisted run lifecycle publishes; SSE and polling consume | Redis is ephemeral transport; PostgreSQL run state is authoritative |

## Known capability gaps carried to later phases

- Agent run creation, user-scoped persisted history, and detail are complete.
  Runs are synchronous, so an event stream is not currently a missing contract.
- Admin audit-log and metrics endpoints lack a matching frontend client/view.
- Project membership endpoints lack a matching frontend client/view.
- Existing R support exports a quanteda script and runs optional parity tests;
  it is not yet an executable R analysis engine. See
  `docs/r-quanteda-architecture.md`.
