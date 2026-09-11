# Backend–Frontend Capability Matrix

Source-verified from FastAPI router registration, typed frontend clients,
`frontend/src/app/router.tsx`, and referenced views. Status means current
surface availability, not an endorsement of scientific interpretation.

| Capability | API prefix / owner | Frontend API and route | Status | Missing / notes |
| --- | --- | --- | --- | --- |
| Sign-in, verification, password reset, MFA | `/auth` | `api/auth.ts`, auth pages | COMPLETE | Protected routes enforce authenticated access. |
| User profile, sessions, avatar | `/users`, `/profile` | user/profile APIs, `/profile` | COMPLETE | Stored avatars resolve through the authenticated content endpoint. |
| Projects and task board | `/projects` | `api/projects.ts`, `/projects`, `/projects/:projectId` | COMPLETE | Project CRUD and tasks are reachable. |
| Project membership | `/projects/{id}/members` | none | API_ONLY | No membership client or management UI found. |
| Calendar | `/calendar/items` | `api/calendar.ts`, `/calendar` | COMPLETE | List/create exposed. |
| Notifications and preferences | `/notifications` | `api/notifications.ts`, `/notifications` | COMPLETE | Read state and preferences exposed. |
| Profile/admin user management | `/admin/users` | `api/admin.ts`, `/admin/users` | COMPLETE | Admin route is guarded. |
| Admin audit logs and metrics | `/admin/audit-logs`, `/admin/metrics` | none | API_ONLY | No frontend client/view found. |
| Platform subscription, API keys, webhooks, flags | `/platform` | `api/platform.ts`, `/platform` | COMPLETE | User-level operations exposed. |
| Platform admin config/plans/templates | `/platform/admin`, `/settings` | platform/settings APIs, `/admin/platform`, `/admin/settings` | COMPLETE | Admin routes are guarded. |
| AI studio | `/ai` | `api/ai.ts`, `/ai` | COMPLETE | Prompt, document, run, review, feedback, and evaluation client operations are present. |
| Agent execution | `/agent/runs` | `api/agent.ts`, `/agent` | COMPLETE | User-scoped persisted history and detail are available. Runs remain synchronous, so no event stream is required. |
| RAG document/retrieval workflow | `/rag` | `api/rag.ts`, `/rag` | COMPLETE | Upload, indexing status, chunks, retrieval, ask, history, and deletion exposed. |
| Memory management | `/memory` | `api/memory.ts`, `/memory` | COMPLETE | List/search/detail/create/delete/forget/audit exposed. |
| Observability links and health | `/observability` | feature-local API, `/observability` | COMPLETE | Health and configured external shortcuts exposed. |
| Research corpus and source documents | `/research/corpora`, `/research/documents` | `api/textResearch.ts`, `/research/:projectId/corpus` | COMPLETE | Corpus/document lifecycle, metadata, source text, segmentation and QA exposed. |
| Cleaning and preparation | `/research/cleaning*`, preprocessing profiles | text-research API, `/prepare` | COMPLETE | Profiles, preview, cleaning and segmentation status exposed. |
| Codebooks, campaigns, annotations, adjudication | `/research/codebooks`, campaigns, annotations | text-research API, `/codebook`, `/annotation`, `/reliability` | COMPLETE | Blind campaign policy is consumed by the annotation/reliability flows. |
| Reliability and disagreement review | `/research/*reliability*`, adjudication routes | text-research API, `/reliability` | COMPLETE | Run metrics, disagreements, and campaign adjudication exposed. |
| Quantitative analysis and dictionaries | analysis routes, dictionaries | text-research API, `/analysis`, `/dictionaries` | COMPLETE | Results, warnings, provenance, and dictionary administration are reachable. |
| Comparative/contextual analysis | comparative/contextual routes | text-research API, `/comparative`, `/explorer`, `/contextual` | COMPLETE | Descriptive comparative output is presented separately from causal claims. |
| Dataset snapshots and classification | classifier/snapshot routes | text-research API, `/classification`, `/active-learning` | COMPLETE | Snapshot, train, predict, active-learning assignment, and diagnostics exposed. |
| Models, lifecycle, predictions, drift | model/prediction/drift routes | text-research API, `/models`, `/predictions`, `/drift` | COMPLETE | Lifecycle timeline, prediction sets, and drift review exposed. |
| Topics, robustness, statistical/measurement tools | topic/robustness/analysis routes | text-research API, `/topics`, `/robustness`, analysis tabs | COMPLETE | Heavy route views are lazy loaded. |
| Runs, provenance, cancellation, exports | run/export routes | text-research API, `/runs`, `/exports` | COMPLETE | Run details, comparison, cancellation, rerun, manifests, CSV/JSON and quanteda script export exposed. |
| Executable R/quanteda analysis | analysis-engines + R runtime selector | `/analysis-engines`, `research_r` worker image | COMPLETE | Dedicated `worker-r` Celery image; readiness heartbeat; frequencies/DFM/KWIC/dictionary/keyness/cooccurrence; durable artifacts + provenance. See `docs/r-quanteda-architecture.md`. |
