# AI module

## Ownership

- `router.py`: public router aggregator only.
- `overview_routes.py` and `provider_service.py`: overview and provider descriptors.
- `prompt_routes.py` / `prompt_service.py`: prompt templates and versions.
- `document_routes.py` / `document_service.py`: legacy AI document/retrieval compatibility.
- `run_routes.py` / `run_service.py`: generation runs, provider execution, usage, and cost.
- `review_routes.py` / `review_service.py`: reviews and feedback.
- `evaluation_routes.py` / `evaluation_service.py`: datasets, cases, queued evaluation runs.
- `service.py`: compatibility facade and cross-capability overview coordinator.

## Document contract

AI does not own document persistence or ingestion. When `RAG_ENABLED=true`, legacy
`/api/v1/ai/documents*` operations delegate to `LegacyAiDocumentService`, which uses RAG
documents, ingestion, retrieval, and pgvector. Disabled document operations return 503;
disabled retrieval returns an empty result. Do not add parallel AI ingestion.

### Legacy route and table retirement

`/api/v1/ai/documents*` is a compatibility surface, not a second ingestion system.

1. Current release: keep routes backed by RAG; instrument consumers and migrate new clients to
   `/api/v1/rag/documents*` and `/api/v1/rag/retrieve`.
2. After two releases with no unapproved legacy consumers: mark legacy routes deprecated in the
   OpenAPI contract and publish the removal release.
3. Before removal: verify all legacy `ai_documents` data has a corresponding `rag_documents` row,
   take a database backup, and run retrieval parity checks.
4. Removal release: delete compatibility routes and ORM metadata, then add a dedicated migration
   dropping `ai_document_chunks` before `ai_documents`.

Never drop the legacy tables in the same migration that introduces RAG evaluation or run metadata.

## Generation behavior

`PromptContextService` is the single RAG + memory context composer used by direct runs, agent runs,
and RAG answers. Runs persist retrieval/memory degradation and filtered-injection counts. Generation
endpoints enforce a shared authenticated rate limit and request timeout.

The agent endpoint is single-turn orchestration. Durable conversation state comes from explicit
`run_id`, project scope, and the memory worker; it is not an autonomous multi-step tool loop.

## Compatibility guarantees

The aggregator registers 26 method/path pairs. Capability splits must preserve dependencies,
schemas, status codes, provider selection, token/cost accounting, transaction boundaries, and
failure behavior. `modules/ai/tests/test_route_registration.py` detects missing or duplicate
route registrations.
