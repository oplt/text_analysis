"""Architecture and setup for the RAG module."""

# RAG Module (LangChain)

Document upload, parsing, chunking, embedding, retrieval, and cited answers — integrated with the existing AI agent and memory layers.

## Architecture

```text
backend/modules/rag/
  domain/           # ParsedDocument, DocumentChunk, RagAnswer, enums
  application/      # ingestion, retrieval, answer, prompt context, legacy AI docs
  infrastructure/   # LangChain splitters/loaders, vector store, repos
  api/              # /api/v1/rag routes
```

Application services include:

- `legacy_ai_document_service.py` — backs `/api/v1/ai/documents*` when RAG is enabled
- `prompt_context_service.py` — shared bounded RAG + memory context for all generation paths
- `rag_tool.py` — low-level agent retrieval helper

LangChain is used **only** in `infrastructure/` (`langchain_text_splitters`, optional `langchain_community` loaders). PDF/DOCX/CSV parsing and text splitting run via `asyncio.to_thread` so the API event loop stays responsive during indexing.

## Context order (agent + memory + RAG)

1. System/developer prompt
2. Authenticated user identity (JWT — never from message text)
3. User/project memory (`MemoryService`)
4. RAG document chunks (`PromptContextService` → `RetrievalService`)
5. Session/working context
6. User question

## Configuration

```env
RAG_ENABLED=true
RAG_VECTOR_BACKEND=pgvector    # pgvector | qdrant
RAG_EMBEDDING_PROVIDER=local   # uses existing AiProviderRegistry
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=150
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.3
RAG_MAX_CONTEXT_TOKENS=6000
RAG_RERANK_ENABLED=false
RAG_RERANK_CANDIDATE_MULTIPLIER=3
RAG_ALLOWED_FILE_TYPES=pdf,txt,md,docx,csv
RAG_MAX_FILE_BYTES=10485760
```

Embeddings use the same provider registry as `/api/v1/ai` — no hardcoded API keys.

## Vector backend

| Backend | Status |
|---------|--------|
| `pgvector` | Default — `rag_chunks.embedding` column with HNSW cosine index; SQL `ORDER BY embedding <=> query` |
| JSON fallback | Bounded scan of `embedding_json` when pgvector is unavailable or vectors were not indexed |

Local heuristic embeddings use `RAG_EMBEDDING_DIMENSIONS` (default 1536) so dev indexes can use pgvector. Re-index existing documents after changing dimensions.

## AI document route consolidation

When `RAG_ENABLED=true` (default), legacy `/api/v1/ai/documents*` routes are served by `LegacyAiDocumentService` in this module (invoked from `AiService` / `ai/router.py`):

| AI route | Backing |
|----------|---------|
| `GET /ai/documents` | `rag_documents` |
| `POST /ai/documents` | upload + async index |
| `POST /ai/documents/upload` | upload + async index |
| `POST /ai/retrieve` | pgvector retrieval via `RetrievalService` |

The frontend can keep using `/ai/documents`; documents land in one corpus. Set `RAG_EMBEDDING_PROVIDER` empty to inherit `AI_EMBEDDING_PROVIDER`.

After deploy, run:

```sh
cd backend && alembic upgrade head
```

Migration `c4e8f2a91d03` enables the `vector` extension and creates `ix_rag_chunks_embedding_hnsw`.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/rag/documents/upload` | Upload + queue indexing (returns job) |
| GET | `/api/v1/rag/documents` | List documents |
| GET | `/api/v1/rag/documents/{id}` | Get document |
| DELETE | `/api/v1/rag/documents/{id}` | Soft delete |
| POST | `/api/v1/rag/documents/{id}/index` | Queue re-index (202 + job) |
| GET | `/api/v1/rag/documents/{id}/chunks` | List chunks |
| POST | `/api/v1/rag/retrieve` | Similarity search |
| POST | `/api/v1/rag/ask` | RAG answer with citations |
| GET | `/api/v1/rag/queries` | Query history |
| GET | `/api/v1/rag/jobs/{id}` | Ingestion job status |

## Upload and index

All parsing, chunking, and embedding runs in the background via `index_rag_document_task` — HTTP handlers only persist the file and enqueue work.

```bash
curl -X POST http://localhost:8000/api/v1/rag/documents/upload \
  -H "Cookie: access_token=..." \
  -F "file=@notes.md" \
  -F "project_id=<optional-uuid>"
```

Response includes `document` and `ingestion_job`. Poll `GET /api/v1/rag/jobs/{id}` for status.

Re-index an existing document:

```bash
curl -X POST http://localhost:8000/api/v1/rag/documents/{document_id}/index \
  -H "Cookie: access_token=..."
```

Returns `202 Accepted` with a pending job.

With `CELERY_TASK_ALWAYS_EAGER=true` (local dev default), indexing runs in a daemon background thread inside the API process so the HTTP response is not blocked. This still consumes API CPU/memory for embedding work — use real Celery workers (`CELERY_TASK_ALWAYS_EAGER=false`) in production.

## Ask

```json
POST /api/v1/rag/ask
{
  "query": "What database did we choose?",
  "project_id": "<optional>"
}
```

Returns `answer`, `citations[]`, `no_context_found` when nothing matches, and `ai_run_id` linking to the `ai_runs` record (cost/tokens/review parity with `/ai/runs`).

The request is bounded by `RAG_ASK_TIMEOUT_SECONDS`. Degradation fields identify retrieval or
memory failures, and `injection_chunks_filtered` reports excluded unsafe chunks. See
[`docs/runbooks/ai-rag-degraded-mode.md`](../../../docs/runbooks/ai-rag-degraded-mode.md).

Generation goes through `GenerationPort` (`backend/lib/generation_port.py`, adapter: `AiServiceGenerationPort`) using the prompt template keyed by `RAG_ASK_PROMPT_TEMPLATE_KEY` (default `rag-answer`), with built-in defaults when that template is absent.

## Agent integration

`AgentService` calls the same `PromptContextService` as `/rag/ask` and `/ai/runs` when
`RAG_ENABLED=true`.

## Access control

- Documents scoped by `user_id`
- `project_id` requires project membership (`ProjectAccessPort.ensure_project_access`)
- Retrieval never returns another user's chunks
- Admins can delete/index only when `is_admin` checks pass

## Prompt injection safety

Retrieved chunks are wrapped with untrusted-context rules. Document instructions cannot override system safety policies.

## Setup

```sh
cd backend
uv sync
alembic upgrade head
```

## Tests

```sh
PYTHONPATH=. uv run --project backend python -m unittest discover -s backend/modules/rag/tests -v
```

## Vector backend

Only `pgvector` is supported. The app fails fast at startup if `RAG_VECTOR_BACKEND` is set to an unsupported value. External vector DBs (e.g. Qdrant) require a new adapter implementation before enabling a new backend name in config.
