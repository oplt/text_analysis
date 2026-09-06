"""Architecture and setup for the agent memory module."""

# Agent Memory (Mem0)

Production-ready layered memory for authenticated AI agents. Mem0 is the vector memory engine; this app wraps it behind `MemoryService` so no other module imports the Mem0 SDK directly.

## Memory layers

| Layer | Scope | Mem0 key | Purpose |
|-------|-------|----------|---------|
| Working | Single request | — (in-memory only) | Current message, tool results, retrieved memories for one LLM call |
| Session | Conversation run | `run_id` | Task state, plans, short-lived context |
| User | Authenticated user | `user_id` | Preferences, goals, instructions |
| Agent | Persona / tooling | `agent_id` | Behavior rules, tool-use patterns |
| Project | Workspace / codebase | `project_id` metadata | Architecture decisions, domain facts |
| Episodic | Timeline | `user_id` + event metadata | Important chronological events |
| Audit | Local Postgres | — | Provenance for add/search/delete (no secrets) |

**Identity rule:** `user_id` always comes from JWT cookie auth (`get_current_user`). The agent never infers identity from chat text.

## Module layout

```text
backend/modules/memory/
  domain/           # models, enums, storage policies
  application/      # MemoryService, router, extractor, consolidator, context builder
  infrastructure/   # Mem0 adapter, audit/registry repos, metrics
  api/              # /api/v1/memory routes
backend/modules/ai/
  application/      # AgentService — recall before LLM, extract after
  agent_router.py   # /api/v1/agent/runs
  router.py         # /api/v1/ai/* prompts, runs, legacy document routes
```

## Configuration

```env
MEM0_MODE=hosted          # hosted | self_hosted | oss
MEM0_API_KEY=
MEM0_ORG_ID=
MEM0_PROJECT_ID=
MEM0_BASE_URL=            # for self-hosted Mem0 server
MEMORY_ENABLED=true
MEMORY_WRITE_ENABLED=true
MEMORY_AUDIT_ENABLED=true
MEMORY_DEFAULT_LIMIT=10
MEMORY_MIN_CONFIDENCE=0.65
MEMORY_SESSION_TTL_DAYS=30
```

Set `MEMORY_ENABLED=false` or leave `MEM0_API_KEY` empty to run chat without memory (degraded mode).

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/memory` | List user memories |
| GET | `/api/v1/memory/{id}` | Get one memory |
| POST | `/api/v1/memory` | Create memory (manual) |
| DELETE | `/api/v1/memory/{id}` | Delete memory |
| POST | `/api/v1/memory/search` | Semantic search |
| POST | `/api/v1/memory/forget/{id}` | Forget with reason |
| GET | `/api/v1/memory/audit` | Audit log for current user |
| POST | `/api/v1/agent/runs` | AI run with memory recall + extraction |

## Agent flow

1. Authenticated user hits `POST /api/v1/agent/runs`.
2. `AgentService` resolves `user.id`, `agent_id`, `run_id`, optional `project_id`.
3. `AgentPromptContextBuilder` (in `backend/modules/ai/application/`) performs **one** memory recall via `MemoryService.recall_for_prompt`, loads RAG document context in parallel, and assembles the system prompt block.
4. `AiService.run_prompt` executes the LLM call (no duplicate document retrieval when RAG is enabled).
5. `MemoryExtractor` + `MemoryPolicyService` decide what to persist; `MemoryRouter` picks the layer.
6. Audit rows written to `memory_audit_logs`; references in `memory_registry`.

## Prompt injection safety

Retrieved memories are appended as background context with an explicit instruction not to override system safety rules.

## Database

```sh
cd backend && alembic upgrade head
```

Tables: `memory_audit_logs`, `memory_registry` (references only — content stays in Mem0).

## Example agent run

```json
POST /api/v1/agent/runs
{
  "prompt_template_key": "assistant",
  "variables": { "topic": "FastAPI" },
  "user_message": "I prefer beginner-friendly FastAPI examples.",
  "agent_id": "default",
  "project_id": "<optional-project-uuid>"
}
```

Response includes `memory_run_id` for session-scoped recall on follow-up turns (pass as `run_id`).

## Tests

```sh
cd backend && python -m unittest discover -s tests -p 'test_memory*.py'
```
