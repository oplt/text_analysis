# ADR 0003: Module Dependency Graph

## Status
Accepted

## Context
ADR 0001 established a modular monolith with bounded contexts, but cross-module imports are only enforced by convention and review. `adaptation.txt` flagged coupling risks (`rag → ai.providers`, `memory → projects.repository`) and the need for an explicit allowed dependency graph.

## Decision
Document the allowed import directions below. New code must not introduce cycles or skip layers (e.g. `api/` importing another module's `repository` directly).

### Allowed dependencies

| Module | May import from |
|--------|-----------------|
| `identity_access`, `users`, `profile` | `core`, `db`, `lib` |
| `projects` | above + peer modules via service interfaces only when needed |
| `notifications`, `audit`, `settings`, `platform`, `admin` | `core`, `db`, `lib`, peer `service`/`schemas` as needed |
| `ai` | `core`, `db`, `lib`, `rag.application` (legacy document shim when `RAG_ENABLED`) |
| `rag` | `core`, `db`, `lib`, `ai.providers` (embeddings/generation registry), `lib.generation_port` adapter to AI runs |
| `memory` | `core`, `db`, `lib`, `lib.project_access` (project scoping) |
| `observability` | `core`, `lib` — must not import feature modules |
| `api`, `workers` | any module's public `api`/`application` layers |

### Forbidden patterns

- Feature module → another module's `router.py`
- `domain/` → `infrastructure/` imports in the wrong direction
- Circular imports between `ai`, `rag`, and `memory`
- Selecting unimplemented infrastructure backends without startup validation (see `validate_rag_config()`)

### Shared code

Cross-cutting helpers belong in `backend/lib/` (vectors, caches, pagination) rather than duplicated in modules.

## Consequences

Positive:

- Reviewers can reject PRs that violate the graph without debating ad hoc
- Future import-linter CI rule can mirror this table

Trade-offs:

- Some pragmatic imports remain only inside port adapters (`AiServiceGenerationPort`, `PlatformServiceReadPort`) until AI/platform expose narrower public APIs
- Layered layout (`domain/application/infrastructure`) applies to new high-churn modules only; flat modules are not renamed in bulk

## Follow-ups

- `backend/tests/test_module_boundaries.py` enforces router, domain/infrastructure, and observability
  boundaries in CI without runtime imports.
- AI ↔ RAG ↔ memory orchestration is consolidated behind RAG-owned `PromptContextService`; AI keeps
  only a thin adapter for its agent response contract.
