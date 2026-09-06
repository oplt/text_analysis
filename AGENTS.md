# AI Assistant Guidelines

## Architecture

* Backend: FastAPI modular monolith in `backend/modules/`
* Core/shared concerns: `backend/core/`
* Frontend: React + Vite in `frontend/src/`; prefer `features/*`
* Workers: Celery in `backend/workers/`
* Observability: `backend/observability/` and `observability/`
* Check `DESIGN.md` and `docs/adr/` for architectural changes

## Conventions

* Python: type hints, async SQLAlchemy
* Follow `router -> service/application -> repository/infrastructure`
* Do not import peer modules' routers
* Domain code must not depend on infrastructure
* Reuse existing helpers before adding new ones
* Add Alembic migrations for DB schema changes

## AI / RAG

* RAG owns document ingestion
* `/api/v1/ai/documents*` is a legacy compatibility layer backed by RAG
* Do not add parallel ingestion/vector-storage paths
* Vector backend: pgvector only
* Unsupported backends must fail validation

## Validation

* Run relevant tests
* Run `make check`
* Backend: relevant `pytest` tests
* Frontend: `cd frontend && npm test`
* Run E2E tests when changing covered user flows
* Never claim tests passed unless actually run

## Do not

* Commit secrets, `.env`, tokens, credentials, or `dump.rdb`
* Fabricate test results or API behavior
* Bypass module boundaries
* Duplicate existing infrastructure/helpers
* Make unrelated broad refactors
