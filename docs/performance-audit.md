# Performance And Architecture Audit

Date: 2026-04-02

## Current alignment

Backend:

- FastAPI, SQLAlchemy, Alembic, PostgreSQL, Redis, and Pydantic Settings are already in place.
- Celery now handles asynchronous jobs with Redis as the broker/result backend.
- The backend is structured as a modular monolith under `backend/app/modules`.
- Health endpoints, rate limiting, correlation IDs, and telemetry bootstrap hooks already exist.

Frontend:

- React, TypeScript, Vite, MUI, TanStack Query, React Hook Form, Zod, Vitest, and Playwright config are present.
- Route-level lazy loading is already enabled in the SPA router.

## Completed in this pass

- Fixed broken frontend relative imports under `frontend/src/pages`.
- Corrected the theme hook import so `AppLayout` uses the provider that actually exports `useColorMode`.
- Aligned frontend and backend contracts for:
  - admin users list pagination and search
  - admin user roles
  - `/users/me` returning `mfa_enabled`
  - reset-password payload naming
  - verification resend flow requiring an email address
- Added vendor chunk splitting in the Vite build to avoid shipping a single oversized main bundle.
- Removed the duplicate `@vitejs/plugin-react` entry from `frontend/package.json`.
- Added an ADR documenting the modular-monolith decision.
- Added Celery-backed async email delivery on top of the existing Redis layer.

## Observed gaps that still remain

- The frontend is still partly page-centric under `frontend/src/pages` instead of fully feature-colocated.
- There are no backend tests yet, and the frontend test setup has no actual test cases committed.
- Avatar upload remains a placeholder path instead of object storage integration.
- Optional observability dependencies are documented in `backend/pyproject.toml` but not enabled by default.
- Celery worker monitoring is not yet exposed through dedicated operational dashboards or metrics.
- The domain module is currently `projects`; if the business scope grows, it should either become the explicit core domain package or be renamed to match the actual product language.

## Recommended next steps

1. Move frontend pages into feature folders with local components and hooks.
2. Add backend integration tests for auth, refresh rotation, notifications, and admin pagination.
3. Add Playwright smoke coverage for sign-in, password reset, and protected-route navigation.
4. Enable Sentry and OpenTelemetry packages in backend dependencies for production environments.
5. Introduce structured logging and worker metrics around auth, email, admin, and notification workflows.
