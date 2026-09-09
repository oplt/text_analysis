# Contributing

Thank you for contributing to Text Analysis.

## Getting started

1. Fork the repository and create a feature branch from `main`.
2. Install backend and frontend dependencies following `README.md`.
3. Run relevant tests before opening a pull request:
   - `make check` for repository-wide checks
   - Backend: `pytest backend/modules/text_research/tests/`
   - Frontend: `cd frontend && npm test`

## Code guidelines

- Follow the modular monolith layout: `router → application/service → repository/infrastructure`.
- Keep domain code free of infrastructure imports.
- Add Alembic migrations for database schema changes.
- Prefer focused changes with tests for new behavior.

## Pull requests

- Describe the problem and the approach clearly.
- Link related issues when applicable.
- Keep commits focused; avoid unrelated refactors.

## Reporting issues

Use GitHub issues for bugs and feature requests. Include reproduction steps, expected behavior, and environment details when possible.
