# Text Analysis

Full-stack platform for computational text research: corpus management, human annotation, inter-coder reliability, quantitative analysis, supervised classification, topic modeling, and reproducible export.

Built as a modular FastAPI backend with a React frontend, PostgreSQL (pgvector), Redis, and Celery workers.

---

## Features

| Area | Capabilities |
|------|----------------|
| **Identity** | Sign-up / sign-in, sessions, optional MFA, admin controls |
| **Projects** | Multi-project workspace with membership |
| **RAG** | Document upload (PDF/DOCX), chunking, pgvector retrieval |
| **Text Research** | Research corpora, segmentation, annotation, reliability (κ / α), classifiers, topics, comparative explorer, exports |
| **Platform** | Billing hooks, API keys, webhooks, feature flags, email templates |
| **Observability** | Structured logging, Prometheus metrics, optional OpenTelemetry |

Detailed research workflow: [docs/text-research.md](docs/text-research.md)

---

## Architecture

```text
text_analysis/
├── backend/                 # FastAPI modular monolith
│   ├── api/                 # App entry, middleware, routers
│   ├── core/                # Config, security, cache, storage
│   ├── db/                  # SQLAlchemy base / session
│   ├── modules/             # Bounded contexts (identity, rag, text_research, …)
│   ├── workers/             # Celery tasks
│   └── alembic/             # Database migrations
├── frontend/                # React + Vite + MUI
│   └── src/features/        # Feature-oriented UI (incl. text-research)
├── infra/                   # Docker Compose (Postgres, Redis, Mailpit, MinIO, …)
├── observability/           # Local observability stack
└── docs/                    # Architecture notes and runbooks
```

**Layering (backend modules):** `router → application/service → repository/infrastructure`  
Domain code does not depend on infrastructure. Peer modules do not import each other’s routers.

**Vector store:** PostgreSQL + **pgvector** only.

---

## Tech stack

| Layer | Technologies |
|-------|----------------|
| Backend | Python 3.12+, FastAPI, SQLAlchemy (async), Alembic, Celery, Redis, Pydantic Settings |
| ML / text | scikit-learn, NumPy, SciPy, pandas, joblib |
| Frontend | React 19, TypeScript, Vite, MUI, TanStack Query, React Router, Zod |
| Data | PostgreSQL 16 + pgvector, Redis |
| Quality | Ruff, Biome, Vitest, Playwright, pre-commit |

---

## Prerequisites

- Python **3.12+**
- Node.js **20+** (recommended)
- PostgreSQL with **pgvector**
- Redis
- [uv](https://github.com/astral-sh/uv) or `pip` for backend deps (project uses a venv under `backend/.venv`)

Optional for full local parity: Docker (infra services), Mailpit (email), MinIO (object storage).

---

## Quick start

### 1. Clone and configure

```bash
git clone <repository-url>
cd text_analysis

cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
# Optional: cp infra/.env.example infra/.env
```

Edit `backend/.env` so `DATABASE_URL` and `REDIS_URL` match your local services. Example:

```env
DATABASE_URL=postgresql+asyncpg://text_analysis:text_analysis@localhost:5432/text_analysis
REDIS_URL=redis://127.0.0.1:6380/0
JWT_SECRET=<at-least-32-character-secret>
REQUIRE_EMAIL_VERIFICATION=false
```

Ensure the Postgres role and database exist and that the password in `DATABASE_URL` is correct.

### 2. Install dependencies

```bash
# Backend
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
# or: uv sync

# Frontend
cd ../frontend
npm install
```

### 3. Migrate the database

From the **repository root**:

```bash
make db-migrate
```

Or from `backend/`:

```bash
./scripts/migrate.sh upgrade head
# equivalent:
.venv/bin/python -m alembic upgrade head
```

Migrations live in `backend/alembic/versions/`. Do **not** run `alembic init` at the repo root.

### 4. Run the app

```bash
make local-dev
```

This starts Redis (if needed), the API, Celery worker, and Vite via `Procfile.dev`.

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| API docs | http://localhost:8000/docs |
| API base | http://localhost:8000/api/v1 |

Frontend dev requests to `/api` are proxied to the backend.

Without the observability stack:

```bash
make local-dev-no-observability
```

---

## Common commands

| Command | Description |
|---------|-------------|
| `make local-dev` | Migrate DB and start the local process stack |
| `make db-migrate` | Apply Alembic migrations to head |
| `make check` | Lint / format / TypeScript checks |
| `make fix` | Auto-fix lint issues |
| `make install-hooks` | Install pre-commit hooks |

Backend tests (example):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 backend/.venv/bin/python -m pytest backend/modules/text_research/tests/ -q
```

Frontend tests:

```bash
cd frontend
npm test
npm run build
```

E2E (Playwright; requires running backend and test credentials):

```bash
cd frontend
E2E_TEST_EMAIL=... E2E_TEST_PASSWORD=... npm run test:e2e
```

---

## Text Research (research UI)

1. Sign in and open or create a **project**.
2. On the project detail page, choose **Open Text Research**.
3. Work through the flow: corpus → segmentation → annotation → reliability → analysis / classification / topics → export.

API prefix: `/api/v1/research`  
UI routes: `/research/:projectId/*`

See [docs/text-research.md](docs/text-research.md) for methods, Celery tasks, and demo seed behavior.

---

## Environment notes

- **Email verification** can be disabled for local development via `REQUIRE_EMAIL_VERIFICATION=false`.
- **Object storage** (MinIO/S3) is optional for basic auth and research flows; missing MinIO may log warnings at startup.
- Never commit `.env`, secrets, tokens, or `dump.rdb`.
- Prefer `backend/.venv/bin/python -m alembic` over a globally installed Alembic binary.

---

## Documentation

| Document | Topic |
|----------|--------|
| [docs/text-research.md](docs/text-research.md) | Text Research module |
| [DESIGN.md](DESIGN.md) | System design |
| [docs/adr/](docs/adr/) | Architecture decision records |
| [docs/logging.md](docs/logging.md) | Logging |
| [observability/README.md](observability/README.md) | Metrics / tracing stack |
| [AGENTS.md](AGENTS.md) | Conventions for AI-assisted development |

---

## License

See [`LICENSE`](LICENSE) (MIT). Community docs: [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), [`CITATION.cff`](CITATION.cff), [`CHANGELOG.md`](CHANGELOG.md).
