# Combined R + Python Celery worker for the `research_r` queue.
# Build from repository root:
#   docker build -f r_engine/Dockerfile -t text-analysis-r-worker:local .
#
# The image contains pinned R/quanteda (renv) and the backend Celery app.
# It does NOT run an R HTTP API — CMD starts Celery on research_r.

FROM rocker/r-ver:4.4.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    LIBARROW_BINARY=true \
    RESEARCH_R_ENABLED=true \
    RESEARCH_RSCRIPT_PATH=Rscript \
    RESEARCH_R_ENGINE_ENTRYPOINT=/opt/r_engine/run_analysis.R \
    RESEARCH_QUEUE_R=research_r \
    PYTHONPATH=/app \
    R_LIBS_USER=/opt/r_engine/renv/library

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /usr/local/bin/uv
RUN uv python install 3.12

WORKDIR /opt/r_engine
COPY r_engine/renv.lock r_engine/.Rprofile ./
COPY r_engine/R ./R
COPY r_engine/run_analysis.R ./
RUN R -q -e 'install.packages("renv", repos="https://cloud.r-project.org"); renv::restore(prompt = FALSE)'

WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock ./backend/
COPY backend/api ./backend/api
COPY backend/core ./backend/core
COPY backend/db ./backend/db
COPY backend/lib ./backend/lib
COPY backend/modules ./backend/modules
COPY backend/observability ./backend/observability
COPY backend/workers ./backend/workers
COPY backend/__init__.py ./backend/__init__.py

WORKDIR /app/backend
RUN uv sync --frozen --no-dev

WORKDIR /app
CMD ["uv", "--directory", "/app/backend", "run", "celery", \
     "-A", "backend.workers.celery_app:celery_app", "worker", \
     "--loglevel=INFO", \
     "--hostname=worker-r@%h", \
     "--queues=research_r", \
     "--concurrency=1"]
