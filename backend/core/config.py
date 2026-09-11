import json
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )
    APP_NAME: str = "CORPUS"
    APP_ENV: str = "dev"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    LOG_TO_CONSOLE: bool = True
    LOG_TO_FILE: bool = True
    LOG_FILE_PATH: str = "logs/logs.txt"
    LOG_RETENTION_DAYS: int = 1
    LOG_FORMAT: str = "text"
    SLOW_REQUEST_MS: int = 1000
    SLOW_JOB_MS: int = 5000
    SLOW_EXTERNAL_CALL_MS: int = 3000
    CORE_DOMAIN_SINGULAR: str = "Project"
    CORE_DOMAIN_PLURAL: str = "Projects"
    PLATFORM_DEFAULT_MODULE_PACK: str = "full_platform"

    DATABASE_URL: str
    # SQLAlchemy async connection pool (API process).
    # With PgBouncer (transaction pooling), keep these modest and prefer
    # DB_POOL_RECYCLE below server idle timeouts; see docs/runbooks/postgres-pgbouncer.md.
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    # Per Celery worker *process* pool. Keep small: total connections ≈
    # worker_processes × (DB_WORKER_POOL_SIZE + DB_WORKER_MAX_OVERFLOW).
    DB_WORKER_POOL_SIZE: int = 2
    DB_WORKER_MAX_OVERFLOW: int = 2
    DB_WORKER_POOL_TIMEOUT: int = 30
    DB_WORKER_POOL_RECYCLE: int = 1800
    REDIS_URL: str
    CACHE_ENABLED: bool = True
    CACHE_EMBEDDING_TTL_SECONDS: int = 600
    CACHE_EMBEDDING_MAX_TEXT_CHARS: int = 4000
    # Max in-process single-flight locks for identical embedding misses.
    CACHE_EMBEDDING_FLIGHT_LOCKS_MAX: int = 1024
    CACHE_RETRIEVAL_TTL_SECONDS: int = 180
    CACHE_PLATFORM_TTL_SECONDS: int = 300
    CACHE_SETTINGS_TTL_SECONDS: int = 60
    CACHE_OBSERVABILITY_STATUS_TTL_SECONDS: int = 30
    CACHE_MEMORY_SEARCH_TTL_SECONDS: int = 60
    CACHE_QUERY_EMBEDDING_TTL_SECONDS: int = 3600
    CACHE_USER_PROFILE_TTL_SECONDS: int = 90
    CACHE_PROJECT_LIST_TTL_SECONDS: int = 30
    CACHE_CALENDAR_TTL_SECONDS: int = 60
    CACHE_USER_DIRECTORY_TTL_SECONDS: int = 60
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""
    CELERY_TASK_ALWAYS_EAGER: bool = False
    CELERY_TASK_DEFAULT_QUEUE: str = "default"
    CELERY_EMAIL_QUEUE: str = "email"
    CELERY_RESULT_EXPIRES_SECONDS: int = 3600
    EAGER_BACKGROUND_MAX_WORKERS: int = 4
    EAGER_BACKGROUND_MAX_PENDING: int = 16

    JWT_SECRET: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_TOKEN_EXPIRE_DAYS: int

    FRONTEND_URL: str = "http://localhost:5173"
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"
    COOKIE_DOMAIN: str | None = None
    ADMIN_SIGNUP_INVITE_CODE: str = ""
    ACCESS_COOKIE_NAME: str = "access_token"
    REFRESH_COOKIE_NAME: str = "refresh_token"
    CSRF_COOKIE_NAME: str = "csrf_token"
    CSRF_HEADER_NAME: str = "X-CSRF-Token"
    PUBLIC_RATE_LIMIT_REQUESTS: int = 120
    PUBLIC_RATE_LIMIT_WINDOW_SECONDS: int = 60
    AUTH_SIGNIN_LIMIT: int = 10
    AUTH_SIGNIN_WINDOW_SECONDS: int = 60
    AUTH_SIGNUP_LIMIT: int = 5
    AUTH_SIGNUP_WINDOW_SECONDS: int = 3600
    AUTH_FAILURE_LIMIT: int = 8
    AUTH_FAILURE_WINDOW_SECONDS: int = 900
    # Playwright/CI only: raise rate-limit ceilings without disabling them.
    # Ignored when APP_ENV=production even if set.
    E2E_RELAX_RATE_LIMITS: bool = False
    HEALTH_READY_PUBLIC: bool = False
    HEALTH_VERSION_PUBLIC: bool = False
    REQUIRE_EMAIL_VERIFICATION: bool = False  # disabled for local dev; re-enable in production
    # Only these reverse proxies may supply X-Forwarded-* headers.  Leave empty
    # when the application receives client traffic directly.
    TRUSTED_PROXY_IPS: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Email verification / password reset token TTLs (seconds)
    VERIFICATION_TOKEN_TTL: int = 86400  # 24 h
    PASSWORD_RESET_TOKEN_TTL: int = 3600  # 1 h

    # SMTP — leave empty to skip sending (useful in dev)
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = "noreply@example.com"
    SMTP_TLS: bool = True

    # Observability
    SENTRY_DSN: str = ""
    SENTRY_TRACES_SAMPLE_RATE: float = 0.2
    OTLP_ENDPOINT: str = ""  # e.g. http://localhost:4317
    OTLP_INSECURE: bool = True
    OTEL_SERVICE_NAME: str = "fastapi-backend"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = ""
    OTEL_EXPORTER_OTLP_PROTOCOL: str = "http/protobuf"
    OTEL_TRACES_EXPORTER: str = "none"
    GRAFANA_PUBLIC_URL: str = "http://localhost:3001"
    PROMETHEUS_PUBLIC_URL: str = "http://localhost:9090"
    TEMPO_PUBLIC_URL: str = "http://localhost:3200"
    GRAFANA_APP_OVERVIEW_DASHBOARD_PATH: str = "/d/fastapi-overview/fastapi-overview"
    GRAFANA_API_DASHBOARD_PATH: str = "/d/fastapi-overview/fastapi-overview"
    GRAFANA_FRONTEND_DASHBOARD_PATH: str = ""
    GRAFANA_DATABASE_DASHBOARD_PATH: str = ""
    GRAFANA_CACHE_DASHBOARD_PATH: str = ""
    GRAFANA_WORKERS_DASHBOARD_PATH: str = ""
    GRAFANA_SCHEDULED_TASKS_DASHBOARD_PATH: str = ""
    GRAFANA_ERRORS_DASHBOARD_PATH: str = "/d/fastapi-overview/fastapi-overview"
    GRAFANA_TEMPO_EXPLORE_PATH: str = "/explore"

    # Object storage (S3-compatible, e.g. AWS S3 or MinIO)
    STORAGE_BUCKET: str = ""
    STORAGE_REGION: str = "us-east-1"
    STORAGE_ENDPOINT_URL: str = ""
    STORAGE_ACCESS_KEY: str = ""
    STORAGE_SECRET_KEY: str = ""
    STORAGE_USE_SSL: bool = False
    STORAGE_FORCE_PATH_STYLE: bool = True
    STORAGE_PUBLIC_BASE_URL: str = ""
    STORAGE_AUTO_CREATE_BUCKET: bool = True
    STORAGE_PUBLIC_READ: bool = False
    STORAGE_AVATAR_MAX_BYTES: int = 5 * 1024 * 1024

    AI_DEFAULT_PROVIDER: str = "local"
    AI_EMBEDDING_PROVIDER: str = "local"
    AI_LOCAL_MODEL_NAME: str = "local-heuristic"
    AI_MAX_OUTPUT_TOKENS: int = 1024
    AI_EVALUATION_CONCURRENCY: int = 3
    AI_RATE_LIMIT_REQUESTS: int = 30
    AI_RATE_LIMIT_WINDOW_SECONDS: int = 60
    AI_REQUEST_TIMEOUT_SECONDS: float = 60.0
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_DEFAULT_MODEL: str = "gpt-4.1-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com/v1"
    ANTHROPIC_DEFAULT_MODEL: str = "claude-3-5-sonnet-latest"

    # Mem0 / agent memory
    MEM0_MODE: str = "hosted"
    MEM0_API_KEY: str = ""
    MEM0_ORG_ID: str = ""
    MEM0_PROJECT_ID: str = ""
    MEM0_BASE_URL: str = ""
    MEMORY_ENABLED: bool = True
    MEMORY_WRITE_ENABLED: bool = True
    MEMORY_AUDIT_ENABLED: bool = True
    MEMORY_DEFAULT_LIMIT: int = 10
    MEMORY_MIN_CONFIDENCE: float = 0.65
    MEMORY_SESSION_TTL_DAYS: int = 30

    # RAG
    RAG_ENABLED: bool = True
    RAG_VECTOR_BACKEND: str = "pgvector"
    RAG_EMBEDDING_PROVIDER: str = ""
    RAG_EMBEDDING_MODEL: str = "text-embedding-3-small"
    RAG_EMBEDDING_DIMENSIONS: int = 1536
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 150
    RAG_TOP_K: int = 5
    RAG_SCORE_THRESHOLD: float = 0.3
    RAG_RERANK_ENABLED: bool = False
    RAG_RERANK_CANDIDATE_MULTIPLIER: int = 3
    RAG_MAX_CONTEXT_TOKENS: int = 6000
    RAG_ALLOWED_FILE_TYPES: str = "pdf,txt,md,docx,csv"
    RAG_MAX_FILE_BYTES: int = 10 * 1024 * 1024
    RAG_UPLOAD_CHUNK_SIZE: int = 256 * 1024
    RAG_EMBEDDING_BATCH_SIZE: int = 32
    RAG_EMBEDDING_BATCH_CONCURRENCY: int = 2
    RAG_ASK_PROMPT_TEMPLATE_KEY: str = "rag-answer"
    RAG_ASK_TIMEOUT_SECONDS: float = 45.0

    # Text Research (text research) model/vectorizer artifact storage
    RESEARCH_ARTIFACT_DIR: str = "var/research_artifacts"
    RESEARCH_LARGE_CORPUS_DOCUMENT_THRESHOLD: int = 50
    # Workload-aware async dispatch: above any threshold → AnalysisRun + Celery.
    RESEARCH_ASYNC_UNIT_THRESHOLD: int = 5_000
    RESEARCH_ASYNC_TOKEN_THRESHOLD: int = 500_000
    RESEARCH_ASYNC_PAIR_THRESHOLD: int = 2_000_000
    # Layered stage cache (L1 process / L2 Redis / L3 shared artifacts).
    RESEARCH_STAGE_CACHE_TTL_SECONDS: int = 604800  # 7 days
    # Bump this value to invalidate research-stage caches without deleting a
    # shared object-store bucket.  Configure an equivalent lifecycle rule in
    # S3/MinIO for the research-stage-cache/ prefix.
    RESEARCH_CACHE_GENERATION: str = "v1"
    RESEARCH_STAGE_OBJECT_TTL_DAYS: int = 7
    RESEARCH_STAGE_LOCK_TTL_SECONDS: int = 120
    RESEARCH_STAGE_LOCK_WAIT_SECONDS: int = 60
    RESEARCH_STAGE_L1_MAX_ENTRIES: int = 32
    RESEARCH_STAGE_L1_MAX_BYTES: int = 16 * 1024 * 1024

    # Nested parallelism controls for Celery CPU workers (§22).
    # Keep BLAS/OpenMP and sklearn/joblib at 1 inside each worker process so
    # scale-out comes from Celery --concurrency, not threads-within-processes.
    RESEARCH_APPLY_THREAD_LIMITS: bool = True
    RESEARCH_WORKER_BLAS_THREADS: int = 1
    RESEARCH_SKLEARN_N_JOBS: int = 1
    RESEARCH_JOBLIB_N_JOBS: int = 1
    # Optional container/image digest stamped into run provenance manifests.
    RESEARCH_IMAGE_DIGEST: str = ""

    # Text research (text research)
    RESEARCH_ARTIFACT_ROOT: str = ""
    RESEARCH_QUEUE_LIGHT: str = "research_light"
    RESEARCH_QUEUE_CPU: str = "research_cpu"
    RESEARCH_QUEUE_IO: str = "research_io"
    RESEARCH_QUEUE_NLP: str = "research_nlp"
    RESEARCH_QUEUE_MEMORY: str = "research_memory"
    RESEARCH_QUEUE_GPU: str = "research_gpu"
    # Optional isolated R/quanteda runtime.  Keeping this disabled means an
    # installation without R starts and serves Python analyses normally.
    RESEARCH_R_ENABLED: bool = False
    RESEARCH_RSCRIPT_PATH: str = "Rscript"
    RESEARCH_R_ENGINE_PATH: str = "../r_engine"
    RESEARCH_R_ENGINE_ENTRYPOINT: str = "../r_engine/run_analysis.R"
    RESEARCH_R_TIMEOUT_SECONDS: int = 600
    RESEARCH_R_MAX_OUTPUT_MB: int = 100
    RESEARCH_R_WORK_DIR: str = "/tmp/text-analysis-r"
    # Durable R artifact collection limits (worker job artifacts/ directory).
    RESEARCH_R_MAX_ARTIFACTS: int = 32
    RESEARCH_R_MAX_ARTIFACT_MB: int = 100
    RESEARCH_R_MAX_ARTIFACTS_TOTAL_MB: int = 250
    RESEARCH_QUEUE_R: str = "research_r"
    # Operator-declared R worker isolation (mirrored from Compose/K8s limits).
    # Enforceable limits live in the container cgroup; these fields stamp provenance
    # and document intent. ExecutionSpec.cpu / memory_mb remain advisory only.
    CELERY_R_CONCURRENCY: int = 1
    RESEARCH_R_WORKER_MEMORY_LIMIT: str = ""
    RESEARCH_R_WORKER_CPUS: str = ""
    RESEARCH_R_WORKER_PIDS_LIMIT: int | None = None

    CORS_ALLOWED_ORIGINS: Annotated[list[str], NoDecode] = Field(default_factory=list)

    @property
    def celery_broker_url(self) -> str:
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def celery_result_backend(self) -> str:
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

    @property
    def e2e_rate_limits_relaxed(self) -> bool:
        """True only for explicit non-production E2E/CI rate-limit relaxation."""
        return bool(self.E2E_RELAX_RATE_LIMITS) and not self.is_production

    def effective_rate_limit(self, configured: int, *, relaxed_floor: int) -> int:
        """Return configured limit, optionally raising a floor for E2E (never disables)."""
        if configured <= 0:
            return configured
        if self.e2e_rate_limits_relaxed:
            return max(configured, relaxed_floor)
        return configured

    @property
    def effective_public_rate_limit_requests(self) -> int:
        return self.effective_rate_limit(self.PUBLIC_RATE_LIMIT_REQUESTS, relaxed_floor=5_000)

    @property
    def effective_auth_signin_limit(self) -> int:
        return self.effective_rate_limit(self.AUTH_SIGNIN_LIMIT, relaxed_floor=100)

    @property
    def effective_auth_signup_limit(self) -> int:
        return self.effective_rate_limit(self.AUTH_SIGNUP_LIMIT, relaxed_floor=50)

    @property
    def allowed_origins(self) -> list[str]:
        return self.CORS_ALLOWED_ORIGINS or [self.FRONTEND_URL]

    @property
    def content_security_policy(self) -> str:
        connect_src = " ".join(dict.fromkeys(["'self'", *self.allowed_origins]))
        return (
            "default-src 'self'; "
            f"connect-src {connect_src}; "
            "img-src 'self' data: blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; "
            "base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
        )

    @field_validator("COOKIE_SAMESITE")
    @classmethod
    def validate_cookie_samesite(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"lax", "strict", "none"}:
            raise ValueError("COOKIE_SAMESITE must be one of: lax, strict, none")
        return normalized

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            return "INFO"
        return normalized

    @field_validator("LOG_RETENTION_DAYS")
    @classmethod
    def validate_log_retention_days(cls, value: int) -> int:
        if value < 1 or value > 365:
            return 1
        return value

    @field_validator("SLOW_REQUEST_MS")
    @classmethod
    def validate_slow_request_ms(cls, value: int) -> int:
        return value if value >= 1 else 1000

    @field_validator("SLOW_JOB_MS")
    @classmethod
    def validate_slow_job_ms(cls, value: int) -> int:
        return value if value >= 1 else 5000

    @field_validator("SLOW_EXTERNAL_CALL_MS")
    @classmethod
    def validate_slow_external_call_ms(cls, value: int) -> int:
        return value if value >= 1 else 3000

    @field_validator("JWT_SECRET")
    @classmethod
    def validate_jwt_secret(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 32 or stripped.lower() in {"replace-me", "changeme", "secret"}:
            raise ValueError("JWT_SECRET must be a high-entropy secret with at least 32 characters")
        return stripped

    @field_validator("ACCESS_TOKEN_EXPIRE_MINUTES")
    @classmethod
    def validate_access_ttl(cls, value: int) -> int:
        if value <= 0 or value > 30:
            raise ValueError("ACCESS_TOKEN_EXPIRE_MINUTES must be between 1 and 30")
        return value

    @field_validator("REFRESH_TOKEN_EXPIRE_DAYS")
    @classmethod
    def validate_refresh_ttl(cls, value: int) -> int:
        if value <= 0 or value > 30:
            raise ValueError("REFRESH_TOKEN_EXPIRE_DAYS must be between 1 and 30")
        return value

    @field_validator("CORS_ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_cors_allowed_origins(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, str):
            normalized = value.strip()
            if normalized.startswith(("'", '"')) and normalized.endswith(("'", '"')):
                normalized = normalized[1:-1].strip()
            if normalized.startswith("["):
                parsed = json.loads(normalized)
                if not isinstance(parsed, list):
                    raise ValueError(
                        "CORS_ALLOWED_ORIGINS must be a list or comma-separated string"
                    )
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in normalized.split(",") if item.strip()]
        return value

    @field_validator("TRUSTED_PROXY_IPS", mode="before")
    @classmethod
    def parse_trusted_proxy_ips(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def validate_security_posture(self):
        if self.COOKIE_SAMESITE == "none" and not self.COOKIE_SECURE:
            raise ValueError("COOKIE_SECURE must be true when COOKIE_SAMESITE is 'none'")
        if self.is_production and not self.COOKIE_SECURE:
            raise ValueError("COOKIE_SECURE must be enabled in production")
        if self.is_production and any(
            origin.startswith("http://") for origin in self.allowed_origins
        ):
            raise ValueError("CORS_ALLOWED_ORIGINS/FRONTEND_URL must use https in production")
        return self


settings = Settings()
