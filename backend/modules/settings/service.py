import asyncio
import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.cache import (
    SETTINGS_CONFIG_ENTRIES_CACHE_KEY,
    SETTINGS_DATABASE_CACHE_KEY,
    cache_get_or_load_json,
    get_local_cached_json,
    invalidate_settings_config_cache,
    invalidate_settings_related_caches,
    set_local_cached_json,
)
from backend.core.config import ENV_FILE, Settings, settings
from backend.modules.settings.env_file_store import EnvFileStore
from backend.modules.settings.models import AppSetting
from backend.modules.settings.repository import SettingsRepository
from backend.modules.settings.schemas import (
    ConfigEntryResponse,
    ConfigEntryUpdate,
    ConfigSettingsResponse,
)

CONFIG_NOTICE = (
    "Config values are saved to backend/.env. Values read directly from"
    " `settings` update immediately, "
    "but infrastructure-bound changes may still require a backend restart."
)

logger = logging.getLogger(__name__)

DEPRECATED_CONFIG_KEY_REPLACEMENTS = {
    "AI_DOCUMENT_MAX_BYTES": "RAG_MAX_FILE_BYTES",
    "AI_DOCUMENT_CHUNK_SIZE": "RAG_CHUNK_SIZE",
    "AI_DOCUMENT_CHUNK_OVERLAP": "RAG_CHUNK_OVERLAP",
}

CONFIG_FIELD_METADATA: dict[str, dict[str, Any]] = {
    "APP_NAME": {
        "description": "Application name used by FastAPI metadata.",
        "requires_restart": True,
    },
    "APP_ENV": {"description": "Application environment name.", "requires_restart": True},
    "APP_HOST": {"description": "Backend bind host.", "requires_restart": True},
    "APP_PORT": {"description": "Backend bind port.", "requires_restart": True},
    "LOG_LEVEL": {"description": "Application log level.", "requires_restart": True},
    "CORE_DOMAIN_SINGULAR": {
        "description": "Default singular label for the core domain.",
        "requires_restart": True,
    },
    "CORE_DOMAIN_PLURAL": {
        "description": "Default plural label for the core domain.",
        "requires_restart": True,
    },
    "PLATFORM_DEFAULT_MODULE_PACK": {
        "description": "Default module pack applied to cloned apps.",
        "requires_restart": True,
    },
    "DATABASE_URL": {
        "description": "Primary PostgreSQL connection string.",
        "requires_restart": True,
    },
    "REDIS_URL": {"description": "Primary Redis connection string.", "requires_restart": True},
    "CELERY_BROKER_URL": {
        "description": "Optional Celery broker URL override.",
        "requires_restart": True,
    },
    "CELERY_RESULT_BACKEND": {
        "description": "Optional Celery result backend override.",
        "requires_restart": True,
    },
    "CELERY_TASK_ALWAYS_EAGER": {
        "description": "Execute Celery tasks inline instead of queueing them.",
        "requires_restart": True,
    },
    "CELERY_TASK_DEFAULT_QUEUE": {
        "description": "Default queue name for asynchronous jobs.",
        "requires_restart": True,
    },
    "CELERY_EMAIL_QUEUE": {
        "description": "Queue name used for outbound email jobs.",
        "requires_restart": True,
    },
    "CELERY_RESULT_EXPIRES_SECONDS": {
        "description": "How long Celery task results are retained in seconds.",
        "requires_restart": True,
    },
    "JWT_SECRET": {"description": "JWT signing secret.", "requires_restart": False},
    "JWT_ALGORITHM": {"description": "JWT signing algorithm.", "requires_restart": False},
    "ACCESS_TOKEN_EXPIRE_MINUTES": {
        "description": "Access token TTL in minutes.",
        "requires_restart": False,
    },
    "REFRESH_TOKEN_EXPIRE_DAYS": {
        "description": "Refresh token TTL in days.",
        "requires_restart": False,
    },
    "FRONTEND_URL": {"description": "Frontend base URL.", "requires_restart": True},
    "COOKIE_SECURE": {
        "description": "Whether auth cookies require HTTPS.",
        "requires_restart": True,
    },
    "ADMIN_SIGNUP_INVITE_CODE": {
        "description": "Invite code required for admin registration during sign-up.",
        "requires_restart": False,
    },
    "VERIFICATION_TOKEN_TTL": {
        "description": "Email verification token TTL in seconds.",
        "requires_restart": False,
    },
    "PASSWORD_RESET_TOKEN_TTL": {
        "description": "Password reset token TTL in seconds.",
        "requires_restart": False,
    },
    "SMTP_HOST": {"description": "SMTP hostname.", "requires_restart": False},
    "SMTP_PORT": {"description": "SMTP port.", "requires_restart": False},
    "SMTP_USER": {"description": "SMTP username.", "requires_restart": False},
    "SMTP_PASSWORD": {"description": "SMTP password.", "requires_restart": False},
    "SMTP_FROM": {"description": "Email sender address.", "requires_restart": False},
    "SMTP_TLS": {"description": "Use TLS for outbound SMTP.", "requires_restart": False},
    "SENTRY_DSN": {"description": "Sentry DSN.", "requires_restart": True},
    "SENTRY_TRACES_SAMPLE_RATE": {
        "description": "Fraction of requests to trace in Sentry.",
        "requires_restart": True,
    },
    "OTLP_ENDPOINT": {
        "description": "OpenTelemetry collector endpoint.",
        "requires_restart": True,
    },
    "OTLP_INSECURE": {
        "description": "Allow insecure gRPC OTLP transport.",
        "requires_restart": True,
    },
    "STORAGE_BUCKET": {
        "description": "Object storage bucket for uploaded assets.",
        "requires_restart": True,
    },
    "STORAGE_REGION": {"description": "Object storage region.", "requires_restart": True},
    "STORAGE_ENDPOINT_URL": {
        "description": "Custom S3-compatible endpoint URL, e.g. MinIO.",
        "requires_restart": True,
    },
    "STORAGE_ACCESS_KEY": {
        "description": "Object storage access key.",
        "requires_restart": True,
    },
    "STORAGE_SECRET_KEY": {
        "description": "Object storage secret key.",
        "requires_restart": True,
    },
    "STORAGE_USE_SSL": {
        "description": "Use HTTPS for object storage traffic.",
        "requires_restart": True,
    },
    "STORAGE_FORCE_PATH_STYLE": {
        "description": "Force path-style S3 URLs; useful for MinIO.",
        "requires_restart": True,
    },
    "STORAGE_PUBLIC_BASE_URL": {
        "description": "Optional public base URL used to build asset URLs.",
        "requires_restart": True,
    },
    "STORAGE_AUTO_CREATE_BUCKET": {
        "description": "Create the storage bucket automatically on startup.",
        "requires_restart": True,
    },
    "STORAGE_PUBLIC_READ": {
        "description": "Apply a public-read policy to the storage bucket.",
        "requires_restart": True,
    },
    "STORAGE_AVATAR_MAX_BYTES": {
        "description": "Maximum avatar upload size in bytes.",
        "requires_restart": False,
    },
    "AI_DEFAULT_PROVIDER": {
        "description": "Default provider key used for AI generation.",
        "requires_restart": False,
    },
    "AI_EMBEDDING_PROVIDER": {
        "description": "Provider used for document embeddings.",
        "requires_restart": False,
    },
    "AI_LOCAL_MODEL_NAME": {
        "description": "Label used for the built-in local heuristic model.",
        "requires_restart": False,
    },
    "RAG_MAX_FILE_BYTES": {
        "description": "Maximum source document size for RAG ingestion.",
        "requires_restart": False,
    },
    "RAG_CHUNK_SIZE": {
        "description": "Chunk size in characters for RAG document ingestion.",
        "requires_restart": False,
    },
    "RAG_CHUNK_OVERLAP": {
        "description": "Chunk overlap in characters for RAG document ingestion.",
        "requires_restart": False,
    },
    "AI_MAX_OUTPUT_TOKENS": {
        "description": "Maximum output tokens requested from AI providers.",
        "requires_restart": False,
    },
    "AI_EVALUATION_CONCURRENCY": {
        "description": "Maximum number of evaluation cases to run in parallel.",
        "requires_restart": False,
    },
    "OPENAI_API_KEY": {
        "description": "OpenAI API key used by the AI provider adapter.",
        "requires_restart": False,
    },
    "OPENAI_BASE_URL": {
        "description": "Override for OpenAI-compatible API base URL.",
        "requires_restart": False,
    },
    "OPENAI_DEFAULT_MODEL": {
        "description": "Default OpenAI chat model for prompt versions.",
        "requires_restart": False,
    },
    "OPENAI_EMBEDDING_MODEL": {
        "description": "Default OpenAI embedding model.",
        "requires_restart": False,
    },
    "ANTHROPIC_API_KEY": {
        "description": "Anthropic API key used by the AI provider adapter.",
        "requires_restart": False,
    },
    "ANTHROPIC_BASE_URL": {
        "description": "Override for Anthropic API base URL.",
        "requires_restart": False,
    },
    "ANTHROPIC_DEFAULT_MODEL": {
        "description": "Default Anthropic model for prompt versions.",
        "requires_restart": False,
    },
}

TYPE_LABELS = {
    str: "string",
    int: "integer",
    bool: "boolean",
}

REDACTED_SECRET = "********"
SENSITIVE_KEY_MARKERS = (
    "SECRET",
    "PASSWORD",
    "TOKEN",
    "KEY",
    "DSN",
    "DATABASE_URL",
    "REDIS_URL",
)


class SettingsService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = SettingsRepository(db)

    async def list_database_settings(self) -> list[AppSetting]:
        cached = await cache_get_or_load_json(
            SETTINGS_DATABASE_CACHE_KEY,
            ttl_seconds=settings.CACHE_SETTINGS_TTL_SECONDS,
            loader=self._load_database_settings_for_cache,
        )
        return [self._setting_from_cache(item) for item in cached]

    async def _load_database_settings_for_cache(self) -> list[dict]:
        rows = await self.repo.list_all()
        return [self._setting_to_cache(row) for row in rows]

    async def create_database_setting(
        self, key: str, value: str, description: str | None
    ) -> AppSetting:
        existing = await self.repo.get_by_key(key)
        if existing:
            raise HTTPException(
                status_code=409, detail="A database setting with this key already exists"
            )

        setting = await self.repo.create(key=key, value=value, description=description)
        await self.db.commit()
        await self.db.refresh(setting)
        await invalidate_settings_related_caches(key)
        logger.info("Database setting created key=%s", key)
        return setting

    async def update_database_setting(self, setting_id: str, updates: dict[str, Any]) -> AppSetting:
        setting = await self.repo.get_by_id(setting_id)
        if not setting:
            raise HTTPException(status_code=404, detail="Database setting not found")

        for field, value in updates.items():
            setattr(setting, field, value)

        await self.db.commit()
        await self.db.refresh(setting)
        await invalidate_settings_related_caches(setting.key)
        logger.info("Database setting updated key=%s", setting.key)
        return setting

    async def delete_database_setting(self, setting_id: str) -> None:
        setting = await self.repo.get_by_id(setting_id)
        if not setting:
            raise HTTPException(status_code=404, detail="Database setting not found")

        setting_key = setting.key
        await self.repo.delete(setting)
        await self.db.commit()
        await invalidate_settings_related_caches(setting_key)
        logger.info("Database setting deleted key=%s", setting_key)

    @staticmethod
    def _setting_to_cache(setting: AppSetting) -> dict:
        return {
            "id": setting.id,
            "key": setting.key,
            "value": setting.value,
            "description": setting.description,
            "updated_at": setting.updated_at.isoformat(),
        }

    @staticmethod
    def _setting_from_cache(payload: dict) -> AppSetting:
        data = dict(payload)
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            data["updated_at"] = datetime.fromisoformat(updated_at)
        return AppSetting(**data)

    @classmethod
    async def list_config_entries(cls) -> ConfigSettingsResponse:
        cached = get_local_cached_json(SETTINGS_CONFIG_ENTRIES_CACHE_KEY)
        if cached is not None:
            return ConfigSettingsResponse.model_validate(cached)

        response = await asyncio.to_thread(cls._build_config_entries)
        set_local_cached_json(
            SETTINGS_CONFIG_ENTRIES_CACHE_KEY,
            response.model_dump(mode="json"),
            ttl_seconds=settings.CACHE_SETTINGS_TTL_SECONDS,
        )
        return response

    @classmethod
    def _build_config_entries(cls) -> ConfigSettingsResponse:
        env_entries = {
            key: value
            for key, value in cls._read_env_entries().items()
            if key not in DEPRECATED_CONFIG_KEY_REPLACEMENTS
        }
        items: list[ConfigEntryResponse] = []
        known_fields = Settings.model_fields
        ordered_keys = list(env_entries)

        for key in known_fields:
            if key not in env_entries:
                ordered_keys.append(key)

        for key in ordered_keys:
            if key in env_entries:
                value = env_entries[key]
            elif key in known_fields:
                value = cls._serialize_value(getattr(settings, key))
            else:
                value = ""

            items.append(
                ConfigEntryResponse(
                    key=key,
                    value=REDACTED_SECRET if cls._is_secret_key(key) and value else value,
                    value_type=cls._get_value_type(key),
                    description=CONFIG_FIELD_METADATA.get(key, {}).get("description"),
                    requires_restart=CONFIG_FIELD_METADATA.get(key, {}).get(
                        "requires_restart", True
                    ),
                    is_custom=key not in known_fields,
                    is_secret=cls._is_secret_key(key),
                )
            )

        return ConfigSettingsResponse(items=items, notice=CONFIG_NOTICE)

    @classmethod
    def _update_config_entries_sync(
        cls, updates: Iterable[ConfigEntryUpdate]
    ) -> ConfigSettingsResponse:
        update_items = list(updates)
        seen_keys: set[str] = set()
        raw_updates: dict[str, str] = {}

        for item in update_items:
            key = item.key
            replacement = DEPRECATED_CONFIG_KEY_REPLACEMENTS.get(key)
            if replacement:
                raise HTTPException(
                    status_code=422,
                    detail=f"Config key {key} is deprecated; use {replacement}",
                )
            if key in seen_keys:
                raise HTTPException(status_code=400, detail=f"Duplicate config key: {key}")
            seen_keys.add(key)
            raw_updates[key] = item.value

        merged_known_values = {key: getattr(settings, key) for key in Settings.model_fields}
        for key, value in raw_updates.items():
            if cls._is_secret_key(key) and value == REDACTED_SECRET:
                value = cls._read_env_entries().get(
                    key, cls._serialize_value(getattr(settings, key, ""))
                )
                raw_updates[key] = value
            if key in merged_known_values:
                merged_known_values[key] = value

        try:
            validated = Settings.model_validate(merged_known_values)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        normalized_updates = raw_updates.copy()
        for key in Settings.model_fields:
            if key in normalized_updates:
                normalized_updates[key] = cls._serialize_value(getattr(validated, key))

        EnvFileStore(ENV_FILE).update(normalized_updates, parse_line=cls._parse_env_line)

        for key in Settings.model_fields:
            setattr(settings, key, getattr(validated, key))

        invalidate_settings_config_cache()
        logger.info("Environment config updated keys=%s", sorted(normalized_updates))
        return cls._build_config_entries()

    @classmethod
    async def update_config_entries(
        cls, updates: Iterable[ConfigEntryUpdate]
    ) -> ConfigSettingsResponse:
        return await asyncio.to_thread(cls._update_config_entries_sync, list(updates))

    @staticmethod
    def _get_value_type(key: str) -> str:
        field = Settings.model_fields.get(key)
        if not field:
            return "string"
        return TYPE_LABELS.get(field.annotation, "string")

    @staticmethod
    def _serialize_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _is_secret_key(key: str) -> bool:
        normalized = key.upper()
        return any(marker in normalized for marker in SENSITIVE_KEY_MARKERS)

    @staticmethod
    def _parse_env_line(line: str) -> tuple[str, str] | None:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            return None

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            return None
        return key, value.strip()

    @classmethod
    def _read_env_entries(cls) -> dict[str, str]:
        if not ENV_FILE.exists():
            return {}

        entries: dict[str, str] = {}
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            parsed = cls._parse_env_line(line)
            if parsed:
                key, value = parsed
                entries[key] = value
        return entries
