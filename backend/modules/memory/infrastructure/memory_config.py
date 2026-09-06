from __future__ import annotations

from dataclasses import dataclass

from backend.core.config import settings


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    enabled: bool
    write_enabled: bool
    audit_enabled: bool
    default_limit: int
    min_confidence: float
    session_ttl_days: int
    mem0_mode: str
    mem0_api_key: str
    mem0_org_id: str
    mem0_project_id: str
    mem0_base_url: str
    app_id: str

    @classmethod
    def from_settings(cls) -> MemoryConfig:
        return cls(
            enabled=settings.MEMORY_ENABLED,
            write_enabled=settings.MEMORY_WRITE_ENABLED,
            audit_enabled=settings.MEMORY_AUDIT_ENABLED,
            default_limit=settings.MEMORY_DEFAULT_LIMIT,
            min_confidence=settings.MEMORY_MIN_CONFIDENCE,
            session_ttl_days=settings.MEMORY_SESSION_TTL_DAYS,
            mem0_mode=settings.MEM0_MODE,
            mem0_api_key=settings.MEM0_API_KEY,
            mem0_org_id=settings.MEM0_ORG_ID,
            mem0_project_id=settings.MEM0_PROJECT_ID,
            mem0_base_url=settings.MEM0_BASE_URL,
            app_id=settings.APP_NAME,
        )

    @property
    def is_hosted(self) -> bool:
        return self.mem0_mode in {"hosted", "self_hosted"}

    @property
    def mem0_configured(self) -> bool:
        if not self.enabled:
            return False
        if self.mem0_mode == "oss":
            return True
        return bool(self.mem0_api_key or self.mem0_base_url)


def validate_memory_config(config: MemoryConfig | None = None) -> None:
    resolved = config or MemoryConfig.from_settings()
    if resolved.mem0_mode not in {"hosted", "self_hosted", "oss"}:
        raise RuntimeError(f"Unsupported MEM0_MODE={resolved.mem0_mode!r}")
    if resolved.default_limit < 1:
        raise RuntimeError("MEMORY_DEFAULT_LIMIT must be at least 1")
    if not 0 <= resolved.min_confidence <= 1:
        raise RuntimeError("MEMORY_MIN_CONFIDENCE must be between 0 and 1")
    if resolved.session_ttl_days < 1:
        raise RuntimeError("MEMORY_SESSION_TTL_DAYS must be at least 1")
