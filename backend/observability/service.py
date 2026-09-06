import asyncio
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit

import httpx
from sqlalchemy import text

from backend.core.cache import (
    OBSERVABILITY_STATUS_CACHE_KEY,
    cache_get_or_load_model,
)
from backend.core.cache import redis_client
from backend.core.config import Settings
from backend.db.session import engine
from backend.observability.schemas import (
    ObservabilityDashboards,
    ObservabilityLinks,
    ObservabilityStatus,
    ObservabilityStatusItem,
    ObservabilityToolLink,
)

TECHNICAL_ACCESS_REQUIRED = True

_observability_http_client: httpx.AsyncClient | None = None


def _get_observability_http_client() -> httpx.AsyncClient:
    global _observability_http_client
    if _observability_http_client is None or _observability_http_client.is_closed:
        _observability_http_client = httpx.AsyncClient(timeout=1.5, follow_redirects=True)
    return _observability_http_client


async def close_observability_http_client() -> None:
    global _observability_http_client
    if _observability_http_client is not None and not _observability_http_client.is_closed:
        await _observability_http_client.aclose()
    _observability_http_client = None


def build_public_url(base_url: str, path: str = "") -> str | None:
    base = base_url.strip()
    if not base:
        return None
    parts = urlsplit(base)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None

    normalized_base_path = parts.path.rstrip("/")
    normalized_path = path.strip()
    if normalized_path:
        normalized_path = "/" + normalized_path.strip("/")
    joined_path = f"{normalized_base_path}{normalized_path}" or "/"

    return urlunsplit((parts.scheme, parts.netloc, joined_path, "", ""))


def _tool_link(url: str | None, allowed: bool) -> ObservabilityToolLink:
    return ObservabilityToolLink(
        url=url if allowed else None,
        configured=url is not None,
        allowed=allowed,
    )


class ObservabilityService:
    def __init__(self, settings: Settings):
        self._settings = settings

    def get_links(self, *, is_admin: bool) -> ObservabilityLinks:
        has_technical_access = is_admin
        grafana_base = build_public_url(self._settings.GRAFANA_PUBLIC_URL)
        prometheus_graph = build_public_url(self._settings.PROMETHEUS_PUBLIC_URL, "/graph")
        tempo_explore = build_public_url(
            self._settings.GRAFANA_PUBLIC_URL,
            self._settings.GRAFANA_TEMPO_EXPLORE_PATH,
        )

        def dashboard(path: str) -> ObservabilityToolLink:
            url = (
                build_public_url(self._settings.GRAFANA_PUBLIC_URL, path)
                if path.strip()
                else None
            )
            return _tool_link(url, has_technical_access)

        return ObservabilityLinks(
            grafana_base_url=_tool_link(grafana_base, has_technical_access),
            prometheus_url=_tool_link(prometheus_graph, has_technical_access),
            tempo_explore_url=_tool_link(tempo_explore, has_technical_access),
            dashboards=ObservabilityDashboards(
                application_overview=dashboard(self._settings.GRAFANA_APP_OVERVIEW_DASHBOARD_PATH),
                api=dashboard(self._settings.GRAFANA_API_DASHBOARD_PATH),
                frontend=dashboard(self._settings.GRAFANA_FRONTEND_DASHBOARD_PATH),
                database=dashboard(self._settings.GRAFANA_DATABASE_DASHBOARD_PATH),
                cache=dashboard(self._settings.GRAFANA_CACHE_DASHBOARD_PATH),
                workers=dashboard(self._settings.GRAFANA_WORKERS_DASHBOARD_PATH),
                scheduled_tasks=dashboard(self._settings.GRAFANA_SCHEDULED_TASKS_DASHBOARD_PATH),
                errors=dashboard(self._settings.GRAFANA_ERRORS_DASHBOARD_PATH),
            ),
        )

    async def get_status(self) -> ObservabilityStatus:
        return await cache_get_or_load_model(
            OBSERVABILITY_STATUS_CACHE_KEY,
            ObservabilityStatus,
            ttl_seconds=self._settings.CACHE_OBSERVABILITY_STATUS_TTL_SECONDS,
            loader=self._build_status,
        )

    async def _build_status(self) -> ObservabilityStatus:
        checked_at = datetime.now(UTC).isoformat()

        prometheus_url = build_public_url(self._settings.PROMETHEUS_PUBLIC_URL)
        grafana_url = build_public_url(self._settings.GRAFANA_PUBLIC_URL)
        tempo_url = build_public_url(self._settings.TEMPO_PUBLIC_URL)

        database, cache, prometheus, grafana, tempo = await asyncio.gather(
            self._database_status(checked_at),
            self._cache_status(checked_at),
            self._http_status(prometheus_url, "Prometheus", checked_at),
            self._http_status(grafana_url, "Grafana", checked_at),
            self._http_status(tempo_url, "Tempo", checked_at),
        )

        return ObservabilityStatus(
            api=ObservabilityStatusItem(
                status="healthy",
                detail="API is responding",
                last_checked_at=checked_at,
            ),
            frontend=ObservabilityStatusItem(
                status="unknown",
                detail="Frontend status check is not configured",
                last_checked_at=checked_at,
            ),
            database=database,
            cache=cache,
            workers=ObservabilityStatusItem(
                status="unknown",
                detail="Worker queue depth check is not configured",
                queue_depth=None,
                last_checked_at=checked_at,
            ),
            background_jobs=ObservabilityStatusItem(
                status="unknown",
                detail="Background job health check is not configured",
                last_checked_at=checked_at,
            ),
            error_rate=ObservabilityStatusItem(
                status="unknown",
                detail="Error-rate summary is available in Grafana when metrics are configured",
                value=None,
                last_checked_at=checked_at,
            ),
            request_latency=ObservabilityStatusItem(
                status="unknown",
                detail=(
                    "Request latency summary is available in Grafana when metrics are configured"
                ),
                value=None,
                last_checked_at=checked_at,
            ),
            prometheus=prometheus,
            grafana=grafana,
            tempo=tempo,
        )

    async def _database_status(self, checked_at: str) -> ObservabilityStatusItem:
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return ObservabilityStatusItem(
                status="healthy",
                detail="Database connection OK",
                last_checked_at=checked_at,
            )
        except Exception:
            return ObservabilityStatusItem(
                status="down",
                detail="Database connection failed",
                last_checked_at=checked_at,
            )

    async def _cache_status(self, checked_at: str) -> ObservabilityStatusItem:
        try:
            await redis_client.ping()
            return ObservabilityStatusItem(
                status="healthy",
                detail="Redis connection OK",
                last_checked_at=checked_at,
            )
        except Exception:
            return ObservabilityStatusItem(
                status="unknown",
                detail="Cache check is unavailable",
                last_checked_at=checked_at,
            )

    async def _http_status(
        self,
        url: str | None,
        service_name: str,
        checked_at: str,
    ) -> ObservabilityStatusItem:
        if not url:
            return ObservabilityStatusItem(
                status="not_configured",
                detail=f"{service_name} URL is not configured",
                url=None,
                last_checked_at=checked_at,
            )

        try:
            client = _get_observability_http_client()
            response = await client.get(url)
            if response.status_code < 500:
                return ObservabilityStatusItem(
                    status="healthy",
                    detail=f"{service_name} is reachable",
                    url=url,
                    last_checked_at=checked_at,
                )
            return ObservabilityStatusItem(
                status="degraded",
                detail=f"{service_name} returned HTTP {response.status_code}",
                url=url,
                last_checked_at=checked_at,
            )
        except httpx.RequestError:
            return ObservabilityStatusItem(
                status="down",
                detail=f"{service_name} is not reachable",
                url=url,
                last_checked_at=checked_at,
            )
