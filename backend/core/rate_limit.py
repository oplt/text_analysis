import logging

from fastapi import HTTPException, Request

from backend.core.cache import get_async_redis_client

logger = logging.getLogger("backend.rate_limit")


def _build_rate_limit_exception(ttl: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=f"Too many attempts. Try again in {ttl} seconds.",
        headers={"Retry-After": str(ttl)},
    )


async def check_rate_limit(key: str, max_attempts: int, window_seconds: int) -> None:
    try:
        client = get_async_redis_client()
        count = await client.incr(key)
        if count == 1:
            await client.expire(key, window_seconds)
        if count > max_attempts:
            ttl = await client.ttl(key)
            raise _build_rate_limit_exception(ttl)
    except HTTPException:
        raise
    except Exception:
        logger.warning("rate limit check skipped for key=%s", key, exc_info=True)


async def enforce_rate_limit(key: str, max_attempts: int) -> None:
    try:
        client = get_async_redis_client()
        count = await client.get(key)
        if count is None:
            return
        if int(count) > max_attempts:
            ttl = await client.ttl(key)
            raise _build_rate_limit_exception(ttl)
    except HTTPException:
        raise
    except Exception:
        logger.warning("rate limit enforce skipped for key=%s", key, exc_info=True)


async def increment_rate_limit(key: str, window_seconds: int) -> int:
    try:
        client = get_async_redis_client()
        count = await client.incr(key)
        if count == 1:
            await client.expire(key, window_seconds)
        return count
    except Exception:
        logger.warning("rate limit increment skipped for key=%s", key, exc_info=True)
        return 0


async def clear_rate_limit(key: str) -> None:
    try:
        await get_async_redis_client().delete(key)
    except Exception:
        logger.warning("rate limit clear skipped for key=%s", key, exc_info=True)


def auth_rate_limit_key(request: Request, email: str) -> str:
    client_ip = request.client.host if request.client else "unknown"
    return f"rate_limit:auth:{client_ip}:{email}"
