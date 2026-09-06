from fastapi import HTTPException, Request

from backend.core.cache import redis_client


def _build_rate_limit_exception(ttl: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=f"Too many attempts. Try again in {ttl} seconds.",
        headers={"Retry-After": str(ttl)},
    )


async def check_rate_limit(key: str, max_attempts: int, window_seconds: int) -> None:
    count = await redis_client.incr(key)
    if count == 1:
        await redis_client.expire(key, window_seconds)
    if count > max_attempts:
        ttl = await redis_client.ttl(key)
        raise _build_rate_limit_exception(ttl)


async def enforce_rate_limit(key: str, max_attempts: int) -> None:
    count = await redis_client.get(key)
    if count is None:
        return
    if int(count) > max_attempts:
        ttl = await redis_client.ttl(key)
        raise _build_rate_limit_exception(ttl)


async def increment_rate_limit(key: str, window_seconds: int) -> int:
    count = await redis_client.incr(key)
    if count == 1:
        await redis_client.expire(key, window_seconds)
    return count


async def clear_rate_limit(key: str) -> None:
    await redis_client.delete(key)


def auth_rate_limit_key(request: Request, email: str) -> str:
    client_ip = request.client.host if request.client else "unknown"
    return f"rate_limit:auth:{client_ip}:{email}"
