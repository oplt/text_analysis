from __future__ import annotations

from typing import Generic, TypeVar

from fastapi import Query
from pydantic import BaseModel
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):  # noqa: UP046
    items: list[T]
    total: int
    limit: int
    offset: int


class PaginationParams(BaseModel):
    limit: int
    offset: int


def pagination_params(
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> PaginationParams:
    return PaginationParams(limit=limit, offset=offset)


def paginated_response(  # noqa: UP047
    items: list[T],
    *,
    total: int,
    limit: int,
    offset: int,
) -> PaginatedResponse[T]:
    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)


async def paginate_scalars(
    db: AsyncSession,
    stmt: Select,
    *,
    limit: int,
    offset: int,
) -> tuple[list, int]:
    """Execute a paginated scalar select. stmt must not already apply offset/limit."""
    count_stmt = select(func.count()).select_from(stmt.order_by(None).subquery())
    total = int(await db.scalar(count_stmt) or 0)
    result = await db.execute(stmt.offset(offset).limit(limit))
    return list(result.scalars().all()), total
