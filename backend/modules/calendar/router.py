from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.calendar.schemas import CalendarItemCreate, CalendarItemResponse
from backend.modules.calendar.service import CalendarService
from backend.modules.identity_access.models import User

router = APIRouter()


@router.get("/items", response_model=list[CalendarItemResponse])
async def list_calendar_items(
    start_date: date,
    end_date: date,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = CalendarService(db)
    return await service.list_items(current_user, start_date, end_date)


@router.post("/items", response_model=CalendarItemResponse, status_code=201)
async def create_calendar_item(
    payload: CalendarItemCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = CalendarService(db)
    return await service.create_item(current_user, payload)

