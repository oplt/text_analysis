import asyncio
from datetime import date

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.lib.resource_cache import (
    calendar_items_cache_key,
    get_cached_model_items,
    invalidate_calendar_cache,
    set_cached_model_items,
)
from backend.modules.calendar.models import CalendarEntry
from backend.modules.calendar.repository import CalendarRepository
from backend.modules.calendar.schemas import CalendarItemCreate, CalendarItemResponse
from backend.modules.identity_access.models import User
from backend.modules.projects.models import Project, ProjectTask
from backend.modules.projects.repository import ProjectsRepository
from backend.modules.projects.schemas import ProjectTaskCreate
from backend.modules.projects.service import ProjectsService


class CalendarService:
    MAX_RANGE_DAYS = 90

    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = CalendarRepository(db)
        self.projects_repo = ProjectsRepository(db)
        self.projects_service = ProjectsService(db)

    async def list_items(
        self,
        user: User,
        start_date: date,
        end_date: date,
    ) -> list[CalendarItemResponse]:
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        if (end_date - start_date).days > self.MAX_RANGE_DAYS:
            raise HTTPException(
                status_code=400,
                detail=f"Date range cannot exceed {self.MAX_RANGE_DAYS} days",
            )

        cache_key = calendar_items_cache_key(user.id, start_date, end_date)
        cached = await get_cached_model_items(cache_key, CalendarItemResponse)
        if cached is not None:
            return cached

        entries, tasks = await asyncio.gather(
            self.repo.list_entries_by_user_and_range(user.id, start_date, end_date),
            self.projects_repo.list_tasks_due_for_user(user.id, start_date, end_date),
        )

        items = [self._entry_to_response(entry) for entry in entries]
        items.extend(self._task_to_response(task, project) for task, project in tasks)
        items.sort(
            key=lambda item: (
                item.date.isoformat(),
                item.start_time.isoformat() if item.start_time else "99:99:99",
                item.created_at.isoformat(),
            )
        )
        await set_cached_model_items(
            cache_key,
            items,
            ttl_seconds=settings.CACHE_CALENDAR_TTL_SECONDS,
        )
        return items

    async def create_item(self, user: User, payload: CalendarItemCreate) -> CalendarItemResponse:
        if payload.type == "task":
            response = await self._create_task_item(user, payload)
        else:
            response = await self._create_entry_item(user, payload)
        await invalidate_calendar_cache(user.id)
        return response

    async def _create_entry_item(
        self,
        user: User,
        payload: CalendarItemCreate,
    ) -> CalendarItemResponse:
        if payload.project_id:
            raise HTTPException(status_code=400, detail="Project can only be set for task items")
        if payload.assignee_id:
            raise HTTPException(status_code=400, detail="Assignee can only be set for task items")
        if payload.priority not in {None, "medium"}:
            raise HTTPException(status_code=400, detail="Priority can only be set for task items")
        if payload.end_time and not payload.start_time:
            raise HTTPException(
                status_code=400,
                detail="Start time is required when end time is set",
            )
        if payload.start_time and payload.end_time and payload.end_time <= payload.start_time:
            raise HTTPException(status_code=400, detail="End time must be after start time")

        entry = await self.repo.create_entry(
            user_id=user.id,
            entry_type=payload.type,
            title=payload.title.strip(),
            description=payload.description.strip() if payload.description else None,
            scheduled_for=payload.date,
            start_time=payload.start_time,
            end_time=payload.end_time,
        )
        await self.db.commit()
        await self.db.refresh(entry)
        return self._entry_to_response(entry)

    async def _create_task_item(
        self,
        user: User,
        payload: CalendarItemCreate,
    ) -> CalendarItemResponse:
        if not payload.project_id:
            raise HTTPException(status_code=400, detail="Project is required when creating a task")

        task, _ = await self.projects_service.create_task(
            user.id,
            user,
            payload.project_id,
            ProjectTaskCreate(
                title=payload.title.strip(),
                description=payload.description.strip() if payload.description else None,
                status="todo",
                priority=payload.priority or "medium",
                due_date=payload.date,
                assignee_id=payload.assignee_id or user.id,
            ),
        )
        project = await self.projects_service.get_project(user.id, payload.project_id)
        return self._task_to_response(task, project)

    @staticmethod
    def _entry_to_response(entry: CalendarEntry) -> CalendarItemResponse:
        return CalendarItemResponse(
            id=entry.id,
            source="planner",
            type=entry.type,
            title=entry.title,
            description=entry.description,
            date=entry.scheduled_for,
            start_time=entry.start_time,
            end_time=entry.end_time,
            created_at=entry.created_at,
        )

    @staticmethod
    def _task_to_response(task: ProjectTask, project: Project) -> CalendarItemResponse:
        if not task.due_date:
            raise HTTPException(status_code=500, detail="Task is missing a due date")

        return CalendarItemResponse(
            id=task.id,
            source="task",
            type="task",
            title=task.title,
            description=task.description,
            date=task.due_date,
            project_id=project.id,
            project_name=project.name,
            priority=task.priority,
            status=task.status,
            created_at=task.created_at,
        )
