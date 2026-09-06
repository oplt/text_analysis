import asyncio
from datetime import date

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from backend.lib.resource_cache import (
    get_cached_model_list,
    invalidate_calendar_cache,
    invalidate_project_list_cache,
    project_list_cache_key,
    set_cached_model_list,
)
from backend.modules.identity_access.models import User
from backend.modules.notifications.repository import NotificationsRepository
from backend.modules.projects.models import Project, ProjectTask
from backend.modules.projects.repository import ProjectsRepository
from backend.modules.projects.schemas import (
    ProjectMemberCreate,
    ProjectMemberResponse,
    ProjectResponse,
    ProjectTaskCreate,
    ProjectTaskReorderRequest,
    ProjectTaskUpdate,
)
from backend.modules.users.repository import UsersRepository


class ProjectsService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = ProjectsRepository(db)
        self.users_repo = UsersRepository(db)
        self.notifications_repo = NotificationsRepository(db)

    async def create_project(self, owner_id: str, name: str, description: str | None) -> Project:
        project = await self.repo.create(owner_id, name, description)
        await self.db.commit()
        await self.db.refresh(project)
        await invalidate_project_list_cache(owner_id)
        return project

    async def list_projects(
        self,
        user_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[ProjectResponse], int]:
        cache_key = project_list_cache_key(user_id, limit, offset)
        cached = await get_cached_model_list(cache_key, ProjectResponse)
        if cached is not None:
            return cached

        projects, total = await self.repo.list_accessible_by_user(
            user_id, limit=limit, offset=offset
        )
        items = [
            ProjectResponse(
                id=project.id,
                name=project.name,
                description=project.description,
                created_at=project.created_at,
            )
            for project in projects
        ]
        await set_cached_model_list(
            cache_key,
            items,
            total=total,
            ttl_seconds=settings.CACHE_PROJECT_LIST_TTL_SECONDS,
        )
        return items, total

    async def get_project(self, user_id: str, project_id: str) -> Project:
        return await self._get_project_or_404(user_id, project_id)

    async def list_members(self, user_id: str, project_id: str) -> list[ProjectMemberResponse]:
        project = await self._get_project_or_404(user_id, project_id)
        return [
            ProjectMemberResponse(
                user_id=user.id,
                email=user.email,
                full_name=user.full_name,
                role=member.role,
            )
            for member, user in await self.repo.list_members_with_users(project.id)
        ]

    async def add_or_update_member(
        self,
        user_id: str,
        project_id: str,
        payload: ProjectMemberCreate,
    ) -> ProjectMemberResponse:
        project = await self._get_project_owner_or_404(user_id, project_id)
        member_user = await self.users_repo.get_active_user_by_id(payload.user_id)
        if not member_user:
            raise HTTPException(status_code=404, detail="Member not found")
        member = await self.repo.get_membership(project.id, member_user.id)
        if member:
            if member.role == "owner":
                raise HTTPException(
                    status_code=400,
                    detail="The project owner role cannot be changed",
                )
            member.role = payload.role
            await self.db.flush()
        else:
            member = await self.repo.add_member(project.id, member_user.id, payload.role)
        await self.db.commit()
        await self._invalidate_task_view_caches(user_id, member_user.id)
        return ProjectMemberResponse(
            user_id=member_user.id,
            email=member_user.email,
            full_name=member_user.full_name,
            role=member.role,
        )

    async def remove_member(self, user_id: str, project_id: str, member_user_id: str) -> None:
        project = await self._get_project_owner_or_404(user_id, project_id)
        member = await self.repo.get_membership(project.id, member_user_id)
        if not member:
            raise HTTPException(status_code=404, detail="Project member not found")
        if member.role == "owner":
            raise HTTPException(status_code=400, detail="The project owner cannot be removed")
        await self.repo.delete_member(member)
        await self.db.commit()
        await self._invalidate_task_view_caches(user_id, member_user_id)

    async def list_tasks(
        self,
        user_id: str,
        project_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[tuple[ProjectTask, User | None]], int]:
        project = await self._get_project_or_404(user_id, project_id)
        return await self.repo.list_tasks_with_assignees(
            project.id, limit=limit, offset=offset
        )

    async def create_task(
        self,
        user_id: str,
        actor: User,
        project_id: str,
        payload: ProjectTaskCreate,
    ) -> tuple[ProjectTask, User | None]:
        project = await self._get_project_for_write_or_404(user_id, project_id)
        assignee = await self._get_assignee_or_404(project.id, payload.assignee_id)
        position = await self.repo.get_next_task_position(project.id, payload.status)
        task = await self.repo.create_task(
            project_id=project.id,
            title=payload.title,
            description=payload.description,
            status=payload.status,
            priority=payload.priority,
            due_date=payload.due_date,
            assignee_id=assignee.id if assignee else None,
            position=position,
        )

        await self._notify_assignment(project, task, actor, None, assignee)
        await self._notify_due_date_change(project, task, actor, None, assignee)

        await self.db.commit()
        task_row = await self.repo.get_task_with_assignee(project.id, task.id)
        if not task_row:
            raise HTTPException(status_code=500, detail="Failed to load created task")
        await self._invalidate_project_view_caches(project.id)
        return task_row

    async def update_task(
        self,
        user_id: str,
        actor: User,
        project_id: str,
        task_id: str,
        payload: ProjectTaskUpdate,
    ) -> tuple[ProjectTask, User | None]:
        project = await self._get_project_for_write_or_404(user_id, project_id)
        task = await self.repo.get_task_by_id(project.id, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        fields_set = payload.model_fields_set
        previous_status = task.status
        previous_due_date = task.due_date
        previous_assignee_id = task.assignee_id

        if "title" in fields_set:
            task.title = payload.title or task.title
        if "description" in fields_set:
            task.description = payload.description
        if "priority" in fields_set and payload.priority is not None:
            task.priority = payload.priority
        if "due_date" in fields_set:
            task.due_date = payload.due_date

        assignee = None
        if "assignee_id" in fields_set:
            assignee = await self._get_assignee_or_404(project.id, payload.assignee_id)
            task.assignee_id = assignee.id if assignee else None
        elif task.assignee_id:
            assignee = await self.users_repo.get_active_user_by_id(task.assignee_id)

        if "status" in fields_set and payload.status is not None and payload.status != task.status:
            task.status = payload.status
            task.position = await self.repo.get_next_task_position(project.id, payload.status)

        await self._normalize_positions(project.id)
        await self._notify_assignment(project, task, actor, previous_assignee_id, assignee)
        await self._notify_due_date_change(project, task, actor, previous_due_date, assignee)
        await self._notify_status_change(project, task, actor, previous_status)

        await self.db.commit()
        task_row = await self.repo.get_task_with_assignee(project.id, task.id)
        if not task_row:
            raise HTTPException(status_code=500, detail="Failed to load updated task")
        await self._invalidate_project_view_caches(project.id)
        return task_row

    async def delete_task(self, user_id: str, project_id: str, task_id: str) -> None:
        project = await self._get_project_for_write_or_404(user_id, project_id)
        task = await self.repo.get_task_by_id(project.id, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        await self.repo.delete_task(task)
        await self._normalize_positions(project.id)
        await self.db.commit()
        await self._invalidate_project_view_caches(project.id)

    async def reorder_tasks(
        self,
        user_id: str,
        actor: User,
        project_id: str,
        payload: ProjectTaskReorderRequest,
    ) -> list[tuple[ProjectTask, User | None]]:
        project = await self._get_project_for_write_or_404(user_id, project_id)
        task_rows, _ = await self.repo.list_tasks_with_assignees(
            project.id, limit=MAX_PAGE_LIMIT
        )
        tasks_by_id = {task.id: task for task, _ in task_rows}
        previous_status_by_id = {task.id: task.status for task, _ in task_rows}

        seen_ids: list[str] = []
        for column in payload.columns:
            for position, task_id in enumerate(column.task_ids):
                task = tasks_by_id.get(task_id)
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found in reorder payload")
                task.status = column.status
                task.position = position
                seen_ids.append(task_id)

        if len(seen_ids) != len(tasks_by_id) or set(seen_ids) != set(tasks_by_id):
            raise HTTPException(
                status_code=400,
                detail="Reorder payload must include every task exactly once",
            )

        await self._normalize_positions(project.id)

        await asyncio.gather(
            *[
                self._notify_status_change(project, task, actor, previous_status_by_id[task.id])
                for task, _ in task_rows
            ]
        )

        await self.db.commit()
        rows, _ = await self.repo.list_tasks_with_assignees(project.id, limit=MAX_PAGE_LIMIT)
        return rows

    async def _get_project_or_404(self, user_id: str, project_id: str) -> Project:
        project = await self.repo.get_by_id_for_user(project_id, user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    async def _get_project_for_write_or_404(self, user_id: str, project_id: str) -> Project:
        project = await self._get_project_or_404(user_id, project_id)
        membership = await self.repo.get_membership(project.id, user_id)
        if not membership or membership.role not in {"owner", "editor"}:
            raise HTTPException(status_code=403, detail="Project editor access required")
        return project

    async def _get_project_owner_or_404(self, user_id: str, project_id: str) -> Project:
        project = await self._get_project_or_404(user_id, project_id)
        membership = await self.repo.get_membership(project.id, user_id)
        if not membership or membership.role != "owner":
            raise HTTPException(status_code=403, detail="Project owner access required")
        return project

    async def _get_assignee_or_404(self, project_id: str, assignee_id: str | None) -> User | None:
        if not assignee_id:
            return None
        assignee = await self.users_repo.get_active_user_by_id(assignee_id)
        if not assignee:
            raise HTTPException(status_code=404, detail="Assignee not found")
        if not await self.repo.get_membership(project_id, assignee.id):
            raise HTTPException(status_code=422, detail="Assignee must be a project member")
        return assignee

    async def _normalize_positions(self, project_id: str) -> None:
        rows, _ = await self.repo.list_tasks_with_assignees(project_id, limit=MAX_PAGE_LIMIT)
        grouped: dict[str, list[ProjectTask]] = {}
        for task, _ in rows:
            grouped.setdefault(task.status, []).append(task)

        for tasks in grouped.values():
            for index, task in enumerate(tasks):
                task.position = index

        await self.db.flush()

    async def _invalidate_project_view_caches(self, project_id: str) -> None:
        member_rows = await self.repo.list_members_with_users(project_id)
        user_ids = [member.user_id for member, _ in member_rows]
        await self._invalidate_task_view_caches(*user_ids)

    @staticmethod
    async def _invalidate_task_view_caches(*user_ids: str | None) -> None:
        affected_users = {user_id for user_id in user_ids if user_id}
        await asyncio.gather(
            *[
                operation(user_id)
                for user_id in affected_users
                for operation in (invalidate_calendar_cache, invalidate_project_list_cache)
            ]
        )

    async def _notify_assignment(
        self,
        project: Project,
        task: ProjectTask,
        actor: User,
        previous_assignee_id: str | None,
        assignee: User | None,
    ) -> None:
        if not assignee or assignee.id == previous_assignee_id or assignee.id == actor.id:
            return

        await self.notifications_repo.create(
            user_id=assignee.id,
            type="task_assigned",
            title=f"Task assigned: {task.title}",
            body=(
                f"{self._actor_label(actor)} assigned you the task "
                f"\"{task.title}\" in project \"{project.name}\"."
            ),
        )

    async def _notify_due_date_change(
        self,
        project: Project,
        task: ProjectTask,
        actor: User,
        previous_due_date: date | None,
        assignee: User | None,
    ) -> None:
        if (
            not assignee
            or assignee.id == actor.id
            or task.due_date is None
            or task.due_date == previous_due_date
        ):
            return

        await self.notifications_repo.create(
            user_id=assignee.id,
            type="task_due_date_updated",
            title=f"Due date updated: {task.title}",
            body=(
                f"{self._actor_label(actor)} set the due date for \"{task.title}\" "
                f"to {task.due_date.isoformat()} in project \"{project.name}\"."
            ),
        )

    async def _notify_status_change(
        self,
        project: Project,
        task: ProjectTask,
        actor: User,
        previous_status: str,
    ) -> None:
        if task.status == previous_status or project.owner_id == actor.id:
            return

        if task.status not in {"review", "done"}:
            return

        target_label = "review" if task.status == "review" else "done"
        await self.notifications_repo.create(
            user_id=project.owner_id,
            type="task_status_changed",
            title=f"Task moved to {target_label}: {task.title}",
            body=(
                f"{self._actor_label(actor)} moved \"{task.title}\" to {target_label} "
                f"in project \"{project.name}\"."
            ),
        )

    @staticmethod
    def _actor_label(actor: User) -> str:
        return actor.full_name or actor.email
