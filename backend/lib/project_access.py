from __future__ import annotations

from typing import Protocol

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.projects.models import Project
from backend.modules.projects.repository import ProjectsRepository


class ProjectAccessPort(Protocol):
    async def ensure_project_access(self, user_id: str, project_id: str) -> None: ...

    async def get_project_for_user(self, project_id: str, user_id: str) -> Project | None: ...

    async def filter_accessible_project_ids(
        self,
        user_id: str,
        project_ids: set[str],
    ) -> set[str]: ...


class SqlAlchemyProjectAccessPort:
    def __init__(self, db: AsyncSession):
        self._repo = ProjectsRepository(db)

    async def ensure_project_access(self, user_id: str, project_id: str) -> None:
        project = await self._repo.get_by_id_for_user(project_id, user_id)
        if project is None:
            raise HTTPException(status_code=403, detail="Project access denied")

    async def get_project_for_user(self, project_id: str, user_id: str) -> Project | None:
        return await self._repo.get_by_id_for_user(project_id, user_id)

    async def filter_accessible_project_ids(
        self,
        user_id: str,
        project_ids: set[str],
    ) -> set[str]:
        return await self._repo.filter_accessible_project_ids(user_id, project_ids)
