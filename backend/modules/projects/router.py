from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.pagination import (
    PaginatedResponse,
    PaginationParams,
    paginated_response,
    pagination_params,
)
from backend.modules.identity_access.models import User
from backend.modules.projects.models import Project, ProjectTask
from backend.modules.projects.schemas import (
    ProjectCreate,
    ProjectResponse,
    ProjectTaskAssigneeResponse,
    ProjectTaskCreate,
    ProjectTaskReorderRequest,
    ProjectTaskResponse,
    ProjectTaskUpdate,
)
from backend.modules.projects.service import ProjectsService

router = APIRouter()


def _project_to_response(project: Project) -> ProjectResponse:
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        created_at=project.created_at,
    )


def _task_to_response(task: ProjectTask, assignee: User | None) -> ProjectTaskResponse:
    return ProjectTaskResponse(
        id=task.id,
        project_id=task.project_id,
        title=task.title,
        description=task.description,
        status=task.status,
        priority=task.priority,
        due_date=task.due_date,
        position=task.position,
        assignee=(
            ProjectTaskAssigneeResponse(
                id=assignee.id,
                email=assignee.email,
                full_name=assignee.full_name,
            )
            if assignee
            else None
        ),
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@router.get("", response_model=PaginatedResponse[ProjectResponse])
async def list_projects(
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ProjectsService(db)
    projects, total = await service.list_projects(
        current_user.id, limit=pagination.limit, offset=pagination.offset
    )
    return paginated_response(projects, total=total, limit=pagination.limit, offset=pagination.offset)


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    payload: ProjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ProjectsService(db)
    project = await service.create_project(current_user.id, payload.name, payload.description)
    return _project_to_response(project)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ProjectsService(db)
    project = await service.get_project(current_user.id, project_id)
    return _project_to_response(project)


@router.get("/{project_id}/tasks", response_model=PaginatedResponse[ProjectTaskResponse])
async def list_project_tasks(
    project_id: str,
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ProjectsService(db)
    tasks, total = await service.list_tasks(
        current_user.id,
        project_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_task_to_response(task, assignee) for task, assignee in tasks],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/{project_id}/tasks", response_model=ProjectTaskResponse, status_code=201)
async def create_project_task(
    project_id: str,
    payload: ProjectTaskCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ProjectsService(db)
    task, assignee = await service.create_task(current_user.id, current_user, project_id, payload)
    return _task_to_response(task, assignee)


@router.patch("/{project_id}/tasks/{task_id}", response_model=ProjectTaskResponse)
async def update_project_task(
    project_id: str,
    task_id: str,
    payload: ProjectTaskUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ProjectsService(db)
    task, assignee = await service.update_task(
        current_user.id,
        current_user,
        project_id,
        task_id,
        payload,
    )
    return _task_to_response(task, assignee)


@router.delete("/{project_id}/tasks/{task_id}", status_code=204)
async def delete_project_task(
    project_id: str,
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ProjectsService(db)
    await service.delete_task(current_user.id, project_id, task_id)


@router.put("/{project_id}/tasks/reorder", response_model=list[ProjectTaskResponse])
async def reorder_project_tasks(
    project_id: str,
    payload: ProjectTaskReorderRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ProjectsService(db)
    tasks = await service.reorder_tasks(current_user.id, current_user, project_id, payload)
    return [_task_to_response(task, assignee) for task, assignee in tasks]
