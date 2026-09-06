from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.core.schemas import RequestModel

TaskStatus = Literal["backlog", "todo", "in_progress", "review", "done"]
TaskPriority = Literal["low", "medium", "high", "urgent"]


class ProjectCreate(RequestModel):
    name: str = Field(min_length=2, max_length=255)
    description: str | None = Field(default=None, max_length=1000)


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str | None
    created_at: datetime


class ProjectTaskAssigneeResponse(BaseModel):
    id: str
    email: str
    full_name: str | None


class ProjectTaskCreate(RequestModel):
    title: str = Field(min_length=2, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    status: TaskStatus = "backlog"
    priority: TaskPriority = "medium"
    due_date: date | None = None
    assignee_id: str | None = None


class ProjectTaskUpdate(RequestModel):
    title: str | None = Field(default=None, min_length=2, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    status: TaskStatus | None = None
    priority: TaskPriority | None = None
    due_date: date | None = None
    assignee_id: str | None = None


class ProjectTaskResponse(BaseModel):
    id: str
    project_id: str
    title: str
    description: str | None
    status: TaskStatus
    priority: TaskPriority
    due_date: date | None
    position: int
    assignee: ProjectTaskAssigneeResponse | None = None
    created_at: datetime
    updated_at: datetime


class ProjectTaskReorderColumn(RequestModel):
    status: TaskStatus
    task_ids: list[str]


class ProjectTaskReorderRequest(RequestModel):
    columns: list[ProjectTaskReorderColumn]
