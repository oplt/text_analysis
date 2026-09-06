from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.admin import get_admin_user
from backend.api.deps.db import get_db
from backend.modules.audit.request_logging import log_request_audit_event
from backend.modules.identity_access.models import User
from backend.modules.platform.email_template_service import EmailTemplateService
from backend.modules.platform.schemas import (
    EmailTemplateCreate,
    EmailTemplateResponse,
    EmailTemplateUpdate,
)

router = APIRouter()


def _email_template_to_response(template) -> EmailTemplateResponse:
    return EmailTemplateResponse(
        id=template.id,
        key=template.key,
        name=template.name,
        subject_template=template.subject_template,
        html_template=template.html_template,
        text_template=template.text_template,
        is_active=template.is_active,
        updated_at=template.updated_at,
    )


@router.get("/admin/email-templates", response_model=list[EmailTemplateResponse])
async def list_email_templates(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    service = EmailTemplateService(db)
    return [_email_template_to_response(item) for item in await service.list_email_templates()]


@router.post("/admin/email-templates", response_model=EmailTemplateResponse, status_code=201)
async def create_email_template(
    payload: EmailTemplateCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    service = EmailTemplateService(db)
    template = await service.create_email_template(payload.model_dump())
    await log_request_audit_event(
        db,
        request,
        action="admin.email_template_created",
        actor_user_id=admin.id,
        resource_type="email_template",
        resource_id=template.id,
        metadata={"key": template.key},
    )
    await db.commit()
    return _email_template_to_response(template)


@router.patch("/admin/email-templates/{template_id}", response_model=EmailTemplateResponse)
async def update_email_template(
    template_id: str,
    payload: EmailTemplateUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    service = EmailTemplateService(db)
    template = await service.update_email_template(
        template_id, payload.model_dump(exclude_unset=True)
    )
    await log_request_audit_event(
        db,
        request,
        action="admin.email_template_updated",
        actor_user_id=admin.id,
        resource_type="email_template",
        resource_id=template.id,
    )
    await db.commit()
    return _email_template_to_response(template)
