from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.admin import get_admin_user
from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.audit.request_logging import log_request_audit_event
from backend.modules.identity_access.models import User
from backend.modules.platform.billing_service import BillingService
from backend.modules.platform.schemas import (
    SubscriptionPlanCreate,
    SubscriptionPlanResponse,
    SubscriptionPlanUpdate,
    SubscriptionSelectionRequest,
    UserSubscriptionResponse,
)

router = APIRouter()


def _plan_to_response(plan) -> SubscriptionPlanResponse:
    return SubscriptionPlanResponse(
        id=plan.id,
        code=plan.code,
        name=plan.name,
        description=plan.description,
        price_cents=plan.price_cents,
        interval=plan.interval,
        is_active=plan.is_active,
        is_default=plan.is_default,
        features=plan.features_json,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )


def _subscription_to_response(subscription, plan) -> UserSubscriptionResponse:
    return UserSubscriptionResponse(
        id=subscription.id,
        status=subscription.status,
        cancel_at_period_end=subscription.cancel_at_period_end,
        started_at=subscription.started_at,
        current_period_end=subscription.current_period_end,
        created_at=subscription.created_at,
        updated_at=subscription.updated_at,
        plan=_plan_to_response(plan),
    )


@router.get("/billing/plans", response_model=list[SubscriptionPlanResponse])
async def list_subscription_plans(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    service = BillingService(db)
    await service.ensure_module_enabled("billing")
    plans = await service.list_plans()
    return [_plan_to_response(plan) for plan in plans if plan.is_active]


@router.get("/billing/subscription", response_model=UserSubscriptionResponse | None)
async def get_subscription(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = BillingService(db)
    subscription = await service.get_subscription_for_user(current_user)
    if subscription is None:
        return None
    plan = await service.repo.get_plan_by_id(subscription.plan_id)
    if plan is None:
        return None
    return _subscription_to_response(subscription, plan)


@router.put("/billing/subscription", response_model=UserSubscriptionResponse)
async def select_subscription_plan(
    payload: SubscriptionSelectionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = BillingService(db)
    subscription = await service.select_plan_for_user(current_user, payload.plan_code)
    plan = await service.repo.get_plan_by_id(subscription.plan_id)
    assert plan is not None
    return _subscription_to_response(subscription, plan)


@router.get("/admin/plans", response_model=list[SubscriptionPlanResponse])
async def list_admin_plans(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    service = BillingService(db)
    return [_plan_to_response(plan) for plan in await service.list_plans()]


@router.post("/admin/plans", response_model=SubscriptionPlanResponse, status_code=201)
async def create_plan(
    payload: SubscriptionPlanCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    service = BillingService(db)
    plan = await service.create_plan(payload.model_dump())
    await log_request_audit_event(
        db,
        request,
        action="admin.plan_created",
        actor_user_id=admin.id,
        resource_type="subscription_plan",
        resource_id=plan.id,
        metadata={"code": plan.code},
    )
    await db.commit()
    return _plan_to_response(plan)


@router.patch("/admin/plans/{plan_id}", response_model=SubscriptionPlanResponse)
async def update_plan(
    plan_id: str,
    payload: SubscriptionPlanUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    service = BillingService(db)
    plan = await service.update_plan(plan_id, payload.model_dump(exclude_unset=True))
    await log_request_audit_event(
        db,
        request,
        action="admin.plan_updated",
        actor_user_id=admin.id,
        resource_type="subscription_plan",
        resource_id=plan.id,
    )
    await db.commit()
    return _plan_to_response(plan)
