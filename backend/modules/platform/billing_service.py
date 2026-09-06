from datetime import UTC, datetime, timedelta

from fastapi import HTTPException

from backend.modules.identity_access.models import User
from backend.modules.platform.config_service import PlatformConfigService
from backend.modules.platform.models import SubscriptionPlan, UserSubscription


class BillingService(PlatformConfigService):
    async def list_plans(self) -> list[SubscriptionPlan]:
        return await self.repo.list_plans()

    async def create_plan(self, payload: dict) -> SubscriptionPlan:
        if await self.repo.get_plan_by_code(payload["code"]) is not None:
            raise HTTPException(status_code=409, detail="A plan with this code already exists")

        plan = await self.repo.create_plan(
            code=payload["code"],
            name=payload["name"],
            description=payload.get("description"),
            price_cents=payload["price_cents"],
            interval=payload["interval"],
            is_active=True,
            is_default=payload.get("is_default", False),
            features_json=payload.get("features", []),
        )
        await self._normalize_default_plan(plan)
        await self.db.commit()
        await self.db.refresh(plan)
        return plan

    async def update_plan(self, plan_id: str, payload: dict) -> SubscriptionPlan:
        plan = await self.repo.get_plan_by_id(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")

        for field, value in payload.items():
            if field == "features":
                plan.features_json = value
            else:
                setattr(plan, field, value)

        await self._normalize_default_plan(plan)
        await self.db.commit()
        await self.db.refresh(plan)
        return plan

    async def get_subscription_for_user(self, user: User) -> UserSubscription | None:
        await self.ensure_module_enabled("billing")
        return await self.repo.get_subscription_for_user(user.id)

    async def select_plan_for_user(self, user: User, plan_code: str) -> UserSubscription:
        await self.ensure_module_enabled("billing")
        plan = await self.repo.get_plan_by_code(plan_code)
        if not plan or not plan.is_active:
            raise HTTPException(status_code=404, detail="Plan not found")

        subscription = await self.repo.get_subscription_for_user(user.id)
        current_period_end = self._calculate_period_end(plan.interval)
        if subscription is None:
            subscription = await self.repo.create_subscription(
                user_id=user.id,
                plan_id=plan.id,
                status="active",
                cancel_at_period_end=False,
                started_at=datetime.now(UTC),
                current_period_end=current_period_end,
            )
        else:
            subscription.plan_id = plan.id
            subscription.status = "active"
            subscription.cancel_at_period_end = False
            subscription.current_period_end = current_period_end

        await self.db.commit()
        await self.db.refresh(subscription)
        return subscription

    async def _normalize_default_plan(self, active_plan: SubscriptionPlan) -> None:
        if not active_plan.is_default:
            return
        plans = await self.repo.list_plans()
        for plan in plans:
            if plan.id != active_plan.id and plan.is_default:
                plan.is_default = False

    @staticmethod
    def _calculate_period_end(interval: str) -> datetime | None:
        now = datetime.now(UTC)
        normalized = interval.lower()
        if normalized == "month":
            return now + timedelta(days=30)
        if normalized == "year":
            return now + timedelta(days=365)
        if normalized == "week":
            return now + timedelta(days=7)
        if normalized == "lifetime":
            return None
        return now + timedelta(days=30)
