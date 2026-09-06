from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.pagination import DEFAULT_PAGE_LIMIT, paginate_scalars
from backend.modules.platform.models import (
    ApiKey,
    EmailTemplate,
    FeatureFlag,
    SubscriptionPlan,
    UserSubscription,
    WebhookEndpoint,
)


class PlatformRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_plans(self) -> list[SubscriptionPlan]:
        result = await self.db.execute(
            select(SubscriptionPlan).order_by(SubscriptionPlan.price_cents.asc())
        )
        return list(result.scalars().all())

    async def get_plan_by_id(self, plan_id: str) -> SubscriptionPlan | None:
        result = await self.db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.id == plan_id)
        )
        return result.scalar_one_or_none()

    async def get_plan_by_code(self, code: str) -> SubscriptionPlan | None:
        result = await self.db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.code == code)
        )
        return result.scalar_one_or_none()

    async def create_plan(self, **kwargs) -> SubscriptionPlan:
        plan = SubscriptionPlan(**kwargs)
        self.db.add(plan)
        await self.db.flush()
        return plan

    def add_plans(self, rows: list[dict]) -> None:
        self.db.add_all([SubscriptionPlan(**row) for row in rows])

    async def upsert_default_plans(self, rows: list[dict]) -> int:
        statement = (
            insert(SubscriptionPlan).values(rows).on_conflict_do_nothing(index_elements=["code"])
        )
        result = await self.db.execute(statement)
        return int(result.rowcount or 0)

    async def list_subscriptions(self) -> list[UserSubscription]:
        result = await self.db.execute(
            select(UserSubscription).order_by(UserSubscription.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_subscription_for_user(self, user_id: str) -> UserSubscription | None:
        result = await self.db.execute(
            select(UserSubscription).where(UserSubscription.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def create_subscription(self, **kwargs) -> UserSubscription:
        subscription = UserSubscription(**kwargs)
        self.db.add(subscription)
        await self.db.flush()
        return subscription

    async def list_api_keys_for_user(
        self,
        user_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[ApiKey], int]:
        stmt = select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc())
        return await paginate_scalars(self.db, stmt, limit=limit, offset=offset)

    async def get_api_key_for_user(self, user_id: str, api_key_id: str) -> ApiKey | None:
        result = await self.db.execute(
            select(ApiKey).where(ApiKey.user_id == user_id, ApiKey.id == api_key_id)
        )
        return result.scalar_one_or_none()

    async def create_api_key(self, **kwargs) -> ApiKey:
        api_key = ApiKey(**kwargs)
        self.db.add(api_key)
        await self.db.flush()
        return api_key

    async def list_webhooks_for_user(
        self,
        user_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[WebhookEndpoint], int]:
        stmt = (
            select(WebhookEndpoint)
            .where(WebhookEndpoint.user_id == user_id)
            .order_by(WebhookEndpoint.created_at.desc())
        )
        return await paginate_scalars(self.db, stmt, limit=limit, offset=offset)

    async def get_webhook_for_user(self, user_id: str, webhook_id: str) -> WebhookEndpoint | None:
        result = await self.db.execute(
            select(WebhookEndpoint).where(
                WebhookEndpoint.user_id == user_id, WebhookEndpoint.id == webhook_id
            )
        )
        return result.scalar_one_or_none()

    async def create_webhook(self, **kwargs) -> WebhookEndpoint:
        webhook = WebhookEndpoint(**kwargs)
        self.db.add(webhook)
        await self.db.flush()
        return webhook

    async def delete_webhook(self, webhook: WebhookEndpoint) -> None:
        await self.db.delete(webhook)
        await self.db.flush()

    async def list_feature_flags(self) -> list[FeatureFlag]:
        result = await self.db.execute(select(FeatureFlag).order_by(FeatureFlag.key.asc()))
        return list(result.scalars().all())

    async def get_feature_flag_by_key(self, key: str) -> FeatureFlag | None:
        result = await self.db.execute(select(FeatureFlag).where(FeatureFlag.key == key))
        return result.scalar_one_or_none()

    async def get_feature_flag_by_id(self, feature_flag_id: str) -> FeatureFlag | None:
        result = await self.db.execute(select(FeatureFlag).where(FeatureFlag.id == feature_flag_id))
        return result.scalar_one_or_none()

    async def create_feature_flag(self, **kwargs) -> FeatureFlag:
        flag = FeatureFlag(**kwargs)
        self.db.add(flag)
        await self.db.flush()
        return flag

    def add_feature_flags(self, rows: list[dict]) -> None:
        self.db.add_all([FeatureFlag(**row) for row in rows])

    async def upsert_default_feature_flags(self, rows: list[dict]) -> int:
        statement = insert(FeatureFlag).values(rows).on_conflict_do_nothing(index_elements=["key"])
        result = await self.db.execute(statement)
        return int(result.rowcount or 0)

    async def list_email_templates(self) -> list[EmailTemplate]:
        result = await self.db.execute(select(EmailTemplate).order_by(EmailTemplate.key.asc()))
        return list(result.scalars().all())

    async def get_email_template_by_key(self, key: str) -> EmailTemplate | None:
        result = await self.db.execute(select(EmailTemplate).where(EmailTemplate.key == key))
        return result.scalar_one_or_none()

    async def get_email_template_by_id(self, template_id: str) -> EmailTemplate | None:
        result = await self.db.execute(select(EmailTemplate).where(EmailTemplate.id == template_id))
        return result.scalar_one_or_none()

    async def create_email_template(self, **kwargs) -> EmailTemplate:
        template = EmailTemplate(**kwargs)
        self.db.add(template)
        await self.db.flush()
        return template

    def add_email_templates(self, rows: list[dict]) -> None:
        self.db.add_all([EmailTemplate(**row) for row in rows])

    async def upsert_default_email_templates(self, rows: list[dict]) -> int:
        statement = (
            insert(EmailTemplate).values(rows).on_conflict_do_nothing(index_elements=["key"])
        )
        result = await self.db.execute(statement)
        return int(result.rowcount or 0)
