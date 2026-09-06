import hashlib
import hmac
import ipaddress
import json
import logging
import secrets
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

from backend.core.pagination import DEFAULT_PAGE_LIMIT
from backend.modules.identity_access.models import User
from backend.modules.platform.config_service import PlatformConfigService
from backend.modules.platform.models import WebhookEndpoint

logger = logging.getLogger(__name__)


class WebhookService(PlatformConfigService):
    async def list_webhooks_for_user(
        self,
        user: User,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[WebhookEndpoint], int]:
        await self.ensure_module_enabled("webhooks")
        return await self.repo.list_webhooks_for_user(user.id, limit=limit, offset=offset)

    async def create_webhook_for_user(
        self,
        user: User,
        *,
        target_url: str,
        description: str | None,
        events: list[str],
    ) -> WebhookEndpoint:
        await self.ensure_module_enabled("webhooks")
        self._validate_webhook_target(target_url)
        webhook = await self.repo.create_webhook(
            user_id=user.id,
            target_url=target_url,
            description=description,
            secret=secrets.token_urlsafe(24),
            is_active=True,
            events_json=events,
        )
        await self.db.commit()
        await self.db.refresh(webhook)
        logger.info("Webhook created user=%s webhook=%s", user.id, webhook.id)
        return webhook

    async def update_webhook_for_user(
        self, user: User, webhook_id: str, payload: dict
    ) -> WebhookEndpoint:
        await self.ensure_module_enabled("webhooks")
        webhook = await self.repo.get_webhook_for_user(user.id, webhook_id)
        if not webhook:
            raise HTTPException(status_code=404, detail="Webhook endpoint not found")

        for field, value in payload.items():
            if field == "events":
                webhook.events_json = value
            elif field == "target_url" and value is not None:
                self._validate_webhook_target(str(value))
                webhook.target_url = str(value)
            elif value is not None:
                setattr(webhook, field, value)

        await self.db.commit()
        await self.db.refresh(webhook)
        logger.info(
            "Webhook updated user=%s webhook=%s fields=%s", user.id, webhook.id, sorted(payload)
        )
        return webhook

    async def delete_webhook_for_user(self, user: User, webhook_id: str) -> None:
        await self.ensure_module_enabled("webhooks")
        webhook = await self.repo.get_webhook_for_user(user.id, webhook_id)
        if not webhook:
            raise HTTPException(status_code=404, detail="Webhook endpoint not found")
        await self.repo.delete_webhook(webhook)
        await self.db.commit()
        logger.info("Webhook deleted user=%s webhook=%s", user.id, webhook_id)

    async def test_webhook_for_user(self, user: User, webhook_id: str) -> dict:
        await self.ensure_module_enabled("webhooks")
        webhook = await self.repo.get_webhook_for_user(user.id, webhook_id)
        if not webhook:
            raise HTTPException(status_code=404, detail="Webhook endpoint not found")

        metadata = await self.get_platform_metadata()
        payload = {
            "event": "platform.test",
            "sent_at": datetime.now(UTC).isoformat(),
            "app_name": metadata.app_name,
            "core_domain_plural": metadata.core_domain_plural,
            "target_user_id": user.id,
        }
        raw_body = json.dumps(payload).encode("utf-8")
        signature = hmac.new(webhook.secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    webhook.target_url,
                    content=raw_body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Generic-App-Event": payload["event"],
                        "X-Generic-App-Signature": signature,
                    },
                )
            webhook.last_tested_at = datetime.now(UTC)
            webhook.last_response_status = response.status_code
            await self.db.commit()
            await self.db.refresh(webhook)
            if not response.is_success:
                logger.warning(
                    "Webhook test delivery failed user=%s webhook=%s status=%s",
                    user.id,
                    webhook.id,
                    response.status_code,
                )
            return {
                "delivered": response.is_success,
                "status_code": response.status_code,
                "response_preview": response.text[:500] if response.text else None,
                "error": None,
            }
        except httpx.HTTPError as exc:
            webhook.last_tested_at = datetime.now(UTC)
            webhook.last_response_status = None
            await self.db.commit()
            await self.db.refresh(webhook)
            logger.warning(
                "Webhook test request failed user=%s webhook=%s error_type=%s",
                user.id,
                webhook.id,
                type(exc).__name__,
            )
            return {
                "delivered": False,
                "status_code": None,
                "response_preview": None,
                "error": str(exc),
            }

    @staticmethod
    def _validate_webhook_target(target_url: str) -> None:
        parsed = urlparse(target_url)
        host = (parsed.hostname or "").strip().lower()
        if not host:
            raise HTTPException(status_code=422, detail="Webhook target host is required")
        if host in {"localhost", "metadata.google.internal"} or host.endswith(".internal"):
            raise HTTPException(status_code=422, detail="Webhook target host is not allowed")
        if "." not in host and not host.startswith("["):
            raise HTTPException(status_code=422, detail="Webhook target host is not allowed")
        try:
            ip = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            return
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise HTTPException(status_code=422, detail="Webhook target host is not allowed")
