"""E2E rate-limit relaxation must be explicit and unavailable in production."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend.core.config import Settings

REQUIRED_SETTINGS = {
    "DATABASE_URL": "postgresql+asyncpg://app:app@localhost:5432/app_db",
    "REDIS_URL": "redis://localhost:6379/0",
    "JWT_SECRET": "01234567890123456789012345678901",
    "JWT_ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": "15",
    "REFRESH_TOKEN_EXPIRE_DAYS": "7",
}


class E2ERateLimitRelaxTest(unittest.TestCase):
    def test_relax_raises_floors_outside_production(self) -> None:
        env = {
            **REQUIRED_SETTINGS,
            "APP_ENV": "ci",
            "E2E_RELAX_RATE_LIMITS": "true",
            "PUBLIC_RATE_LIMIT_REQUESTS": "120",
            "AUTH_SIGNIN_LIMIT": "10",
            "AUTH_SIGNUP_LIMIT": "5",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings()
        self.assertTrue(settings.e2e_rate_limits_relaxed)
        self.assertEqual(settings.effective_public_rate_limit_requests, 5_000)
        self.assertEqual(settings.effective_auth_signin_limit, 100)
        self.assertEqual(settings.effective_auth_signup_limit, 50)

    def test_relax_ignored_in_production(self) -> None:
        env = {
            **REQUIRED_SETTINGS,
            "APP_ENV": "production",
            "E2E_RELAX_RATE_LIMITS": "true",
            "PUBLIC_RATE_LIMIT_REQUESTS": "120",
            "AUTH_SIGNIN_LIMIT": "10",
            "AUTH_SIGNUP_LIMIT": "5",
            "COOKIE_SECURE": "true",
            "FRONTEND_URL": "https://example.com",
            "CORS_ALLOWED_ORIGINS": "https://example.com",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings()
        self.assertFalse(settings.e2e_rate_limits_relaxed)
        self.assertEqual(settings.effective_public_rate_limit_requests, 120)
        self.assertEqual(settings.effective_auth_signin_limit, 10)
        self.assertEqual(settings.effective_auth_signup_limit, 5)

    def test_relax_does_not_disable_when_configured_positive(self) -> None:
        env = {
            **REQUIRED_SETTINGS,
            "APP_ENV": "dev",
            "E2E_RELAX_RATE_LIMITS": "true",
            "PUBLIC_RATE_LIMIT_REQUESTS": "200",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings()
        self.assertGreater(settings.effective_public_rate_limit_requests, 0)

    def test_zero_public_limit_still_means_disabled(self) -> None:
        env = {
            **REQUIRED_SETTINGS,
            "APP_ENV": "dev",
            "E2E_RELAX_RATE_LIMITS": "true",
            "PUBLIC_RATE_LIMIT_REQUESTS": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings()
        self.assertEqual(settings.effective_public_rate_limit_requests, 0)

    def test_middleware_uses_effective_public_limit(self) -> None:
        from backend.api.middleware.public_rate_limit import PublicRateLimitMiddleware

        with patch("backend.api.middleware.public_rate_limit.settings") as mock_settings:
            mock_settings.effective_public_rate_limit_requests = 0
            mock_settings.PUBLIC_RATE_LIMIT_WINDOW_SECONDS = 60
            middleware = PublicRateLimitMiddleware(app=None)

            async def call_next(_request):
                return "ok"

            class _Request:
                url = type("U", (), {"path": "/api/v1/health"})()
                client = type("C", (), {"host": "127.0.0.1"})()

            import asyncio

            result = asyncio.run(middleware.dispatch(_Request(), call_next))
            self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()
