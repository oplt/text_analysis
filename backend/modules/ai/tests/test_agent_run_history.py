"""Persisted, user-scoped Agent run history contracts."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.modules.ai.agent_router import router
from backend.modules.ai.application.agent_service import AgentService
from fastapi import HTTPException


class AgentRunHistoryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_runs_is_scoped_to_the_authenticated_user(self) -> None:
        service = AgentService(MagicMock())
        service.ai.repo.list_agent_runs_for_user = AsyncMock(return_value=([], 0))
        user = SimpleNamespace(id="user-1")

        result = await service.list_runs(user, limit=20, offset=0)

        self.assertEqual(result, ([], 0))
        service.ai.repo.list_agent_runs_for_user.assert_awaited_once_with(
            "user-1", limit=20, offset=0
        )

    async def test_get_run_returns_not_found_for_another_users_run(self) -> None:
        service = AgentService(MagicMock())
        service.ai.repo.get_agent_run_for_user = AsyncMock(return_value=None)
        user = SimpleNamespace(id="user-1")

        with self.assertRaises(HTTPException) as context:
            await service.get_run(user, "run-owned-by-someone-else")

        self.assertEqual(context.exception.status_code, 404)
        service.ai.repo.get_agent_run_for_user.assert_awaited_once_with(
            "user-1", "run-owned-by-someone-else"
        )


class AgentRunHistoryRouteTests(unittest.TestCase):
    def test_history_and_detail_routes_are_registered(self) -> None:
        registered = {
            (method, route.path) for route in router.routes for method in route.methods or set()
        }

        self.assertEqual(
            registered,
            {
                ("GET", "/runs"),
                ("GET", "/runs/{run_id}"),
                ("POST", "/runs"),
            },
        )
