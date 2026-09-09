import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.lib.generation_port import AiServiceGenerationPort, RagGenerationResult
from backend.lib.project_access import SqlAlchemyProjectAccessPort


class GenerationPortTest(unittest.IsolatedAsyncioTestCase):
    async def test_ai_service_generation_port_maps_run_fields(self):
        db = AsyncMock()
        port = AiServiceGenerationPort(db)
        user = SimpleNamespace(id="user-1")
        fake_run = SimpleNamespace(id="run-1", output_text="answer", model_name="gpt-test")

        with patch.object(port, "_get_service") as get_service:
            service = AsyncMock()
            service.run_rag_answer = AsyncMock(return_value=fake_run)
            get_service.return_value = service

            result = await port.run_rag_answer(
                user,
                query="hello",
                combined_context="context",
                retrieved_chunk_ids=["chunk-1"],
            )

        self.assertEqual(
            result,
            RagGenerationResult(id="run-1", output_text="answer", model_name="gpt-test"),
        )


class ProjectAccessPortTest(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_project_access_allows_project_owner(self):
        db = AsyncMock()
        port = SqlAlchemyProjectAccessPort(db)
        project = SimpleNamespace(id="project-1", owner_id="user-1")
        port._repo.get_by_id_for_user = AsyncMock(return_value=project)

        await port.ensure_project_access("user-1", "project-1")

        port._repo.get_by_id_for_user.assert_awaited_once_with("project-1", "user-1")

    async def test_ensure_project_access_allows_project_member(self):
        db = AsyncMock()
        port = SqlAlchemyProjectAccessPort(db)
        project = SimpleNamespace(id="project-1", owner_id="owner-1")
        port._repo.get_by_id_for_user = AsyncMock(return_value=project)

        await port.ensure_project_access("member-1", "project-1")

        port._repo.get_by_id_for_user.assert_awaited_once_with("project-1", "member-1")

    async def test_ensure_project_access_denies_non_member(self):
        db = AsyncMock()
        port = SqlAlchemyProjectAccessPort(db)
        port._repo.get_by_id_for_user = AsyncMock(return_value=None)

        with self.assertRaises(Exception) as ctx:
            await port.ensure_project_access("user-1", "project-1")

        self.assertEqual(ctx.exception.status_code, 403)

    async def test_ensure_project_access_denies_missing_project(self):
        db = AsyncMock()
        port = SqlAlchemyProjectAccessPort(db)
        port._repo.get_by_id_for_user = AsyncMock(return_value=None)

        with self.assertRaises(Exception) as ctx:
            await port.ensure_project_access("user-1", "missing-project")

        self.assertEqual(ctx.exception.status_code, 403)
