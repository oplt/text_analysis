import unittest
import uuid

from backend.tests.integration_support import (
    api_client,
    auth_request,
    integration_enabled,
    rag_integration_ready,
    register_and_sign_in,
)


@unittest.skipUnless(
    integration_enabled() and rag_integration_ready(),
    "Set RUN_INTEGRATION_TESTS=1 and migrate postgres (alembic upgrade head)",
)
class AgentRunSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_agent_run_returns_completed_response(self):
        async with api_client() as client:
            await register_and_sign_in(client)

            template_key = f"smoke_agent_{uuid.uuid4().hex[:8]}"

            template = await auth_request(
                client,
                "POST",
                "/api/v1/ai/prompts",
                json={
                    "key": template_key,
                    "name": "Smoke Agent Prompt",
                    "description": "Integration smoke prompt",
                },
            )
            self.assertEqual(template.status_code, 201, template.text)
            template_id = template.json()["id"]

            version = await auth_request(
                client,
                "POST",
                f"/api/v1/ai/prompts/{template_id}/versions",
                json={
                    "provider_key": "local",
                    "model_name": "local-heuristic",
                    "system_prompt": "You are a concise assistant.",
                    "user_prompt_template": "Answer briefly.",
                    "is_published": True,
                },
            )
            self.assertEqual(version.status_code, 201, version.text)

            run = await auth_request(
                client,
                "POST",
                "/api/v1/agent/runs",
                json={
                    "prompt_template_key": template_key,
                    "variables": {},
                    "user_message": "What is 2+2?",
                },
            )
            self.assertEqual(run.status_code, 201, run.text)
            body = run.json()
            self.assertEqual(body["status"], "completed")
            self.assertTrue(body["id"])
            self.assertTrue(body["memory_run_id"])
            self.assertIsNotNone(body.get("output_text"))
