import unittest

from backend.modules.ai.router import router as ai_router


class AiRouteRegistrationTest(unittest.TestCase):
    def test_all_capability_routes_are_registered_once(self):
        expected = {
            ("GET", "/overview"),
            ("GET", "/providers"),
            ("GET", "/prompts"),
            ("POST", "/prompts"),
            ("PATCH", "/prompts/{template_id}"),
            ("GET", "/prompts/{template_id}/versions"),
            ("POST", "/prompts/{template_id}/versions"),
            ("PATCH", "/prompts/{template_id}/versions/{version_id}"),
            ("GET", "/documents"),
            ("GET", "/documents/{document_id}"),
            ("POST", "/documents"),
            ("POST", "/documents/upload"),
            ("POST", "/retrieve"),
            ("GET", "/runs"),
            ("POST", "/runs"),
            ("GET", "/reviews"),
            ("POST", "/runs/{run_id}/reviews"),
            ("POST", "/reviews/{review_id}/decision"),
            ("GET", "/runs/{run_id}/feedback"),
            ("POST", "/runs/{run_id}/feedback"),
            ("GET", "/evaluation-datasets"),
            ("POST", "/evaluation-datasets"),
            ("PATCH", "/evaluation-datasets/{dataset_id}"),
            ("GET", "/evaluation-datasets/{dataset_id}/cases"),
            ("POST", "/evaluation-datasets/{dataset_id}/cases"),
            ("GET", "/evaluation-runs"),
            ("POST", "/evaluation-datasets/{dataset_id}/run"),
        }
        registered = [
            (method, route.path) for route in ai_router.routes for method in route.methods or set()
        ]

        self.assertEqual(set(registered), expected)
        self.assertEqual(len(registered), len(expected))
