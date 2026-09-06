import unittest
from unittest.mock import patch

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
class RagPermissionsIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_cross_user_document_access_is_blocked(self):
        async with api_client() as owner_client:
            await register_and_sign_in(owner_client)

            with patch("backend.modules.rag.api.routes.queue_document_indexing"):
                upload = await auth_request(
                    owner_client,
                    "POST",
                    "/api/v1/rag/documents/upload",
                    files={"file": ("notes.txt", b"owner-only content", "text/plain")},
                )
            self.assertEqual(upload.status_code, 201, upload.text)
            document_id = upload.json()["document"]["id"]

            get_as_owner = await owner_client.get(f"/api/v1/rag/documents/{document_id}")
            self.assertEqual(get_as_owner.status_code, 200)

        async with api_client() as other_client:
            await register_and_sign_in(other_client)

            get_as_other = await other_client.get(f"/api/v1/rag/documents/{document_id}")
            self.assertEqual(get_as_other.status_code, 404)

            delete_as_other = await auth_request(
                other_client,
                "DELETE",
                f"/api/v1/rag/documents/{document_id}",
            )
            self.assertEqual(delete_as_other.status_code, 403)

    async def test_upload_to_foreign_project_is_forbidden(self):
        async with api_client() as client:
            await register_and_sign_in(client)

            with patch("backend.modules.rag.api.routes.queue_document_indexing"):
                upload = await auth_request(
                    client,
                    "POST",
                    "/api/v1/rag/documents/upload",
                    data={"project_id": "00000000-0000-0000-0000-000000000099"},
                    files={"file": ("notes.txt", b"project scoped", "text/plain")},
                )
            self.assertEqual(upload.status_code, 403)
            self.assertIn("Project access denied", upload.json()["detail"])
