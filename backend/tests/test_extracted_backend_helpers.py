from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from starlette.requests import Request

from backend.modules.audit.request_logging import log_request_audit_event
from backend.modules.profile.models import UserProfile
from backend.modules.profile.serializers import profile_to_response


class RequestAuditLoggingTest(IsolatedAsyncioTestCase):
    async def test_forwards_request_context_and_event_fields(self) -> None:
        db = object()
        request = Request(
            {
                "type": "http",
                "method": "PATCH",
                "path": "/resource",
                "headers": [(b"user-agent", b"test-agent")],
                "client": ("203.0.113.7", 1234),
                "server": ("testserver", 80),
                "scheme": "http",
                "query_string": b"",
            }
        )
        log = AsyncMock()

        with patch("backend.modules.audit.request_logging.AuditRepository") as repository_type:
            repository_type.return_value.log = log
            await log_request_audit_event(
                db,  # type: ignore[arg-type]
                request,
                action="resource.updated",
                actor_user_id="user-1",
                resource_type="resource",
                resource_id="resource-1",
                metadata={"field": "value"},
            )

        repository_type.assert_called_once_with(db)
        log.assert_awaited_once_with(
            action="resource.updated",
            user_id="user-1",
            resource_type="resource",
            resource_id="resource-1",
            ip_address="203.0.113.7",
            user_agent="test-agent",
            metadata={"field": "value"},
        )


class ProfileSerializerTest(TestCase):
    def test_maps_profile_fields_without_changing_contract(self) -> None:
        profile = UserProfile(
            user_id="user-1",
            bio="Bio",
            avatar_url="https://example.test/avatar.png",
            avatar_storage_key="avatars/user-1/avatar.png",
            location="Brussels",
            website="https://example.test",
        )

        response = profile_to_response(profile)

        self.assertEqual(
            response.model_dump(),
            {
                "user_id": "user-1",
                "bio": "Bio",
                "avatar_url": "https://example.test/avatar.png",
                "location": "Brussels",
                "website": "https://example.test",
            },
        )
