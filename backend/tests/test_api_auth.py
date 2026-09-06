import unittest

from backend.tests.integration_support import (
    api_client,
    auth_request,
    integration_enabled,
    register_and_sign_in,
    register_user,
    sign_in,
    unique_email,
)


@unittest.skipUnless(integration_enabled(), "Set RUN_INTEGRATION_TESTS=1 with postgres/redis running")
class AuthApiIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_sign_up_sign_in_me_and_logout(self):
        async with api_client() as client:
            email = await register_and_sign_in(client)

            me = await client.get("/api/v1/auth/me")
            self.assertEqual(me.status_code, 200)
            self.assertEqual(me.json()["email"], email)

            logout = await auth_request(client, "POST", "/api/v1/auth/logout")
            self.assertEqual(logout.status_code, 204)

            me_after_logout = await client.get("/api/v1/auth/me")
            self.assertEqual(me_after_logout.status_code, 401)

    async def test_me_requires_authentication(self):
        async with api_client() as client:
            response = await client.get("/api/v1/auth/me")
            self.assertEqual(response.status_code, 401)

    async def test_csrf_blocks_mutating_request_without_header(self):
        async with api_client() as client:
            email = await register_and_sign_in(client)

            response = await client.post(
                "/api/v1/auth/logout",
                headers={},
            )
            self.assertEqual(response.status_code, 403)
            self.assertIn("CSRF", response.json()["detail"])

            me = await client.get("/api/v1/auth/me")
            self.assertEqual(me.status_code, 200)
            self.assertEqual(me.json()["email"], email)

    async def test_refresh_rotates_session(self):
        async with api_client() as client:
            email = await register_user(client)
            await sign_in(client, email=email)

            refresh = await auth_request(client, "POST", "/api/v1/auth/refresh")
            self.assertEqual(refresh.status_code, 200)
            self.assertEqual(refresh.json()["user"]["email"], email)

            me = await client.get("/api/v1/auth/me")
            self.assertEqual(me.status_code, 200)

    async def test_invalid_credentials_return_unauthorized(self):
        async with api_client() as client:
            email = await register_user(client)
            response = await client.post(
                "/api/v1/auth/sign-in",
                json={"email": email, "password": "WrongPassword123!"},
            )
            self.assertEqual(response.status_code, 401)


class AuthApiUnitTest(unittest.TestCase):
    def test_unique_email_generates_distinct_values(self):
        first = unique_email("auth")
        second = unique_email("auth")
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith("@example.com"))
