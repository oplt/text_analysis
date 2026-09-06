from datetime import UTC, datetime
from types import SimpleNamespace
from unittest import TestCase

from backend.modules.platform.api_key_routes import _api_key_to_response
from backend.modules.platform.billing_routes import _plan_to_response, _subscription_to_response
from backend.modules.platform.email_template_routes import _email_template_to_response
from backend.modules.platform.feature_flag_routes import _feature_flag_to_response
from backend.modules.platform.webhook_routes import _webhook_to_response

NOW = datetime(2026, 7, 6, tzinfo=UTC)


class PlatformResponseMapperTest(TestCase):
    def test_billing_maps_feature_alias_and_nested_plan(self):
        plan = SimpleNamespace(
            id="plan-1",
            code="growth",
            name="Growth",
            description=None,
            price_cents=4900,
            interval="month",
            is_active=True,
            is_default=False,
            features_json=["api_keys"],
            created_at=NOW,
            updated_at=NOW,
        )
        subscription = SimpleNamespace(
            id="sub-1",
            status="active",
            cancel_at_period_end=False,
            started_at=NOW,
            current_period_end=None,
            created_at=NOW,
            updated_at=NOW,
        )

        plan_response = _plan_to_response(plan)
        subscription_response = _subscription_to_response(subscription, plan)

        self.assertEqual(plan_response.features, ["api_keys"])
        self.assertIsNone(plan_response.description)
        self.assertEqual(subscription_response.plan, plan_response)
        self.assertIsNone(subscription_response.current_period_end)

    def test_api_key_does_not_expose_hash(self):
        api_key = SimpleNamespace(
            id="key-1",
            name="Automation",
            key_prefix="gap_example",
            key_hash="must-not-leak",
            last_used_at=None,
            revoked_at=None,
            created_at=NOW,
        )

        payload = _api_key_to_response(api_key).model_dump()

        self.assertNotIn("key_hash", payload)
        self.assertEqual(payload["key_prefix"], "gap_example")
        self.assertIsNone(payload["revoked_at"])

    def test_webhook_maps_events_without_exposing_secret(self):
        webhook = SimpleNamespace(
            id="hook-1",
            target_url="https://example.test/hook",
            description=None,
            secret="must-not-leak",
            is_active=True,
            events_json=["project.created"],
            last_tested_at=None,
            last_response_status=None,
            created_at=NOW,
            updated_at=NOW,
        )

        payload = _webhook_to_response(webhook).model_dump(mode="json")

        self.assertNotIn("secret", payload)
        self.assertEqual(payload["events"], ["project.created"])
        self.assertIsNone(payload["last_response_status"])

    def test_flag_and_template_preserve_optional_fields(self):
        flag = SimpleNamespace(
            id="flag-1",
            key="beta",
            name="Beta",
            description=None,
            module_key=None,
            is_enabled=True,
            rollout_percentage=25,
            updated_at=NOW,
        )
        template = SimpleNamespace(
            id="template-1",
            key="auth.verify",
            name="Verify",
            subject_template="Verify",
            html_template="<p>Verify</p>",
            text_template=None,
            is_active=True,
            updated_at=NOW,
        )

        flag_payload = _feature_flag_to_response(flag).model_dump()
        template_payload = _email_template_to_response(template).model_dump()

        self.assertIsNone(flag_payload["module_key"])
        self.assertEqual(flag_payload["rollout_percentage"], 25)
        self.assertIsNone(template_payload["text_template"])
