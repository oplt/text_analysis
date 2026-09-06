import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI


class RuntimeObservabilityTest(unittest.TestCase):
    def test_setup_observability_disabled_does_not_crash(self):
        from backend.observability import otel

        with patch("backend.observability.config.env_bool", return_value=False):
            otel._configured = False
            otel.setup_observability(app=None, service_name="fastapi-backend")

    def test_setup_observability_idempotent_fastapi(self):
        from backend.observability import otel

        app = FastAPI()
        calls = []

        class FakeFastAPIInstrumentor:
            @classmethod
            def instrument_app(cls, target_app):
                calls.append(target_app)

        otel._configured = False
        with (
            patch("backend.observability.otel._setup_trace_exporter", return_value=None),
            patch("backend.observability.otel._setup_library_instrumentation", return_value=None),
            patch(
                "opentelemetry.instrumentation.fastapi.FastAPIInstrumentor",
                FakeFastAPIInstrumentor,
            ),
        ):
            otel.setup_traces(app=app, service_name="fastapi-backend")
            otel.setup_traces(app=app, service_name="fastapi-backend")

        self.assertEqual(calls, [app])
        self.assertTrue(app.state.otel_instrumented)

    def test_metrics_endpoint_exposes_prometheus_text(self):
        if find_spec("prometheus_client") is None:
            self.skipTest("prometheus_client is not installed")
        from backend.observability.metrics import setup_metrics

        app = FastAPI()
        setup_metrics(app)

        route = next(route for route in app.routes if getattr(route, "path", "") == "/metrics")
        response = route.endpoint()

        self.assertIn("text/plain", response.media_type)
        self.assertIn(b"python_info", response.body)

    def test_signal_endpoint_accepts_base_or_full_endpoint(self):
        from backend.observability.otel import _signal_endpoint

        self.assertEqual(
            _signal_endpoint("metrics", "http://127.0.0.1:4318"),
            "http://127.0.0.1:4318/v1/metrics",
        )
        self.assertEqual(
            _signal_endpoint("logs", "http://127.0.0.1:4318/v1/traces"),
            "http://127.0.0.1:4318/v1/logs",
        )

    def test_observed_span_records_exception_and_reraises(self):
        from opentelemetry import trace

        from backend.observability import instruments

        captured = SimpleNamespace(recorded=None, status=None)

        class FakeSpan:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def record_exception(self, exc):
                captured.recorded = exc

            def set_status(self, status):
                captured.status = status

            def set_attribute(self, key, value):
                return None

        class FakeTracer:
            def start_as_current_span(self, name, attributes=None):
                return FakeSpan()

        with (
            patch.object(trace, "get_tracer", return_value=FakeTracer()),
            self.assertRaises(ValueError) as exc_info,
            instruments.observed_span("test.span", request_id="req-1"),
        ):
            raise ValueError("boom")

        self.assertIs(captured.recorded, exc_info.exception)
        self.assertIsNotNone(captured.status)


if __name__ == "__main__":
    unittest.main()
