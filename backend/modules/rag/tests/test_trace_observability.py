from __future__ import annotations

import unittest
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from backend.modules.rag.application import trace_context
from backend.modules.rag.application.trace_context import RagTraceContext


class TraceObservabilityTests(unittest.TestCase):
    def test_trace_context_uses_request_id_and_sanitized_stage_data(self):
        trace = RagTraceContext()
        started = trace.measure("dense_branch")
        trace.complete("dense_branch", started, status="failed")

        self.assertTrue(trace.request_id)
        self.assertEqual(trace.branch_status, {"dense_branch": "failed"})
        self.assertIn("dense_branch", trace.stage_timings_ms)
        self.assertNotIn("content", trace.stage_timings_ms)


class CorrelatedTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_resume_and_child_stages_share_trace_without_text(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")
        with patch.object(trace_context, "tracer", tracer):
            with tracer.start_as_current_span("http"):
                carrier = trace_context.trace_carrier()

            @trace_context.traced("rag.generation")
            async def generate():
                diagnostics = RagTraceContext()
                started = diagnostics.measure("citation_validation")
                diagnostics.complete("citation_validation", started)

            with trace_context.resume_trace(carrier):
                await generate()
        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 4)
        self.assertEqual(len({span.context.trace_id for span in spans}), 1)
        self.assertTrue(all("query" not in span.attributes for span in spans))
        provider.shutdown()
