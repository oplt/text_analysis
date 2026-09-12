from __future__ import annotations

import unittest

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
