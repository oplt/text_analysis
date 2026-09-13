"""LATEST-011: per-operation async capability contract."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.modules.text_research.api.quantitative_routes import (
    _enforce_async_contract,
    _reject_unsupported_async,
    corpus_stats,
    frequencies,
    kwic,
)
from backend.modules.text_research.api.schemas_quantitative import (
    AsyncCapableAnalysisRequest,
    FrequencyRequest,
    InlineAnalysisRequest,
    KwicRequest,
    ReadabilityRequest,
)
from backend.modules.text_research.application.run_adapters import (
    ANALYSIS_OPERATION_RUN_TYPES,
    OPERATION_EXECUTION,
    describe_operation_execution_capabilities,
    operation_supports_async,
)


class OperationAsyncRegistryTests(unittest.TestCase):
    def test_registry_covers_all_public_operations(self) -> None:
        self.assertEqual(set(OPERATION_EXECUTION), set(ANALYSIS_OPERATION_RUN_TYPES))

    def test_inline_ops_are_explicit(self) -> None:
        for operation in ("corpus_stats", "kwic", "dictionary", "readability"):
            self.assertFalse(operation_supports_async(operation), msg=operation)

    def test_heavy_ops_support_async(self) -> None:
        for operation in ("frequencies", "dfm", "cooccurrence", "clustering"):
            self.assertTrue(operation_supports_async(operation), msg=operation)

    def test_capabilities_payload_includes_default_mode(self) -> None:
        payload = describe_operation_execution_capabilities()
        self.assertFalse(payload["readability"]["async"])
        self.assertEqual(payload["readability"]["default_execution_mode"], "inline")
        self.assertTrue(payload["frequencies"]["async"])


class AsyncContractEnforcementTests(unittest.TestCase):
    def test_reject_unsupported_async_uses_registry(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            _reject_unsupported_async("kwic", True)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("does not support", str(ctx.exception.detail))

    def test_supported_async_is_allowed(self) -> None:
        _reject_unsupported_async("frequencies", True)  # does not raise

    def test_schema_split_inline_vs_async_capable(self) -> None:
        self.assertTrue(issubclass(FrequencyRequest, AsyncCapableAnalysisRequest))
        self.assertTrue(issubclass(KwicRequest, InlineAnalysisRequest))
        self.assertTrue(issubclass(ReadabilityRequest, InlineAnalysisRequest))
        self.assertFalse(issubclass(FrequencyRequest, InlineAnalysisRequest))


class RouteAsyncContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_kwic_run_async_true_returns_422(self) -> None:
        body = KwicRequest(unit_type="document", keyword="climate", run_async=True)
        with self.assertRaises(HTTPException) as ctx:
            await kwic(
                "corpus-1",
                body,
                db=MagicMock(),
                current_user=SimpleNamespace(id="u1"),
            )
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_corpus_stats_run_async_true_returns_422(self) -> None:
        body = InlineAnalysisRequest(unit_type="document", run_async=True)
        with self.assertRaises(HTTPException) as ctx:
            await corpus_stats(
                "corpus-1",
                body,
                db=MagicMock(),
                current_user=SimpleNamespace(id="u1"),
            )
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_frequencies_run_async_true_is_forwarded(self) -> None:
        body = FrequencyRequest(unit_type="document", run_async=True)
        mock_run = SimpleNamespace(id="run-1")
        with (
            patch(
                "backend.modules.text_research.api.quantitative_routes.QuantitativeAnalysisService"
            ) as svc,
            patch(
                "backend.modules.text_research.api.quantitative_routes._respond",
                return_value={"id": "run-1"},
            ),
        ):
            svc.return_value.frequencies = AsyncMock(return_value=mock_run)
            await frequencies(
                "corpus-1",
                body,
                db=MagicMock(),
                current_user=SimpleNamespace(id="u1"),
            )
        kwargs = svc.return_value.frequencies.await_args.kwargs
        self.assertTrue(kwargs["run_async"])

    def test_enforce_contract_helper(self) -> None:
        _enforce_async_contract("dfm", SimpleNamespace(run_async=False))
        with self.assertRaises(HTTPException):
            _enforce_async_contract("dictionary", SimpleNamespace(run_async=True))


if __name__ == "__main__":
    unittest.main()
