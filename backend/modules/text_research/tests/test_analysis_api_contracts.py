"""TASK-023: executable API contract coverage for analysis orphan routes."""

from __future__ import annotations

from backend.modules.text_research.api.schemas import (
    MeasurementComparisonRequest,
    StatisticalModelRequest,
)
from backend.modules.text_research.api.quantitative_routes import router as quantitative_router
from backend.modules.text_research.api.routes import router as research_router


def _paths(router) -> set[str]:
    out: set[str] = set()
    for route in router.routes:
        path = getattr(route, "path", None)
        if path:
            out.add(path)
    return out


class TestAnalysisRouteContracts:
    def test_quantitative_router_registers_core_paths(self) -> None:
        paths = _paths(quantitative_router)
        assert "/corpora/{corpus_id}/analysis/frequencies" in paths
        assert "/corpora/{corpus_id}/analysis/readability" in paths
        assert "/corpora/{corpus_id}/analysis/kwic" in paths

    def test_main_router_includes_quantitative_and_orphan_paths(self) -> None:
        paths = _paths(research_router)
        assert "/corpora/{corpus_id}/analysis/frequencies" in paths
        assert "/corpora/{corpus_id}/analysis/statistical-model" in paths
        assert "/corpora/{corpus_id}/analysis/measurement-comparison" in paths

    def test_statistical_model_schema_requires_core_fields(self) -> None:
        body = StatisticalModelRequest(
            model="ols",
            dependent_var="y",
            independent_vars=["x1"],
            rows=[{"y": 1.0, "x1": 2.0}],
        )
        assert body.model == "ols"
        assert body.dependent_var == "y"

    def test_measurement_comparison_schema_aligns_arrays(self) -> None:
        body = MeasurementComparisonRequest(
            source_a="human",
            source_b="model",
            values_a=["a", "b"],
            values_b=["a", "c"],
            ids=["1", "2"],
            value_kind="categorical",
        )
        assert len(body.values_a) == len(body.values_b) == len(body.ids or [])
