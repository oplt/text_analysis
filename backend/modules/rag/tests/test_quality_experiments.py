import pytest

from backend.modules.rag.eval.quality_experiments import (
    RetrievalExperiment,
    experiment_matrix,
    experiment_report,
)


def test_matrix_contains_target_values_without_changing_defaults():
    matrix = experiment_matrix()
    assert len(matrix) == 4_374
    assert {item.rrf_k for item in matrix} == {30, 60, 90}
    assert {item.parent_expansion for item in matrix} == {False, True}


def test_report_requires_complete_quality_and_latency_metrics():
    with pytest.raises(ValueError, match="incomplete"):
        experiment_report(
            RetrievalExperiment(256, 0, 20, 20, 30, 0, False, 2),
            {},
            environment={},
        )
