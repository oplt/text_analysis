from backend.modules.text_research.application.engine_comparison import compare_engine_results


def test_frequency_comparison_references_shared_terms_and_differences() -> None:
    comparison = compare_engine_results(
        "frequencies",
        {"frequencies": [{"term": "alpha", "count": 2}, {"term": "beta", "count": 1}]},
        {
            "analysis_result": {
                "results": {
                    "frequencies": [{"term": "alpha", "count": 2}, {"term": "beta", "count": 3}]
                }
            }
        },
    )
    assert comparison["term_overlap"] == 2
    assert comparison["count_equal"] is False
    assert comparison["count_differences"]["beta"] == {"python": 1, "r": 3}


def test_dfm_comparison_uses_persisted_python_and_canonical_r_shapes() -> None:
    comparison = compare_engine_results(
        "dfm",
        {"dfm": {"dimensions": {"units": 2, "features": 2}, "feature_names": ["a", "b"], "nnz": 3}},
        {
            "analysis_result": {
                "results": {
                    "dimensions": {"documents": 2, "features": 2},
                    "feature_names": ["a", "b"],
                    "summary": {"nnz": 3},
                }
            }
        },
    )
    assert comparison["vocabulary_equal"] is True
    assert comparison["vocabulary_overlap"] == 2
