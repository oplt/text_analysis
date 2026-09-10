from backend.modules.text_research.application.engine_comparison import compare_engine_results


def test_frequency_comparison_requires_full_term_set_equality() -> None:
    unequal_terms = compare_engine_results(
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
    assert unequal_terms["term_sets_equal"] is True
    assert unequal_terms["count_equal"] is False
    assert unequal_terms["count_differences"]["beta"] == {"python": 1, "r": 3}

    missing_term = compare_engine_results(
        "frequencies",
        {"frequencies": [{"term": "alpha", "count": 2}, {"term": "beta", "count": 1}]},
        {"frequencies": [{"term": "alpha", "count": 2}]},
    )
    assert missing_term["term_sets_equal"] is False
    assert missing_term["count_equal"] is False


def test_dfm_comparison_uses_sparse_cells() -> None:
    comparison = compare_engine_results(
        "dfm",
        {
            "dfm": {
                "dimensions": {"units": 2, "features": 2},
                "feature_names": ["a", "b"],
                "unit_ids": ["u1", "u2"],
                "nnz": 3,
                "sparse": {"row": [0, 0, 1], "col": [0, 1, 1], "data": [1, 2, 3]},
            }
        },
        {
            "analysis_result": {
                "results": {
                    "dimensions": {"documents": 2, "features": 2},
                    "feature_names": ["a", "b"],
                    "unit_ids": ["u1", "u2"],
                    "summary": {"nnz": 3},
                    "nnz": 3,
                    "sparse_coo": {
                        "rows": [0, 0, 1],
                        "cols": [0, 1, 1],
                        "values": [1, 2, 3],
                    },
                }
            }
        },
    )
    assert comparison["vocabulary_equal"] is True
    assert comparison["cells_equal"] is True
    assert comparison["cell_difference_count"] == 0

    mismatched = compare_engine_results(
        "dfm",
        {
            "dfm": {
                "feature_names": ["a", "b"],
                "unit_ids": ["u1"],
                "sparse": {"row": [0], "col": [0], "data": [1]},
            }
        },
        {
            "feature_names": ["a", "b"],
            "unit_ids": ["u1"],
            "sparse_coo": {"rows": [0], "cols": [0], "values": [9]},
        },
    )
    assert mismatched["cells_equal"] is False
    assert mismatched["cell_difference_count"] == 1


def test_kwic_comparison_uses_occurrence_multiset() -> None:
    equal = compare_engine_results(
        "kwic",
        {
            "matches": [
                {"text_unit_id": "u1", "token_start": 0, "token_end": 1, "keyword": "free"},
                {"text_unit_id": "u1", "token_start": 3, "token_end": 4, "keyword": "free"},
            ]
        },
        {
            "analysis_result": {
                "results": {
                    "matches": [
                        {
                            "text_unit_id": "u1",
                            "token_start": 0,
                            "token_end": 1,
                            "keyword": "free",
                        },
                        {
                            "text_unit_id": "u1",
                            "token_start": 3,
                            "token_end": 4,
                            "keyword": "free",
                        },
                    ]
                }
            }
        },
    )
    assert equal["matches_equal"] is True
    assert equal["match_overlap"] == 2

    collapsed_would_pass_old = compare_engine_results(
        "kwic",
        {
            "matches": [
                {"text_unit_id": "u1", "token_start": 0, "token_end": 1, "keyword": "free"},
                {"text_unit_id": "u1", "token_start": 3, "token_end": 4, "keyword": "free"},
            ]
        },
        {
            "matches": [
                {"text_unit_id": "u1", "token_start": 0, "token_end": 1, "keyword": "free"},
            ]
        },
    )
    assert collapsed_would_pass_old["matches_equal"] is False
    assert collapsed_would_pass_old["python_match_count"] == 2
    assert collapsed_would_pass_old["r_match_count"] == 1
