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
    assert comparison["comparison_complete"] is True
    assert comparison["cells_equal"] is True
    assert comparison["matrix_checksum_equal"] is True
    assert comparison["cell_difference_count"] == 0

    mismatched = compare_engine_results(
        "dfm",
        {
            "dfm": {
                "feature_names": ["a", "b"],
                "unit_ids": ["u1"],
                "nnz": 1,
                "sparse": {"row": [0], "col": [0], "data": [1]},
            }
        },
        {
            "feature_names": ["a", "b"],
            "unit_ids": ["u1"],
            "nnz": 1,
            "sparse_coo": {"rows": [0], "cols": [0], "values": [9]},
        },
    )
    assert mismatched["comparison_complete"] is True
    assert mismatched["cells_equal"] is False
    assert mismatched["cell_difference_count"] == 1


def test_dfm_comparison_preview_only_is_inconclusive() -> None:
    comparison = compare_engine_results(
        "dfm",
        {
            "dfm": {
                "feature_names": ["a"],
                "unit_ids": ["u1"],
                "nnz": 300,
                "sparse": {
                    "row": [0] * 200,
                    "col": [0] * 200,
                    "data": [1.0] * 200,
                    "truncated": True,
                    "nnz": 300,
                    "nnz_exported": 200,
                },
            }
        },
        {
            "feature_names": ["a"],
            "unit_ids": ["u1"],
            "nnz": 300,
            "sparse_coo": {
                "rows": [0] * 200,
                "cols": [0] * 200,
                "values": [1.0] * 200,
                "preview_only": True,
            },
        },
    )
    assert comparison["comparison_complete"] is False
    assert comparison["cells_equal"] is None
    assert comparison["comparison_status"] == "inconclusive"


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


def test_dictionary_comparison_spans_and_prevalence() -> None:
    payload = {
        "total_hits": 2,
        "unit_prevalence": 0.5,
        "document_prevalence": 1.0,
        "by_category": {"theme": {"category": "theme", "hits": 2}},
        "matches": [
            {
                "text_unit_id": "u1",
                "matched_expression": "alpha",
                "token_start": 0,
                "token_end": 1,
                "category": "theme",
                "subcategory": "core",
            },
            {
                "text_unit_id": "u2",
                "matched_expression": "beta",
                "token_start": 1,
                "token_end": 2,
                "category": "theme",
                "subcategory": "core",
            },
        ],
    }
    equal = compare_engine_results("dictionary", payload, {"analysis_result": {"results": payload}})
    assert equal["comparison_status"] == "equal"
    assert equal["spans_equal"] is True
    assert equal["categories_equal"] is True

    mismatched = compare_engine_results(
        "dictionary",
        payload,
        {**payload, "total_hits": 1, "matches": payload["matches"][:1]},
    )
    assert mismatched["comparison_status"] == "unequal"
    assert mismatched["total_hits_equal"] is False


def test_keyness_comparison_uses_documented_tolerances() -> None:
    left = {
        "method": "log_likelihood",
        "features": [
            {
                "feature": "freedom",
                "freq_a": 2,
                "freq_b": 0,
                "keyness_statistic": 1.23456789,
                "p_value": 0.01,
                "p_adjusted": 0.02,
                "effect_direction": "A",
                "log_ratio": 1.5,
            }
        ],
    }
    right = {
        "method": "log_likelihood",
        "features": [
            {
                "feature": "freedom",
                "freq_a": 2,
                "freq_b": 0,
                "keyness_statistic": 1.23456780,
                "p_value": 0.01,
                "p_value_adjusted": 0.02,
                "effect_direction": "A",
                "log_ratio": 1.5,
            }
        ],
    }
    equal = compare_engine_results("keyness", left, right)
    assert equal["features_equal"] is True
    assert equal["comparison_status"] == "equal"

    unequal = compare_engine_results(
        "keyness",
        left,
        {
            "method": "log_likelihood",
            "features": [{**right["features"][0], "freq_a": 9}],
        },
    )
    assert unequal["features_equal"] is False
    assert "freedom" in unequal["feature_differences"]


def test_cooccurrence_refuses_cross_method_comparison() -> None:
    pair_pmi = {
        "term_a": "a",
        "term_b": "b",
        "count": 1,
        "freq_a": 1,
        "freq_b": 1,
        "pmi": 0.1,
        "association_score": 0.1,
    }
    pair_npmi = {
        "term_a": "a",
        "term_b": "b",
        "count": 1,
        "freq_a": 1,
        "freq_b": 1,
        "npmi": 0.1,
        "association_score": 0.1,
    }
    inconclusive = compare_engine_results(
        "cooccurrence",
        {"association_method": "pmi", "directional": False, "pairs": [pair_pmi]},
        {"association_method": "npmi", "directional": False, "pairs": [pair_npmi]},
    )
    assert inconclusive["comparison_status"] == "inconclusive"
    assert inconclusive["pairs_equal"] is None

    equal = compare_engine_results(
        "cooccurrence",
        {
            "association_method": "pmi",
            "directional": False,
            "pairs": [
                {
                    "term_a": "a",
                    "term_b": "b",
                    "count": 2,
                    "freq_a": 3,
                    "freq_b": 4,
                    "pmi": 0.5,
                    "npmi": 0.4,
                    "dice": 0.3,
                    "log_dice": 0.2,
                    "t_score": 0.1,
                    "association_score": 0.5,
                }
            ],
        },
        {
            "association_method": "pmi",
            "directional": False,
            "pairs": [
                {
                    "term_a": "a",
                    "term_b": "b",
                    "count": 2,
                    "freq_a": 3,
                    "freq_b": 4,
                    "pmi": 0.5,
                    "npmi": 0.4,
                    "dice": 0.3,
                    "log_dice": 0.2,
                    "t_score": 0.1,
                    "association_score": 0.5,
                }
            ],
        },
    )
    assert equal["pairs_equal"] is True
    assert equal["comparison_status"] == "equal"
