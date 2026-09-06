from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _EvaluationCaseResult:
    case_id: str
    ai_run_id: str
    score: float
    passed: bool
    notes: str


def score_evaluation_case(
    output_text: str | None,
    output_json: dict | None,
    retrieved_chunk_ids: list[str],
    case,
) -> tuple[float, bool, str]:
    output_score, output_passed, output_note = _score_expected_output(
        output_text, output_json, case
    )
    expected_chunks = set(case.expected_chunk_ids_json or [])
    if not expected_chunks:
        return output_score, output_passed, output_note
    citation_hit_rate = len(expected_chunks & set(retrieved_chunk_ids)) / len(expected_chunks)
    has_expected_output = case.expected_output_json is not None or bool(
        (case.expected_output_text or "").strip()
    )
    score = (
        (output_score * 0.7) + (citation_hit_rate * 0.3)
        if has_expected_output
        else citation_hit_rate
    )
    passed = citation_hit_rate == 1.0 and (output_passed if has_expected_output else True)
    return round(score, 4), passed, f"{output_note}; citation hit rate {citation_hit_rate:.2f}"


def _score_expected_output(
    output_text: str | None,
    output_json: dict | None,
    case,
) -> tuple[float, bool, str]:
    if case.expected_output_json is not None:
        passed = output_json == case.expected_output_json
        return (1.0 if passed else 0.0, passed, "JSON exact match")
    expected_text = (case.expected_output_text or "").strip()
    actual_text = (output_text or "").strip()
    if expected_text:
        passed = expected_text.lower() == actual_text.lower()
        if passed:
            return 1.0, True, "Exact text match"
        partial = 1.0 if expected_text.lower() in actual_text.lower() else 0.0
        return partial, partial >= 1.0, "Substring text comparison"
    return 0.0, False, "No expected output defined"
