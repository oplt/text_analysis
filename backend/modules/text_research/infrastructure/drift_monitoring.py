"""Distribution drift metrics for production classifier monitoring.

Scores are descriptive only — no automatic alerting thresholds are applied.
"""

from __future__ import annotations

import math
from typing import Any


def _normalize_counts(counts: dict[str, int | float]) -> dict[str, float]:
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0:
        return {str(k): 0.0 for k in counts}
    return {str(k): max(0, int(v)) / total for k, v in counts.items()}


def prediction_distribution_drift(
    baseline_label_counts: dict[str, int | float],
    current_label_counts: dict[str, int | float],
) -> dict[str, Any]:
    """Total variation distance and a PSI-like score over label distributions."""
    labels = sorted(set(baseline_label_counts) | set(current_label_counts))
    baseline = _normalize_counts({label: baseline_label_counts.get(label, 0) for label in labels})
    current = _normalize_counts({label: current_label_counts.get(label, 0) for label in labels})

    tvd = 0.5 * sum(abs(baseline[label] - current[label]) for label in labels)

    psi = 0.0
    for label in labels:
        expected = baseline[label]
        observed = current[label]
        if expected <= 0 and observed <= 0:
            continue
        if expected <= 0 or observed <= 0:
            # Laplace-style smoothing for PSI when a bin is empty on one side.
            expected = max(expected, 1e-6)
            observed = max(observed, 1e-6)
        psi += (observed - expected) * math.log(observed / expected)

    return {
        "metric": "prediction_distribution",
        "labels": labels,
        "baseline_counts": {label: int(baseline_label_counts.get(label, 0)) for label in labels},
        "current_counts": {label: int(current_label_counts.get(label, 0)) for label in labels},
        "baseline_proportions": baseline,
        "current_proportions": current,
        "total_variation_distance": tvd,
        "psi_like": psi,
        "interpretation": (
            "Higher TVD/PSI indicate larger shifts in predicted label proportions. "
            "No automatic alert is raised — review context and sample size."
        ),
    }


def score_distribution_drift(
    baseline_scores: list[float],
    current_scores: list[float],
) -> dict[str, Any]:
    """KS statistic when scipy is available; otherwise mean/std shift summary."""
    baseline = [float(x) for x in baseline_scores]
    current = [float(x) for x in current_scores]

    result: dict[str, Any] = {
        "metric": "score_distribution",
        "baseline_n": len(baseline),
        "current_n": len(current),
    }

    if not baseline or not current:
        result["note"] = "Insufficient scores for comparison"
        return result

    try:
        from scipy import stats

        ks = stats.ks_2samp(baseline, current, method="auto")
        result.update(
            {
                "method": "ks_2samp",
                "statistic": float(ks.statistic),
                "pvalue": float(ks.pvalue),
            }
        )
    except Exception:  # noqa: BLE001

        def _mean_std(values: list[float]) -> tuple[float, float]:
            mean = sum(values) / len(values)
            if len(values) < 2:
                return mean, 0.0
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            return mean, math.sqrt(var)

        b_mean, b_std = _mean_std(baseline)
        c_mean, c_std = _mean_std(current)
        result.update(
            {
                "method": "mean_std_shift",
                "baseline_mean": b_mean,
                "baseline_std": b_std,
                "current_mean": c_mean,
                "current_std": c_std,
                "mean_shift": c_mean - b_mean,
                "std_shift": c_std - b_std,
            }
        )

    return result


def feature_presence_drift(
    baseline_top_terms: list[str],
    current_top_terms: list[str],
) -> dict[str, Any]:
    """Jaccard similarity over top feature terms."""
    baseline_set = {str(term) for term in baseline_top_terms}
    current_set = {str(term) for term in current_top_terms}
    union = baseline_set | current_set
    intersection = baseline_set & current_set
    jaccard = len(intersection) / len(union) if union else 1.0

    return {
        "metric": "feature_presence",
        "baseline_terms": list(baseline_top_terms),
        "current_terms": list(current_top_terms),
        "jaccard_similarity": jaccard,
        "added_terms": sorted(current_set - baseline_set),
        "removed_terms": sorted(baseline_set - current_set),
    }


def build_drift_report(*, baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Combine available baseline/current aggregates into one drift report."""
    sections: dict[str, Any] = {}

    baseline_labels = baseline.get("label_counts")
    current_labels = current.get("label_counts")
    if isinstance(baseline_labels, dict) and isinstance(current_labels, dict):
        sections["prediction_distribution"] = prediction_distribution_drift(
            baseline_labels, current_labels
        )

    baseline_scores = baseline.get("scores")
    current_scores = current.get("scores")
    if isinstance(baseline_scores, list) and isinstance(current_scores, list):
        score_section = score_distribution_drift(baseline_scores, current_scores)
        score_section["kind"] = "confidence"
        sections["score_distribution"] = score_section

    baseline_uncertainties = baseline.get("uncertainties")
    current_uncertainties = current.get("uncertainties")
    if (
        isinstance(baseline_uncertainties, list)
        and isinstance(current_uncertainties, list)
        and (baseline_uncertainties or current_uncertainties)
    ):
        uncertainty_section = score_distribution_drift(
            baseline_uncertainties, current_uncertainties
        )
        uncertainty_section["kind"] = "uncertainty"
        uncertainty_section["metric"] = "uncertainty_distribution"
        sections["uncertainty_distribution"] = uncertainty_section

    baseline_terms = baseline.get("top_terms")
    current_terms = current.get("top_terms")
    if isinstance(baseline_terms, list) and isinstance(current_terms, list):
        sections["feature_presence"] = feature_presence_drift(baseline_terms, current_terms)

    warning_level = compute_warning_level(sections)
    n_baseline = baseline.get("n")
    n_current = current.get("n")
    if not isinstance(n_baseline, int):
        n_baseline = (
            int(sum(int(v) for v in baseline_labels.values()))
            if isinstance(baseline_labels, dict)
            else 0
        )
    if not isinstance(n_current, int):
        n_current = (
            int(sum(int(v) for v in current_labels.values()))
            if isinstance(current_labels, dict)
            else 0
        )

    return {
        "baseline": baseline,
        "current": current,
        "sections": sections,
        "summary": {
            "section_count": len(sections),
            "has_prediction_drift": "prediction_distribution" in sections,
            "has_score_drift": "score_distribution" in sections,
            "has_uncertainty_drift": "uncertainty_distribution" in sections,
            "has_feature_drift": "feature_presence" in sections,
            "warning_level": warning_level,
            "n_baseline": n_baseline,
            "n_current": n_current,
            "n_observations": n_baseline + n_current,
        },
    }


# Advisory bands aligned with frontend driftDiagnostics (descriptive, not alerts).
_ADVISORY = {
    "tvd": {"watch": 0.1, "investigate": 0.25},
    "psi": {"watch": 0.1, "investigate": 0.25},
    "ks": {"watch": 0.1, "investigate": 0.25},
    "jaccard": {"watch": 0.7, "investigate": 0.5},  # lower is worse
}


def _band_higher_is_worse(value: float, bands: dict[str, float]) -> str:
    if value >= bands["investigate"]:
        return "investigate"
    if value >= bands["watch"]:
        return "watch"
    return "ok"


def _band_lower_is_worse(value: float, bands: dict[str, float]) -> str:
    if value <= bands["investigate"]:
        return "investigate"
    if value <= bands["watch"]:
        return "watch"
    return "ok"


def compute_warning_level(sections: dict[str, Any]) -> str:
    """Worst advisory band across available sections (ok < watch < investigate)."""
    rank = {"ok": 0, "watch": 1, "investigate": 2}
    level = "ok"

    def escalate(candidate: str) -> None:
        nonlocal level
        if rank.get(candidate, 0) > rank.get(level, 0):
            level = candidate

    pred = sections.get("prediction_distribution")
    if isinstance(pred, dict):
        tvd = pred.get("total_variation_distance")
        psi = pred.get("psi_like")
        if isinstance(tvd, (int, float)):
            escalate(_band_higher_is_worse(float(tvd), _ADVISORY["tvd"]))
        if isinstance(psi, (int, float)):
            escalate(_band_higher_is_worse(float(psi), _ADVISORY["psi"]))

    for key in ("score_distribution", "uncertainty_distribution"):
        scores = sections.get(key)
        if not isinstance(scores, dict):
            continue
        ks = scores.get("statistic")
        if isinstance(ks, (int, float)):
            escalate(_band_higher_is_worse(float(ks), _ADVISORY["ks"]))
        elif isinstance(scores.get("mean_shift"), (int, float)):
            escalate("watch" if abs(float(scores["mean_shift"])) >= 0.1 else "ok")

    features = sections.get("feature_presence")
    if isinstance(features, dict):
        jaccard = features.get("jaccard_similarity")
        if isinstance(jaccard, (int, float)):
            escalate(_band_lower_is_worse(float(jaccard), _ADVISORY["jaccard"]))

    performance = sections.get("performance")
    if isinstance(performance, dict):
        drop = performance.get("difference")
        if isinstance(drop, (int, float)):
            # Absolute accuracy drop; higher magnitude is worse when current is lower.
            escalate(_band_higher_is_worse(abs(float(drop)), {"watch": 0.03, "investigate": 0.08}))

    return level
