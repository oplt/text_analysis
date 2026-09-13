/**
 * Dataset / split helpers for Classification workspace (Phase 13).
 */

export type ClassCountItem = { label: string; value: number };

export type ClassImbalanceAssessment = {
    /** max/min positive count ratio among plotted classes (null if <2 classes). */
    imbalanceRatio: number | null;
    majorityShare: number | null;
    minorityShare: number | null;
    /** Human-readable warnings; empty when no concern. */
    warnings: string[];
};

/**
 * Flag severe class imbalance for scientific review (not an automatic blocker).
 * Heuristic: majority share ≥ 80% or max/min count ratio ≥ 5.
 */
export function assessClassImbalance(items: ClassCountItem[]): ClassImbalanceAssessment {
    const positive = items.filter((item) => item.value > 0);
    if (positive.length < 2) {
        return { imbalanceRatio: null, majorityShare: null, minorityShare: null, warnings: [] };
    }
    const total = positive.reduce((sum, item) => sum + item.value, 0);
    const counts = positive.map((item) => item.value);
    const max = Math.max(...counts);
    const min = Math.min(...counts);
    const imbalanceRatio = min > 0 ? max / min : null;
    const majorityShare = total > 0 ? max / total : null;
    const minorityShare = total > 0 ? min / total : null;
    const warnings: string[] = [];
    if (majorityShare != null && majorityShare >= 0.8) {
        warnings.push(
            `Class imbalance: the largest class is ${(majorityShare * 100).toFixed(0)}% of labeled units. Prefer macro F1 and consider class_weight=balanced or resampling before interpreting accuracy.`
        );
    } else if (imbalanceRatio != null && imbalanceRatio >= 5) {
        warnings.push(
            `Class imbalance: majority/minority count ratio is ${imbalanceRatio.toFixed(1)}×. Report per-class metrics and check whether rare classes have enough support for reliable evaluation.`
        );
    }
    return { imbalanceRatio, majorityShare, minorityShare, warnings };
}

export type SplitPlan = {
    testFraction: number;
    validationFraction: number;
    /** Remainder after val+test, floored at 0. */
    trainFraction: number;
};

export function plannedSplitFractions(testSize: number, valSize: number): SplitPlan {
    const testFraction = Math.max(0, Math.min(0.9, testSize));
    const validationFraction = Math.max(0, Math.min(0.9, valSize));
    const trainFraction = Math.max(0, 1 - testFraction - validationFraction);
    return { testFraction, validationFraction, trainFraction };
}

export function formatPercent(value: number, digits = 0): string {
    return `${(value * 100).toFixed(digits)}%`;
}
