/**
 * Careful, non-prescriptive interpretation helpers for chance-corrected agreement.
 * Benchmarks are illustrative heuristics — disciplines and designs differ.
 */

export type AgreementInterpretation = {
    /** Short heuristic phrase shown near the metric (e.g. "Substantial agreement"). */
    label: string;
    /** Always shown with the label; stresses non-universal thresholds. */
    caveat: string;
};

const CAVEAT =
    "Illustrative heuristic only. Benchmarks for κ/α vary by discipline, codebook difficulty, prevalence, and research purpose — do not treat cutoffs as universal rules.";

/**
 * Map a chance-corrected coefficient to an illustrative verbal band.
 * Inspired by commonly cited Landis & Koch / Krippendorff-style ranges,
 * but labeled as heuristic to avoid oversimplification.
 */
export function interpretChanceCorrectedAgreement(
    value: number | null | undefined
): AgreementInterpretation | null {
    if (value == null || !Number.isFinite(value)) return null;

    let label: string;
    if (value < 0) {
        label = "Below chance";
    } else if (value < 0.2) {
        label = "Slight agreement";
    } else if (value < 0.4) {
        label = "Fair agreement";
    } else if (value < 0.6) {
        label = "Moderate agreement";
    } else if (value < 0.8) {
        label = "Substantial agreement";
    } else {
        label = "Almost perfect agreement";
    }

    return { label, caveat: CAVEAT };
}

export function formatAgreementWithInterpretation(
    value: number | null | undefined,
    digits = 2
): { valueText: string; interpretation: AgreementInterpretation | null } {
    const valueText =
        value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
    return {
        valueText,
        interpretation: interpretChanceCorrectedAgreement(value),
    };
}

export type ReliabilityCoverageSummary = {
    maxCoders: number | null;
    totalPairableUnits: number | null;
    labelsWithCoders: number;
};

export function summarizeReliabilityCoverage(
    byLabel: Record<string, { n_coders?: number | null; n_pairable_units?: number | null }>
): ReliabilityCoverageSummary {
    const rows = Object.values(byLabel);
    if (!rows.length) {
        return { maxCoders: null, totalPairableUnits: null, labelsWithCoders: 0 };
    }
    let maxCoders = 0;
    let totalPairable = 0;
    let labelsWithCoders = 0;
    for (const row of rows) {
        const coders = row.n_coders ?? 0;
        if (coders > 0) labelsWithCoders += 1;
        if (coders > maxCoders) maxCoders = coders;
        totalPairable += row.n_pairable_units ?? 0;
    }
    return {
        maxCoders: maxCoders > 0 ? maxCoders : null,
        totalPairableUnits: totalPairable > 0 ? totalPairable : null,
        labelsWithCoders,
    };
}
