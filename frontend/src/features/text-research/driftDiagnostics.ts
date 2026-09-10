import { asRecord, num } from "./modelRegistryUtils";

export type DriftKind =
    | "feature_input"
    | "prediction_distribution"
    | "class_prevalence"
    | "score_distribution"
    | "uncertainty_distribution"
    | "performance";

export type DriftRowStatus = "ok" | "watch" | "investigate" | "unavailable";

export type DriftDiagnosticRow = {
    id: string;
    kind: DriftKind;
    kindLabel: string;
    metric: string;
    baseline: string;
    current: string;
    value: string;
    threshold: string;
    status: DriftRowStatus;
    note: string;
};

export type DriftProvenance = {
    baselinePredictionSetId: string | null;
    currentPredictionSetId: string | null;
    baselineAnalysisRunId: string | null;
    currentAnalysisRunId: string | null;
    baselineTrainedModelId: string | null;
    currentTrainedModelId: string | null;
    nBaseline: number | null;
    nCurrent: number | null;
    nObservations: number | null;
    aggregation: string | null;
    mode: string | null;
};

/** Advisory review bands only — never claim proven model degradation. */
export const DRIFT_ADVISORY = {
    tvd: { watch: 0.1, investigate: 0.25 },
    psi: { watch: 0.1, investigate: 0.25 },
    jaccard: { watch: 0.7, investigate: 0.5 }, // lower is worse
    ks: { watch: 0.1, investigate: 0.25 },
    f1_drop: { watch: 0.03, investigate: 0.08 },
} as const;

export const DRIFT_DISCLAIMER =
    "Distribution drift flags a shift in inputs, predictions, or prevalence. " +
    "It is not proof of model degradation. Confirm with labeled evaluation before lifecycle changes.";

function formatNum(value: number | null | undefined, digits = 3): string {
    if (value == null || Number.isNaN(value)) return "—";
    return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

function bandHigherIsWorse(
    value: number,
    bands: { watch: number; investigate: number }
): DriftRowStatus {
    if (value >= bands.investigate) return "investigate";
    if (value >= bands.watch) return "watch";
    return "ok";
}

function bandLowerIsWorse(
    value: number,
    bands: { watch: number; investigate: number }
): DriftRowStatus {
    if (value <= bands.investigate) return "investigate";
    if (value <= bands.watch) return "watch";
    return "ok";
}

function sampleSizeLabel(counts: Record<string, unknown> | null): string {
    if (!counts) return "—";
    const total = Object.values(counts).reduce<number>((sum, raw) => {
        const n = num(raw);
        return sum + (n ?? 0);
    }, 0);
    return total > 0 ? `${Math.round(total)} units` : "—";
}

function strOrNull(value: unknown): string | null {
    return typeof value === "string" && value.trim() ? value : null;
}

function pushScoreRows(
    rows: DriftDiagnosticRow[],
    section: Record<string, unknown> | null,
    opts: { idPrefix: string; kind: DriftKind; kindLabel: string; note: string }
): void {
    if (!section) return;
    const ks = num(section.statistic);
    const meanShift = num(section.mean_shift);
    const method = String(section.method ?? "score");
    if (ks != null) {
        rows.push({
            id: `${opts.idPrefix}-ks`,
            kind: opts.kind,
            kindLabel: opts.kindLabel,
            metric: `KS statistic (${method})`,
            baseline: `${num(section.baseline_n) ?? "—"} values`,
            current: `${num(section.current_n) ?? "—"} values`,
            value: formatNum(ks),
            threshold: `watch ≥ ${DRIFT_ADVISORY.ks.watch}; investigate ≥ ${DRIFT_ADVISORY.ks.investigate}`,
            status: bandHigherIsWorse(ks, DRIFT_ADVISORY.ks),
            note: opts.note,
        });
        return;
    }
    rows.push({
        id: `${opts.idPrefix}-mean`,
        kind: opts.kind,
        kindLabel: opts.kindLabel,
        metric: `Mean shift (${method})`,
        baseline: formatNum(num(section.baseline_mean)),
        current: formatNum(num(section.current_mean)),
        value: formatNum(meanShift),
        threshold: "Review mean/std shifts qualitatively",
        status: meanShift == null ? "unavailable" : Math.abs(meanShift) >= 0.1 ? "watch" : "ok",
        note: String(section.note ?? opts.note),
    });
}

export function parseDriftReport(report: Record<string, unknown> | null | undefined): {
    rows: DriftDiagnosticRow[];
    analysisRunId: string | null;
    sectionCount: number;
    warningLevel: DriftRowStatus;
    provenance: DriftProvenance;
} {
    const emptyProvenance: DriftProvenance = {
        baselinePredictionSetId: null,
        currentPredictionSetId: null,
        baselineAnalysisRunId: null,
        currentAnalysisRunId: null,
        baselineTrainedModelId: null,
        currentTrainedModelId: null,
        nBaseline: null,
        nCurrent: null,
        nObservations: null,
        aggregation: null,
        mode: null,
    };

    if (!report) {
        return {
            rows: [],
            analysisRunId: null,
            sectionCount: 0,
            warningLevel: "unavailable",
            provenance: emptyProvenance,
        };
    }

    const sections = asRecord(report.sections) ?? {};
    const baseline = asRecord(report.baseline);
    const current = asRecord(report.current);
    const summary = asRecord(report.summary) ?? {};
    const provenanceRaw = asRecord(report.provenance) ?? {};
    const baselineCounts = asRecord(baseline?.label_counts);
    const currentCounts = asRecord(current?.label_counts);
    const baselineSample = sampleSizeLabel(baselineCounts);
    const currentSample = sampleSizeLabel(currentCounts);
    const rows: DriftDiagnosticRow[] = [];

    const prediction = asRecord(sections.prediction_distribution);
    if (prediction) {
        const tvd = num(prediction.total_variation_distance);
        const psi = num(prediction.psi_like);
        rows.push({
            id: "pred-tvd",
            kind: "prediction_distribution",
            kindLabel: "Label-distribution drift",
            metric: "Total variation distance",
            baseline: baselineSample,
            current: currentSample,
            value: formatNum(tvd),
            threshold: `watch ≥ ${DRIFT_ADVISORY.tvd.watch}; investigate ≥ ${DRIFT_ADVISORY.tvd.investigate}`,
            status: tvd == null ? "unavailable" : bandHigherIsWorse(tvd, DRIFT_ADVISORY.tvd),
            note: "Shift in predicted label mix. Not proven performance loss.",
        });
        rows.push({
            id: "pred-psi",
            kind: "prediction_distribution",
            kindLabel: "Label-distribution drift",
            metric: "PSI-like",
            baseline: baselineSample,
            current: currentSample,
            value: formatNum(psi),
            threshold: `watch ≥ ${DRIFT_ADVISORY.psi.watch}; investigate ≥ ${DRIFT_ADVISORY.psi.investigate}`,
            status: psi == null ? "unavailable" : bandHigherIsWorse(psi, DRIFT_ADVISORY.psi),
            note: "Population stability index–style score on predicted labels.",
        });
        rows.push({
            id: "class-prevalence",
            kind: "class_prevalence",
            kindLabel: "Class prevalence",
            metric: "Label proportion shift (TVD)",
            baseline: baselineSample,
            current: currentSample,
            value: formatNum(tvd),
            threshold: `watch ≥ ${DRIFT_ADVISORY.tvd.watch}; investigate ≥ ${DRIFT_ADVISORY.tvd.investigate}`,
            status: tvd == null ? "unavailable" : bandHigherIsWorse(tvd, DRIFT_ADVISORY.tvd),
            note: "Prevalence of predicted classes changed. Check sampling / label mix before blaming the model.",
        });
    }

    pushScoreRows(rows, asRecord(sections.score_distribution), {
        idPrefix: "score",
        kind: "score_distribution",
        kindLabel: "Confidence drift",
        note: "Confidence/score distribution shift. Does not alone prove accuracy drop.",
    });

    pushScoreRows(rows, asRecord(sections.uncertainty_distribution), {
        idPrefix: "uncertainty",
        kind: "uncertainty_distribution",
        kindLabel: "Uncertainty drift",
        note: "Model uncertainty distribution shift when uncertainty scores are persisted.",
    });

    const features = asRecord(sections.feature_presence);
    if (features) {
        const jaccard = num(features.jaccard_similarity);
        rows.push({
            id: "feature-jaccard",
            kind: "feature_input",
            kindLabel: "Input / feature distribution",
            metric: "Top-term Jaccard similarity",
            baseline: `${(Array.isArray(features.baseline_terms) ? features.baseline_terms : []).length} terms`,
            current: `${(Array.isArray(features.current_terms) ? features.current_terms : []).length} terms`,
            value: formatNum(jaccard),
            threshold: `watch ≤ ${DRIFT_ADVISORY.jaccard.watch}; investigate ≤ ${DRIFT_ADVISORY.jaccard.investigate}`,
            status:
                jaccard == null ? "unavailable" : bandLowerIsWorse(jaccard, DRIFT_ADVISORY.jaccard),
            note: "Overlap of top input features/terms. Vocabulary shift ≠ proven degradation.",
        });
    }

    const performance = asRecord(sections.performance);
    if (performance) {
        const drop =
            num(performance.macro_f1_drop) ??
            num(performance.metric_drop) ??
            (num(performance.difference) != null
                ? Math.abs(num(performance.difference)!)
                : null);
        const baselineF1 =
            num(performance.baseline_macro_f1) ??
            num(performance.baseline_value) ??
            num(performance.baseline_accuracy);
        const currentF1 =
            num(performance.current_macro_f1) ??
            num(performance.current_value) ??
            num(performance.current_accuracy);
        rows.push({
            id: "perf-f1",
            kind: "performance",
            kindLabel: "Performance (labeled)",
            metric: String(performance.metric_name ?? "Labeled accuracy / Macro F1 drop"),
            baseline: formatNum(baselineF1),
            current: formatNum(currentF1),
            value: formatNum(drop),
            threshold: `watch ≥ ${DRIFT_ADVISORY.f1_drop.watch}; investigate ≥ ${DRIFT_ADVISORY.f1_drop.investigate}`,
            status: drop == null ? "unavailable" : bandHigherIsWorse(drop, DRIFT_ADVISORY.f1_drop),
            note: String(
                performance.note ??
                    "Only available when adjudicated/holdout labels exist. Distribution drift alone is not enough."
            ),
        });
    }

    const warningFromSummary = strOrNull(summary.warning_level);
    const worstFromRows = rows.reduce<DriftRowStatus>((worst, row) => {
        const order: DriftRowStatus[] = ["ok", "watch", "investigate"];
        if (row.status === "unavailable") return worst;
        const worstKey = worst === "unavailable" ? "ok" : worst;
        if (order.indexOf(row.status) > order.indexOf(worstKey)) {
            return row.status;
        }
        return worst;
    }, "unavailable");

    const provenance: DriftProvenance = {
        baselinePredictionSetId: strOrNull(provenanceRaw.baseline_prediction_set_id),
        currentPredictionSetId: strOrNull(provenanceRaw.current_prediction_set_id),
        baselineAnalysisRunId: strOrNull(provenanceRaw.baseline_analysis_run_id),
        currentAnalysisRunId: strOrNull(provenanceRaw.current_analysis_run_id),
        baselineTrainedModelId: strOrNull(provenanceRaw.baseline_trained_model_id),
        currentTrainedModelId: strOrNull(provenanceRaw.current_trained_model_id),
        nBaseline: num(provenanceRaw.n_baseline) ?? num(summary.n_baseline) ?? num(baseline?.n),
        nCurrent: num(provenanceRaw.n_current) ?? num(summary.n_current) ?? num(current?.n),
        nObservations: num(provenanceRaw.n_observations) ?? num(summary.n_observations) ?? null,
        aggregation: strOrNull(provenanceRaw.aggregation),
        mode: strOrNull(report.mode),
    };
    if (
        provenance.nObservations == null &&
        provenance.nBaseline != null &&
        provenance.nCurrent != null
    ) {
        provenance.nObservations = provenance.nBaseline + provenance.nCurrent;
    }

    return {
        rows,
        analysisRunId: typeof report.analysis_run_id === "string" ? report.analysis_run_id : null,
        sectionCount: num(summary.section_count) ?? rows.length,
        warningLevel: (warningFromSummary as DriftRowStatus) || worstFromRows,
        provenance,
    };
}

/** Merge optional performance section into a drift API report for display. */
export function withPerformanceSection(
    report: Record<string, unknown>,
    performance: {
        baseline_macro_f1: number | null;
        current_macro_f1: number | null;
        source: string;
    }
): Record<string, unknown> {
    const baseline = performance.baseline_macro_f1;
    const current = performance.current_macro_f1;
    if (baseline == null || current == null) return report;
    const drop = baseline - current;
    const sections = { ...(asRecord(report.sections) ?? {}) };
    sections.performance = {
        metric: "performance",
        metric_name: "Macro F1 drop",
        baseline_macro_f1: baseline,
        current_macro_f1: current,
        macro_f1_drop: drop,
        note: `Estimated from ${performance.source}. Requires labeled evaluation — not inferred from distribution drift.`,
    };
    const summary = { ...(asRecord(report.summary) ?? {}) };
    summary.has_performance_drift = true;
    summary.section_count = Object.keys(sections).length;
    return { ...report, sections, summary };
}

export function topTermsFromCoefficients(
    coefficients: Array<{ feature: string; coefficient: number }>,
    limit = 40
): string[] {
    const byAbs = [...coefficients].sort(
        (a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient)
    );
    const seen = new Set<string>();
    const terms: string[] = [];
    for (const row of byAbs) {
        const feature = row.feature?.trim();
        if (!feature || seen.has(feature)) continue;
        seen.add(feature);
        terms.push(feature);
        if (terms.length >= limit) break;
    }
    return terms;
}

export function predictionSetOptionLabel(ps: {
    id: string;
    trained_model_id: string;
    analysis_run_id: string;
    created_at: string;
    metadata?: Record<string, unknown>;
}): string {
    const unitCount = num(ps.metadata?.unit_count);
    const shortId = ps.id.slice(0, 8);
    const runShort = ps.analysis_run_id.slice(0, 8);
    const when = new Date(ps.created_at).toLocaleString();
    const n = unitCount != null ? `${Math.round(unitCount)} units` : "units unknown";
    return `${shortId}… · run ${runShort}… · ${n} · ${when}`;
}
