import { useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    FormControlLabel,
    List,
    ListItemButton,
    ListItemText,
    MenuItem,
    Radio,
    RadioGroup,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import {
    PlaylistAddCheck as CodebookIcon,
    PlayArrow as RunIcon,
    RateReview as AdjudicateIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    computeReliability,
    computeCampaignReliability,
    getRun,
    getTextUnitContext,
    listAdjudications,
    listCampaignAdjudications,
    listAnnotationCampaigns,
    listDisagreements,
    listCampaignDisagreements,
    saveAdjudication,
    saveCampaignAdjudication,
    type DisagreementItem,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { DisabledWithReason } from "../../../components/ui/DisabledWithReason";
import { HelpTooltip } from "../../../components/ui/HelpTooltip";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { RunStatusPanel } from "../../../components/ui/RunStatusPanel";
import { SectionCard } from "../../../components/ui/SectionCard";
import { PageTabs } from "../../../components/ui/PageTabs";
import { AdvancedSettings } from "../../../components/ui/AdvancedSettings";
import { useTabQueryParam } from "../../../hooks/useTabQueryParam";
import type { HelpTermId } from "../../../config/helpText";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { MatrixHeatmap, MetricCards, ReliabilityComparisonChart, ResultsInspector } from "../components/ResearchCharts";
import { ProvenanceDrawer } from "../components/ProvenanceDrawer";
import { ProvenancePanel } from "../components/ProvenancePanel";
import { ScientificWarnings, collectScientificWarnings } from "../components/ScientificWarnings";
import {
    computeReliabilityDisabledReason,
    RELIABILITY_ANNOTATOR_HINT,
} from "../actionDisabledReasons";
import { useResearchContext } from "../hooks/useResearchContext";
import { useRunEvents } from "../hooks/useRunEvents";
import { activeRunRefetchInterval, isActiveRunStatus } from "../runPolling";
import type { AnnotationLabel } from "../types";
import {
    formatAgreementWithInterpretation,
    interpretChanceCorrectedAgreement,
    summarizeReliabilityCoverage,
} from "../reliabilityInterpretation";

type LabelDecision = "yes" | "no" | "uncertain";

type ConfidenceInterval = {
    lower?: number | null;
    upper?: number | null;
    confidence_level?: number | null;
    bootstrap_samples?: number | null;
    random_seed?: number | null;
};

type RawAgreement = {
    agreement?: number | null;
    sample_size?: number | null;
    ci?: ConfidenceInterval | null;
};

type CohensKappa = {
    observed_agreement?: number | null;
    expected_agreement?: number | null;
    kappa?: number | null;
    sample_size?: number | null;
    ci?: ConfidenceInterval | null;
};

type FleissKappa = {
    kappa?: number | null;
    n_raters?: number | null;
    n_units_included?: number | null;
    n_units_excluded?: number | null;
    reason?: string | null;
    ci?: ConfidenceInterval | null;
};

type KrippendorffAlpha = {
    alpha?: number | null;
    n_units?: number | null;
    n_coders?: number | null;
    missingness?: number | null;
    ci?: ConfidenceInterval | null;
};

type CoderPairAgreement = {
    coders?: string[];
    matrix?: Record<string, Record<string, { agreement?: number | null; n_common_units?: number }>>;
};

type LabelReliability = {
    label_id?: string;
    n_coders?: number;
    n_pairable_units?: number;
    n_units?: number;
    raw_agreement?: RawAgreement | null;
    cohens_kappa?: CohensKappa | null;
    fleiss_kappa?: FleissKappa | null;
    krippendorff_alpha?: KrippendorffAlpha | null;
    pairwise_cohens_kappa?: Record<string, CohensKappa> | null;
    coder_pair_agreement?: CoderPairAgreement | null;
    disagreement_count?: number;
    disagreements?: unknown;
    evaluation_messages?: string[];
    diagnostics?: string[];
    scientific_warnings?: string[];
    primary_metric?: { name?: string; value?: number | null } | null;
    primary_ci?: ConfidenceInterval | null;
    metadata?: Record<string, unknown>;
};

const STAT_HELP_TERM: Record<string, HelpTermId> = {
    kappa: "cohens_kappa",
    fleiss: "fleiss_kappa",
    alpha: "krippendorff_alpha",
    observed: "observed_agreement",
    expected: "expected_agreement",
    missingness: "missingness",
    n_coders: "n_coders",
    pairable: "pairable_units",
};

function formatMetric(value: number | null | undefined, digits = 3): string {
    if (value == null || Number.isNaN(value)) return "—";
    return value.toFixed(digits);
}

function formatCiRange(ci: ConfidenceInterval | null | undefined, digits = 2): string | null {
    if (ci?.lower == null || ci?.upper == null) return null;
    return `[${formatMetric(ci.lower, digits)}–${formatMetric(ci.upper, digits)}]`;
}

function formatMetricWithCi(
    value: number | null | undefined,
    ci: ConfidenceInterval | null | undefined,
    digits = 2
): string {
    const formatted = formatMetric(value, digits);
    const range = formatCiRange(ci, digits);
    return range ? `${formatted} ${range}` : formatted;
}

function primaryMetricLabel(name: string | undefined): string {
    switch (name) {
        case "cohens_kappa":
            return "Cohen's κ";
        case "fleiss_kappa":
            return "Fleiss' κ";
        case "krippendorff_alpha":
            return "Krippendorff's α";
        default:
            return "Primary metric";
    }
}

function reliabilityChartItems(byLabel: Record<string, LabelReliability>) {
    return Object.entries(byLabel).map(([label, row]) => ({
        label,
        kappa: (row.n_coders ?? 0) === 2 ? (row.cohens_kappa?.kappa ?? null) : null,
        fleiss: (row.n_coders ?? 0) >= 3 ? (row.fleiss_kappa?.kappa ?? null) : null,
        alpha: row.krippendorff_alpha?.alpha ?? null,
    }));
}

function StatLabel({ label, helpKey }: { label: string; helpKey: keyof typeof STAT_HELP_TERM }) {
    return (
        <HelpTooltip termId={STAT_HELP_TERM[helpKey]} variant="label">
            {label}
        </HelpTooltip>
    );
}

function pairMatrixValues(pair: CoderPairAgreement | null | undefined): {
    coders: string[];
    values: Array<Array<number | null>>;
    commonUnits: number[][];
} {
    const coders = pair?.coders ?? [];
    const matrix = pair?.matrix ?? {};
    const values = coders.map((row) =>
        coders.map((col) => {
            const cell = matrix[row]?.[col];
            return typeof cell?.agreement === "number" ? cell.agreement : null;
        })
    );
    const commonUnits = coders.map((row) =>
        coders.map((col) => matrix[row]?.[col]?.n_common_units ?? 0)
    );
    return { coders, values, commonUnits };
}

function InterpretationCaption({ value }: { value: number | null | undefined }) {
    const interpretation = interpretChanceCorrectedAgreement(value);
    if (!interpretation) return null;
    return (
        <HelpTooltip termId="agreement_benchmarks" variant="label">
            {interpretation.label}
        </HelpTooltip>
    );
}

function ReliabilityPrimaryMetrics({
    metrics,
    byLabel,
}: {
    metrics: Record<string, unknown>;
    byLabel: Record<string, LabelReliability> | null;
}) {
    const meanAlpha = metrics.mean_krippendorff_alpha as number | null | undefined;
    const meanFleiss = metrics.mean_fleiss_kappa as number | null | undefined;
    const meanCohen = metrics.mean_cohens_kappa as number | null | undefined;
    const alphaFmt = formatAgreementWithInterpretation(meanAlpha, 2);
    const fleissFmt = formatAgreementWithInterpretation(meanFleiss, 2);
    const cohenFmt = formatAgreementWithInterpretation(meanCohen, 2);
    const coverage = summarizeReliabilityCoverage(byLabel ?? {});

    return (
        <Stack spacing={1.5}>
            <MetricCards
                items={[
                    {
                        label: "Mean Krippendorff's α",
                        value: alphaFmt.valueText,
                        description: alphaFmt.interpretation ? (
                            <InterpretationCaption value={meanAlpha} />
                        ) : undefined,
                        helpTermId: "krippendorff_alpha",
                    },
                    {
                        label: "Mean Fleiss' κ",
                        value: fleissFmt.valueText,
                        description: fleissFmt.interpretation ? (
                            <InterpretationCaption value={meanFleiss} />
                        ) : undefined,
                        helpTermId: "fleiss_kappa",
                    },
                    {
                        label: "Mean Cohen's κ",
                        value: cohenFmt.valueText,
                        description: cohenFmt.interpretation ? (
                            <InterpretationCaption value={meanCohen} />
                        ) : undefined,
                        helpTermId: "cohens_kappa",
                    },
                    {
                        label: "Labels evaluated",
                        value:
                            metrics.labels_evaluated != null
                                ? String(metrics.labels_evaluated)
                                : "—",
                        helpTermId: "annotation_agreement",
                    },
                    {
                        label: "Coders (max across labels)",
                        value: coverage.maxCoders != null ? String(coverage.maxCoders) : "—",
                        helpTermId: "n_coders",
                    },
                    {
                        label: "Pairable units (sum)",
                        value:
                            coverage.totalPairableUnits != null
                                ? String(coverage.totalPairableUnits)
                                : "—",
                        helpTermId: "pairable_units",
                    },
                ]}
            />
            <Alert severity="info" icon={false}>
                <Stack direction="row" spacing={0.5} alignItems="flex-start">
                    <Typography variant="body2" color="text.secondary" sx={{ flex: 1 }}>
                        Verbal labels such as “substantial agreement” are{" "}
                        <strong>illustrative heuristics</strong>. Acceptable reliability depends on
                        discipline, codebook difficulty, prevalence, and intended use. Prefer reporting
                        the coefficient, confidence interval, sample size, and codebook version.
                    </Typography>
                    <HelpTooltip termId="agreement_benchmarks" />
                </Stack>
            </Alert>
        </Stack>
    );
}

function LabelReliabilitySummaryTable({
    byLabel,
}: {
    byLabel: Record<string, LabelReliability>;
}) {
    const rows = Object.entries(byLabel);
    if (!rows.length) return null;

    return (
        <Table size="small">
            <TableHead>
                <TableRow>
                    <TableCell>Label</TableCell>
                    <TableCell>Primary metric</TableCell>
                    <TableCell>CI</TableCell>
                    <TableCell>
                        <HelpTooltip termId="agreement_benchmarks" variant="label">
                            Heuristic reading
                        </HelpTooltip>
                    </TableCell>
                    <TableCell>
                        <StatLabel label="n (pairable)" helpKey="pairable" />
                    </TableCell>
                    <TableCell>
                        <StatLabel label="n coders" helpKey="n_coders" />
                    </TableCell>
                </TableRow>
            </TableHead>
            <TableBody>
                {rows.map(([labelName, row]) => {
                    const primaryName = row.primary_metric?.name;
                    const primaryValue = row.primary_metric?.value;
                    const ci = row.primary_ci ?? null;
                    const interpretation = interpretChanceCorrectedAgreement(
                        typeof primaryValue === "number" ? primaryValue : null
                    );
                    return (
                        <TableRow key={labelName}>
                            <TableCell>{labelName}</TableCell>
                            <TableCell>
                                {primaryName && primaryValue != null
                                    ? `${primaryMetricLabel(primaryName)} ${formatMetric(primaryValue, 2)}`
                                    : "—"}
                            </TableCell>
                            <TableCell>{formatCiRange(ci, 2) ?? "—"}</TableCell>
                            <TableCell>
                                {interpretation ? (
                                    <HelpTooltip termId="agreement_benchmarks" variant="label">
                                        {interpretation.label}
                                    </HelpTooltip>
                                ) : (
                                    "—"
                                )}
                            </TableCell>
                            <TableCell>
                                {row.n_pairable_units != null
                                    ? String(row.n_pairable_units)
                                    : "—"}
                            </TableCell>
                            <TableCell>
                                {row.n_coders != null ? String(row.n_coders) : "—"}
                            </TableCell>
                        </TableRow>
                    );
                })}
            </TableBody>
        </Table>
    );
}

function LabelReliabilityCard({
    labelName,
    row,
}: {
    labelName: string;
    row: LabelReliability;
}) {
    const nCoders = row.n_coders ?? 0;
    const twoCoders = nCoders === 2;
    const multiCoders = nCoders >= 3;
    const raw = row.raw_agreement ?? null;
    const kappa = row.cohens_kappa ?? null;
    const fleiss = row.fleiss_kappa ?? null;
    const alpha = row.krippendorff_alpha ?? null;
    const pairwiseCohen = row.pairwise_cohens_kappa ?? null;
    const { coders, values, commonUnits } = pairMatrixValues(row.coder_pair_agreement);
    const warnings = [
        ...(row.scientific_warnings ?? []),
        ...(row.evaluation_messages ?? []),
        ...(row.diagnostics ?? []),
    ];

    const primaryValue =
        typeof row.primary_metric?.value === "number" ? row.primary_metric.value : null;
    const primaryInterpretation = interpretChanceCorrectedAgreement(primaryValue);

    return (
        <Box sx={{ p: 2, borderRadius: 1, bgcolor: "action.hover" }}>
            <Stack spacing={1.5}>
                <Stack
                    direction={{ xs: "column", sm: "row" }}
                    spacing={1}
                    alignItems={{ sm: "baseline" }}
                    justifyContent="space-between"
                >
                    <Typography variant="subtitle1">{labelName}</Typography>
                    {row.primary_metric?.name && primaryValue != null ? (
                        <Typography variant="body2" color="text.secondary">
                            {primaryMetricLabel(row.primary_metric.name)}{" "}
                            {formatMetricWithCi(primaryValue, row.primary_ci, 2)}
                            {primaryInterpretation ? (
                                <>
                                    {" · "}
                                    <HelpTooltip termId="agreement_benchmarks" variant="label">
                                        {primaryInterpretation.label}
                                    </HelpTooltip>
                                </>
                            ) : null}
                        </Typography>
                    ) : null}
                </Stack>

                {twoCoders ? (
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>
                                    <StatLabel label="Raw agreement" helpKey="observed" />
                                </TableCell>
                                <TableCell>
                                    <StatLabel label="Cohen's κ" helpKey="kappa" />
                                </TableCell>
                                <TableCell>
                                    <StatLabel label="Krippendorff's α" helpKey="alpha" />
                                </TableCell>
                                <TableCell>
                                    <StatLabel label="Missingness" helpKey="missingness" />
                                </TableCell>
                                <TableCell>
                                    <StatLabel label="n coders" helpKey="n_coders" />
                                </TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            <TableRow>
                                <TableCell>
                                    {formatMetricWithCi(raw?.agreement, raw?.ci, 2)}
                                </TableCell>
                                <TableCell>
                                    {formatMetricWithCi(kappa?.kappa, kappa?.ci, 2)}
                                </TableCell>
                                <TableCell>
                                    {formatMetricWithCi(alpha?.alpha, alpha?.ci, 2)}
                                </TableCell>
                                <TableCell>{formatMetric(alpha?.missingness, 2)}</TableCell>
                                <TableCell>{String(nCoders)}</TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                ) : null}

                {multiCoders ? (
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>
                                    <StatLabel label="Fleiss' κ" helpKey="fleiss" />
                                </TableCell>
                                <TableCell>
                                    <StatLabel label="Krippendorff's α" helpKey="alpha" />
                                </TableCell>
                                <TableCell>
                                    <StatLabel label="Missingness" helpKey="missingness" />
                                </TableCell>
                                <TableCell>
                                    <StatLabel label="n coders" helpKey="n_coders" />
                                </TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            <TableRow>
                                <TableCell>
                                    {formatMetricWithCi(fleiss?.kappa, fleiss?.ci, 2)}
                                </TableCell>
                                <TableCell>
                                    {formatMetricWithCi(alpha?.alpha, alpha?.ci, 2)}
                                </TableCell>
                                <TableCell>{formatMetric(alpha?.missingness, 2)}</TableCell>
                                <TableCell>{String(nCoders)}</TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                ) : null}

                {nCoders > 0 && nCoders !== 2 && nCoders < 3 ? (
                    <Typography variant="body2" color="text.secondary">
                        Reliability metrics require at least two coders.
                    </Typography>
                ) : null}

                {row.disagreement_count != null ? (
                    <Typography variant="caption" color="text.secondary">
                        Disagreements: {row.disagreement_count}
                    </Typography>
                ) : null}

                {warnings.map((message) => (
                    <Alert key={message} severity="warning" sx={{ py: 0.5 }}>
                        {message}
                    </Alert>
                ))}

                {multiCoders && coders.length > 0 ? (
                    <Box>
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>
                            Coder-to-coder agreement
                        </Typography>
                        <MatrixHeatmap
                            rowLabels={coders}
                            colLabels={coders}
                            values={values}
                            formatCell={(v) => formatMetric(v, 2)}
                            cellDetails={(rowIndex, colIndex) =>
                                values[rowIndex]?.[colIndex] === null
                                    ? "No shared annotations"
                                    : `${commonUnits[rowIndex]?.[colIndex] ?? 0} shared units`
                            }
                        />
                    </Box>
                ) : null}

                {multiCoders && pairwiseCohen && Object.keys(pairwiseCohen).length > 0 ? (
                    <Box>
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>
                            Pairwise Cohen's κ
                        </Typography>
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Coder pair</TableCell>
                                    <TableCell>κ</TableCell>
                                    <TableCell>Observed</TableCell>
                                    <TableCell>n</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {Object.entries(pairwiseCohen).map(([pair, stats]) => (
                                    <TableRow key={pair}>
                                        <TableCell>{pair.replace("|", " × ")}</TableCell>
                                        <TableCell>{formatMetric(stats.kappa, 2)}</TableCell>
                                        <TableCell>
                                            {formatMetric(stats.observed_agreement, 2)}
                                        </TableCell>
                                        <TableCell>
                                            {stats.sample_size != null
                                                ? String(stats.sample_size)
                                                : "—"}
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </Box>
                ) : null}
            </Stack>
        </Box>
    );
}

function disagreementKey(item: DisagreementItem): string {
    return `${item.text_unit_id}:${item.label_id}`;
}

function AdjudicationDecisionForm({
    selectedDisagreement,
    selectedLabel,
    selectedIndex,
    disagreementCount,
    savePending,
    onSave,
    onNext,
}: {
    selectedDisagreement: DisagreementItem;
    selectedLabel: AnnotationLabel | null;
    selectedIndex: number;
    disagreementCount: number;
    savePending: boolean;
    onSave: (payload: { finalValue: LabelDecision; comment: string; advance: boolean }) => void;
    onNext: () => void;
}) {
    const [finalValue, setFinalValue] = useState<LabelDecision | "">("");
    const [adjudicationComment, setAdjudicationComment] = useState("");

    return (
        <>
            <Box>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    Coder judgments
                </Typography>
                <Stack spacing={0.5}>
                    {Object.entries(selectedDisagreement.judgements).map(([coderId, value]) => (
                        <Typography key={coderId} variant="body2">
                            {coderId.slice(0, 8)}… → {value}
                        </Typography>
                    ))}
                </Stack>
            </Box>

            <Box>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    Codebook definition
                </Typography>
                {selectedLabel ? (
                    <LabelDefinition label={selectedLabel} />
                ) : (
                    <Typography variant="body2" color="text.secondary">
                        Label {selectedDisagreement.label_name} (definition not loaded in context).
                    </Typography>
                )}
            </Box>

            <Box>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    Final value
                </Typography>
                <RadioGroup
                    row
                    value={finalValue}
                    onChange={(e) => setFinalValue(e.target.value as LabelDecision)}
                >
                    <FormControlLabel value="yes" control={<Radio size="small" />} label="Yes" />
                    <FormControlLabel value="no" control={<Radio size="small" />} label="No" />
                    <FormControlLabel
                        value="uncertain"
                        control={<Radio size="small" />}
                        label="Uncertain"
                    />
                </RadioGroup>
                <TextField
                    label="Comment"
                    value={adjudicationComment}
                    onChange={(e) => setAdjudicationComment(e.target.value)}
                    fullWidth
                    multiline
                    minRows={2}
                    sx={{ mt: 1 }}
                />
            </Box>

            <Stack direction="row" spacing={1}>
                <Button
                    variant="contained"
                    disabled={!finalValue || savePending}
                    onClick={() =>
                        onSave({
                            finalValue: finalValue as LabelDecision,
                            comment: adjudicationComment,
                            advance: false,
                        })
                    }
                >
                    Save
                </Button>
                <Button
                    variant="contained"
                    disabled={!finalValue || savePending}
                    onClick={() =>
                        onSave({
                            finalValue: finalValue as LabelDecision,
                            comment: adjudicationComment,
                            advance: true,
                        })
                    }
                >
                    Save + Next disagreement
                </Button>
                <Button
                    variant="outlined"
                    disabled={selectedIndex < 0 || selectedIndex >= disagreementCount - 1}
                    onClick={onNext}
                >
                    Next disagreement
                </Button>
            </Stack>
        </>
    );
}

function LabelDefinition({ label }: { label: AnnotationLabel }) {
    return (
        <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: "action.hover" }}>
            <Typography variant="subtitle2">{label.name}</Typography>
            {label.description ? (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                    {label.description}
                </Typography>
            ) : (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                    No definition provided for this label.
                </Typography>
            )}
            {label.inclusion_criteria ? (
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                    Include: {label.inclusion_criteria}
                </Typography>
            ) : null}
            {label.exclusion_criteria ? (
                <Typography variant="caption" display="block">
                    Exclude: {label.exclusion_criteria}
                </Typography>
            ) : null}
        </Box>
    );
}

export default function ReliabilityView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const client = useQueryClient();
    const { showToast } = useSnackbar();

    const [runId, setRunId] = useState<string | null>(null);
    const [selectedDisagreementKey, setSelectedDisagreementKey] = useState<string | null>(null);
    const [selectedReliabilityLabel, setSelectedReliabilityLabel] = useState<string | null>(null);
    const [selectedCampaignId, setSelectedCampaignId] = useState<string>("");
    const [bootstrapSamples, setBootstrapSamples] = useState(2000);
    const [confidenceLevel, setConfidenceLevel] = useState(0.95);
    const [randomSeed, setRandomSeed] = useState<string>("");
    const [tab, setTab] = useTabQueryParam(
        ["overview", "agreement", "disagreements", "by_coder", "methods"] as const,
        "overview",
        "tab",
        {
            aliases: {
                adjudication: "disagreements",
                history: "methods",
            },
        }
    );
    const sseConnected = useRunEvents(runId, ctx.projectId);

    const reliabilityReady =
        Boolean(ctx.selectedCorpusId) &&
        Boolean(ctx.selectedCodebookId) &&
        ctx.labels.length > 0;

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: ({ signal }) => getRun(runId!, signal),
        enabled: Boolean(runId),
        refetchInterval: (query) => activeRunRefetchInterval(query, sseConnected),
    });

    const disagreementsQuery = useQuery({
        queryKey: [
            ...queryKeys.textResearch.disagreements(ctx.selectedCorpusId, ctx.selectedCodebookId),
            selectedCampaignId || "legacy",
        ],
        queryFn: ({ signal }) =>
            selectedCampaignId
                ? listCampaignDisagreements(selectedCampaignId, ctx.selectedCodebookId, signal)
                : listDisagreements(ctx.selectedCorpusId, ctx.selectedCodebookId, signal),
        enabled: Boolean(ctx.selectedCorpusId && ctx.selectedCodebookId),
    });

    const adjudicationsQuery = useQuery({
        queryKey: [
            ...queryKeys.textResearch.adjudications(ctx.selectedCorpusId),
            selectedCampaignId || "legacy",
        ],
        queryFn: ({ signal }) =>
            selectedCampaignId
                ? listCampaignAdjudications(selectedCampaignId, signal)
                : listAdjudications(ctx.selectedCorpusId, signal),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const disagreements = useMemo(
        () => disagreementsQuery.data ?? [],
        [disagreementsQuery.data]
    );

    const effectiveDisagreementKey = useMemo(() => {
        if (
            selectedDisagreementKey &&
            disagreements.some((item) => disagreementKey(item) === selectedDisagreementKey)
        ) {
            return selectedDisagreementKey;
        }
        return disagreements[0] ? disagreementKey(disagreements[0]) : null;
    }, [disagreements, selectedDisagreementKey]);

    const selectedDisagreement = useMemo(() => {
        if (!effectiveDisagreementKey) return null;
        return (
            disagreements.find((item) => disagreementKey(item) === effectiveDisagreementKey) ??
            null
        );
    }, [disagreements, effectiveDisagreementKey]);

    const selectedIndex = selectedDisagreement
        ? disagreements.findIndex(
              (item) =>
                  item.text_unit_id === selectedDisagreement.text_unit_id &&
                  item.label_id === selectedDisagreement.label_id
          )
        : -1;

    const unitContextQuery = useQuery({
        queryKey: ["text-research", "unit-context", selectedDisagreement?.text_unit_id],
        queryFn: ({ signal }) => getTextUnitContext(selectedDisagreement!.text_unit_id, 2, signal),
        enabled: Boolean(selectedDisagreement?.text_unit_id),
    });

    const selectedLabel = useMemo(() => {
        if (!selectedDisagreement) return null;
        return ctx.labels.find((label) => label.id === selectedDisagreement.label_id) ?? null;
    }, [ctx.labels, selectedDisagreement]);

    const campaignsQuery = useQuery({
        queryKey: ["text-research", ctx.projectId, "annotation-campaigns", ctx.selectedCorpusId],
        queryFn: ({ signal }) =>
            listAnnotationCampaigns(ctx.projectId, ctx.selectedCorpusId, signal),
        enabled: Boolean(ctx.projectId && ctx.selectedCorpusId),
    });

    const campaigns = useMemo(() => campaignsQuery.data ?? [], [campaignsQuery.data]);

    const reliabilityMutation = useMutation({
        mutationFn: () => {
            const parsedSeed = randomSeed.trim() ? Number(randomSeed) : undefined;
            const payload = {
                codebook_id: ctx.selectedCodebookId,
                label_ids: ctx.labels.map((l) => l.id),
                bootstrap_samples: bootstrapSamples,
                confidence_level: confidenceLevel,
                random_seed:
                    parsedSeed != null && Number.isFinite(parsedSeed) ? parsedSeed : undefined,
            };
            return selectedCampaignId
                ? computeCampaignReliability(selectedCampaignId, payload)
                : computeReliability(ctx.selectedCorpusId, payload);
        },
        onSuccess: (run) => {
            setRunId(run.id);
            showToast({ message: "Reliability computation started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to compute reliability."),
                severity: "error",
            }),
    });

    const saveAdjudicationMutation = useMutation({
        mutationFn: async (options: {
            advance: boolean;
            finalValue: LabelDecision;
            comment: string;
            disagreement: DisagreementItem;
        }) => {
            const payload = {
                text_unit_id: options.disagreement.text_unit_id,
                label_id: options.disagreement.label_id,
                final_value: options.finalValue,
                comment: options.comment.trim() || undefined,
            };
            if (selectedCampaignId) {
                await saveCampaignAdjudication(selectedCampaignId, payload);
            } else {
                await saveAdjudication(payload);
            }
            return { ...options, saved: options.disagreement };
        },
        onSuccess: (options) => {
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.disagreements(
                    ctx.selectedCorpusId,
                    ctx.selectedCodebookId
                ),
            });
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.adjudications(ctx.selectedCorpusId),
            });
            showToast({ message: "Adjudication saved.", severity: "success" });
            if (options.advance) {
                const remaining = disagreements.filter(
                    (item) =>
                        !(
                            item.text_unit_id === options.saved.text_unit_id &&
                            item.label_id === options.saved.label_id
                        )
                );
                const next = remaining[selectedIndex] ?? remaining[0] ?? null;
                setSelectedDisagreementKey(next ? disagreementKey(next) : null);
            }
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to save adjudication."),
                severity: "error",
            }),
    });

    function selectDisagreement(item: DisagreementItem) {
        setSelectedDisagreementKey(disagreementKey(item));
    }

    function goNextDisagreement() {
        const next = disagreements[selectedIndex + 1];
        if (next) {
            setSelectedDisagreementKey(disagreementKey(next));
        }
    }

    const byLabel = useMemo(() => {
        const results = runQuery.data?.results as
            | { by_label?: Record<string, LabelReliability> }
            | null
            | undefined;
        return results?.by_label ?? null;
    }, [runQuery.data?.results]);

    const metrics = runQuery.data?.metrics ?? null;

    const emptyTitle = !ctx.selectedCorpusId
        ? "Select a corpus"
        : !ctx.selectedCodebookId
          ? "No codebook selected"
          : "No codebook labels";

    const emptyDescription = !ctx.selectedCorpusId
        ? "Choose a corpus in the context bar, then ensure a codebook with labels and overlapping annotations exist."
        : !ctx.selectedCodebookId
          ? "Create a codebook and add labels, then annotate the same units with at least two annotators."
          : "Add labels to the selected codebook, then complete overlapping annotation tasks before measuring agreement.";

    const emptyActionPath = !ctx.selectedCodebookId
        ? `/research/${ctx.projectId}/codebook`
        : `/research/${ctx.projectId}/annotation`;

    const emptyActionLabel = !ctx.selectedCodebookId ? "Create codebook" : "Go to annotation";

    return (
        <Stack spacing={2}>
            <PageTabs
                value={tab}
                onChange={setTab}
                tabs={[
                    { value: "overview", label: "Overview" },
                    { value: "agreement", label: "Agreement" },
                    { value: "disagreements", label: "Disagreements" },
                    { value: "by_coder", label: "By Coder" },
                    { value: "methods", label: "Methods & Provenance", disabled: !ctx.selectedCorpusId },
                ]}
                ariaLabel="Reliability workflow"
            />

            {tab === "overview" ? (
            <>
            <SectionCard
                title="Inter-annotator reliability"
                description="Compute Cohen's κ, Fleiss' κ, and Krippendorff's α with bootstrap confidence intervals for the selected codebook."
            >
                {!reliabilityReady ? (
                    <EmptyState
                        icon={<CodebookIcon fontSize="large" />}
                        title={emptyTitle}
                        description={`${emptyDescription} ${RELIABILITY_ANNOTATOR_HINT}`}
                        action={
                            <Button variant="contained" onClick={() => navigate(emptyActionPath)}>
                                {emptyActionLabel}
                            </Button>
                        }
                    />
                ) : (
                    <>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                            Codebook {ctx.selectedCodebook?.name} · v
                            {ctx.selectedCodebook?.version}
                            {ctx.selectedCodebook?.is_frozen ? " · frozen" : ""}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            Labels included: {ctx.labels.map((l) => l.name).join(", ") || "None"}.
                        </Typography>
                        <Stack
                            direction={{ xs: "column", sm: "row" }}
                            spacing={2}
                            flexWrap="wrap"
                            sx={{ mb: 2 }}
                        >
                            <TextField
                                select
                                size="small"
                                label="Annotation scope"
                                value={selectedCampaignId}
                                onChange={(e) => setSelectedCampaignId(e.target.value)}
                                sx={{ minWidth: 240 }}
                                disabled={campaignsQuery.isLoading}
                            >
                                <MenuItem value="">Legacy / unscoped annotations</MenuItem>
                                {campaigns.map((campaign) => (
                                    <MenuItem key={campaign.id} value={campaign.id}>
                                        {campaign.name}
                                    </MenuItem>
                                ))}
                            </TextField>
                        </Stack>
                        <AdvancedSettings title="Bootstrap & sampling" description="Confidence intervals and reproducibility controls.">
                            <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                                <TextField
                                    size="small"
                                    type="number"
                                    label="Bootstrap samples"
                                    value={bootstrapSamples}
                                    onChange={(e) =>
                                        setBootstrapSamples(Number(e.target.value) || 2000)
                                    }
                                    inputProps={{ min: 100, max: 10000, step: 100 }}
                                    sx={{ width: 160 }}
                                />
                                <TextField
                                    size="small"
                                    type="number"
                                    label="CI level"
                                    value={confidenceLevel}
                                    onChange={(e) =>
                                        setConfidenceLevel(Number(e.target.value) || 0.95)
                                    }
                                    inputProps={{ min: 0.5, max: 0.99, step: 0.01 }}
                                    sx={{ width: 120 }}
                                />
                                <TextField
                                    size="small"
                                    type="number"
                                    label="Random seed"
                                    value={randomSeed}
                                    onChange={(e) => setRandomSeed(e.target.value)}
                                    placeholder="Optional"
                                    sx={{ width: 140 }}
                                />
                            </Stack>
                        </AdvancedSettings>
                        <DisabledWithReason
                            reason={computeReliabilityDisabledReason({
                                corpusId: ctx.selectedCorpusId,
                                codebookId: ctx.selectedCodebookId,
                                labelCount: ctx.labels.length,
                                pending: reliabilityMutation.isPending,
                            })}
                        >
                            <Button
                                variant="contained"
                                startIcon={<RunIcon />}
                                onClick={() => reliabilityMutation.mutate()}
                                disabled={
                                    Boolean(
                                        computeReliabilityDisabledReason({
                                            corpusId: ctx.selectedCorpusId,
                                            codebookId: ctx.selectedCodebookId,
                                            labelCount: ctx.labels.length,
                                            pending: reliabilityMutation.isPending,
                                        })
                                    )
                                }
                                sx={{ mt: 2 }}
                            >
                                Compute reliability
                            </Button>
                        </DisabledWithReason>
                    </>
                )}
            </SectionCard>

            {runId && metrics ? (
                <SectionCard
                    title="Latest summary"
                    description="Primary chance-corrected coefficients with heuristic readings, coverage (coders / units), and caveats."
                >
                    <ReliabilityPrimaryMetrics metrics={metrics} byLabel={byLabel} />
                    <Stack direction="row" spacing={1} sx={{ mt: 2 }} flexWrap="wrap">
                        <Button size="small" variant="outlined" onClick={() => setTab("agreement")}>
                            View agreement & CIs
                        </Button>
                        <Button size="small" variant="outlined" onClick={() => setTab("by_coder")}>
                            View by coder
                        </Button>
                        <Button size="small" variant="outlined" onClick={() => setTab("disagreements")}>
                            Resolve disagreements
                        </Button>
                        <Button size="small" variant="text" onClick={() => setTab("methods")}>
                            Methods & provenance
                        </Button>
                    </Stack>
                </SectionCard>
            ) : null}
            </>
            ) : null}

            {tab === "agreement" ? (
            <>
            {runId ? (
                <SectionCard
                    title="Reliability results"
                    description="Mean coefficients, per-label CIs, n_coders / pairable units, and heuristic readings (not universal cutoffs)."
                >
                    <QueryBoundary
                        isLoading={runQuery.isLoading && !runQuery.data}
                        isError={runQuery.isError}
                        error={runQuery.error}
                        onRetry={() => void runQuery.refetch()}
                    >
                        {runQuery.data ? (
                            <Stack spacing={2}>
                                <RunStatusPanel
                                    dense
                                    title="Reliability run"
                                    status={runQuery.data.status}
                                    runId={runQuery.data.id}
                                    stage={runQuery.data.progress_stage}
                                    startedAt={runQuery.data.started_at}
                                    completedAt={runQuery.data.completed_at}
                                    createdAt={runQuery.data.created_at}
                                    errorMessage={runQuery.data.error_message}
                                    actions={<ProvenanceDrawer run={runQuery.data} />}
                                />

                                {metrics ? (
                                    <ReliabilityPrimaryMetrics metrics={metrics} byLabel={byLabel} />
                                ) : isActiveRunStatus(runQuery.data.status) ? (
                                    <Typography color="text.secondary">
                                        Computing reliability metrics…
                                    </Typography>
                                ) : null}

                                <ScientificWarnings
                                    title="Scientific warnings"
                                    warnings={collectScientificWarnings(
                                        metrics,
                                        runQuery.data.results
                                    )}
                                />

                                {byLabel
                                    ? <>
                                          <LabelReliabilitySummaryTable byLabel={byLabel} />
                                          <ReliabilityComparisonChart
                                              items={reliabilityChartItems(byLabel)}
                                              onSelect={setSelectedReliabilityLabel}
                                          />
                                          {selectedReliabilityLabel ? <Button size="small" onClick={() => setSelectedReliabilityLabel(null)}>Show all labels</Button> : null}
                                      </>
                                    : null}
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : (
                <SectionCard title="Agreement" description="Compute reliability from Overview to inspect per-label metrics.">
                    <EmptyState
                        icon={<CodebookIcon fontSize="large" />}
                        title="No reliability run yet"
                        description="Open Overview to compute agreement for the selected codebook."
                        action={
                            <Button variant="contained" onClick={() => setTab("overview")}>
                                Go to Overview
                            </Button>
                        }
                    />
                </SectionCard>
            )}
            </>
            ) : null}

            {tab === "by_coder" ? (
                <SectionCard
                    title="By coder"
                    description="Coder-pair matrices, pairwise Cohen's κ, and per-label primary metrics with CIs from the latest run."
                >
                    {byLabel ? (
                        <Stack spacing={2}>
                            {Object.entries(byLabel)
                                .filter(
                                    ([labelName]) =>
                                        !selectedReliabilityLabel ||
                                        labelName === selectedReliabilityLabel
                                )
                                .map(([labelName, row]) => (
                                    <LabelReliabilityCard
                                        key={labelName}
                                        labelName={labelName}
                                        row={row}
                                    />
                                ))}
                        </Stack>
                    ) : (
                        <EmptyState
                            icon={<CodebookIcon fontSize="large" />}
                            title="No coder pairwise results"
                            description="Compute reliability on Overview first, then inspect coder-pair agreement here."
                            action={
                                <Button variant="contained" onClick={() => setTab("overview")}>
                                    Go to Overview
                                </Button>
                            }
                        />
                    )}
                </SectionCard>
            ) : null}

            {tab === "disagreements" ? (
            <SectionCard
                title="Disagreement adjudication"
                description="Inspect units where coders differ, set a gold decision, and preserve original annotations for audit."
            >
                {!ctx.selectedCorpusId || !ctx.selectedCodebookId ? (
                    <EmptyState
                        icon={<AdjudicateIcon fontSize="large" />}
                        title={
                            !ctx.selectedCorpusId
                                ? "Select a corpus"
                                : "No codebook selected"
                        }
                        description={
                            !ctx.selectedCorpusId
                                ? "Choose a corpus, then select a codebook with overlapping annotations to adjudicate."
                                : "Select or create a codebook, complete multi-coder annotation, then return here to resolve disagreements."
                        }
                        action={
                            <Button
                                variant="contained"
                                onClick={() =>
                                    navigate(
                                        !ctx.selectedCodebookId
                                            ? `/research/${ctx.projectId}/codebook`
                                            : `/research/${ctx.projectId}/annotation`
                                    )
                                }
                            >
                                {!ctx.selectedCodebookId
                                    ? "Go to codebook"
                                    : "Go to annotation"}
                            </Button>
                        }
                    />
                ) : (
                    <QueryBoundary
                        isLoading={disagreementsQuery.isLoading}
                        isError={disagreementsQuery.isError}
                        error={disagreementsQuery.error}
                        onRetry={() => void disagreementsQuery.refetch()}
                    >
                        {disagreements.length === 0 ? (
                            <EmptyState
                                icon={<AdjudicateIcon fontSize="large" />}
                                title="No open disagreements"
                                description="All overlapping annotations agree, or remaining disagreements have already been adjudicated. Add more overlap in annotation if you need further reliability work."
                                action={
                                    <Button
                                        variant="outlined"
                                        onClick={() =>
                                            navigate(`/research/${ctx.projectId}/annotation`)
                                        }
                                    >
                                        Go to annotation
                                    </Button>
                                }
                            />
                        ) : (
                            <Box
                                sx={{
                                    display: "grid",
                                    gap: 2,
                                    gridTemplateColumns: {
                                        xs: "1fr",
                                        md: "minmax(220px, 0.7fr) minmax(0, 2fr)",
                                    },
                                }}
                            >
                                <List dense sx={{ maxHeight: 520, overflow: "auto" }}>
                                    {disagreements.map((item) => {
                                        const key = disagreementKey(item);
                                        return (
                                            <ListItemButton
                                                key={key}
                                                selected={key === effectiveDisagreementKey}
                                                onClick={() => selectDisagreement(item)}
                                            >
                                                <ListItemText
                                                    primary={item.label_name}
                                                    secondary={`${item.text_unit_id.slice(0, 8)}… · ${Object.keys(item.judgements).length} coders`}
                                                />
                                            </ListItemButton>
                                        );
                                    })}
                                </List>

                                <Stack spacing={2}>
                                    {selectedDisagreement ? (
                                        <>
                                            <Box>
                                                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                                    Source passage
                                                </Typography>
                                                <QueryBoundary
                                                    isLoading={
                                                        unitContextQuery.isLoading &&
                                                        !unitContextQuery.data
                                                    }
                                                    isError={unitContextQuery.isError}
                                                    error={unitContextQuery.error}
                                                    onRetry={() =>
                                                        void unitContextQuery.refetch()
                                                    }
                                                >
                                                    {unitContextQuery.data ? (
                                                        <Stack spacing={1}>
                                                            {unitContextQuery.data.before.map(
                                                                (unit) => (
                                                                    <Typography
                                                                        key={unit.id}
                                                                        variant="body2"
                                                                        color="text.secondary"
                                                                    >
                                                                        {unit.text}
                                                                    </Typography>
                                                                )
                                                            )}
                                                            <Typography
                                                                variant="body1"
                                                                sx={{
                                                                    p: 1.5,
                                                                    borderRadius: 1,
                                                                    bgcolor: "action.selected",
                                                                }}
                                                            >
                                                                {unitContextQuery.data.unit.text}
                                                            </Typography>
                                                            {unitContextQuery.data.after.map(
                                                                (unit) => (
                                                                    <Typography
                                                                        key={unit.id}
                                                                        variant="body2"
                                                                        color="text.secondary"
                                                                    >
                                                                        {unit.text}
                                                                    </Typography>
                                                                )
                                                            )}
                                                            {unitContextQuery.data.document ? (
                                                                <Typography
                                                                    variant="caption"
                                                                    color="text.secondary"
                                                                >
                                                                    {[
                                                                        unitContextQuery.data
                                                                            .document.title,
                                                                        unitContextQuery.data
                                                                            .document
                                                                            .organization,
                                                                        unitContextQuery.data
                                                                            .document
                                                                            .publication_year,
                                                                    ]
                                                                        .filter(Boolean)
                                                                        .join(" · ")}
                                                                </Typography>
                                                            ) : null}
                                                        </Stack>
                                                    ) : null}
                                                </QueryBoundary>
                                            </Box>

                                            <AdjudicationDecisionForm
                                                key={effectiveDisagreementKey ?? "none"}
                                                selectedDisagreement={selectedDisagreement}
                                                selectedLabel={selectedLabel}
                                                selectedIndex={selectedIndex}
                                                disagreementCount={disagreements.length}
                                                savePending={saveAdjudicationMutation.isPending}
                                                onSave={({ finalValue, comment, advance }) =>
                                                    saveAdjudicationMutation.mutate({
                                                        advance,
                                                        finalValue,
                                                        comment,
                                                        disagreement: selectedDisagreement,
                                                    })
                                                }
                                                onNext={goNextDisagreement}
                                            />
                                        </>
                                    ) : (
                                        <Typography color="text.secondary">
                                            Select a disagreement to adjudicate.
                                        </Typography>
                                    )}
                                </Stack>
                            </Box>
                        )}
                    </QueryBoundary>
                )}
            </SectionCard>
            ) : null}

            {tab === "methods" && ctx.selectedCorpusId ? (
                <>
                <SectionCard
                    title="Methods & provenance"
                    description="Bootstrap CI settings and raw reliability payloads for reporting. Defaults also appear on Overview when computing."
                >
                    <AdvancedSettings title="Bootstrap & sampling" defaultExpanded>
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                            <TextField
                                size="small"
                                type="number"
                                label="Bootstrap samples"
                                value={bootstrapSamples}
                                onChange={(e) =>
                                    setBootstrapSamples(Number(e.target.value) || 2000)
                                }
                                inputProps={{ min: 100, max: 10000, step: 100 }}
                                sx={{ width: 160 }}
                            />
                            <TextField
                                size="small"
                                type="number"
                                label="CI level"
                                value={confidenceLevel}
                                onChange={(e) =>
                                    setConfidenceLevel(Number(e.target.value) || 0.95)
                                }
                                inputProps={{ min: 0.5, max: 0.99, step: 0.01 }}
                                sx={{ width: 120 }}
                            />
                            <TextField
                                size="small"
                                type="number"
                                label="Random seed"
                                value={randomSeed}
                                onChange={(e) => setRandomSeed(e.target.value)}
                                placeholder="Optional"
                                sx={{ width: 140 }}
                            />
                        </Stack>
                    </AdvancedSettings>
                    {runQuery.data ? (
                        <Box sx={{ mt: 2 }}>
                            <ProvenancePanel run={runQuery.data} showRaw />
                            <ResultsInspector
                                title="raw reliability data"
                                data={{
                                    metrics: runQuery.data.metrics,
                                    results: runQuery.data.results,
                                }}
                            />
                        </Box>
                    ) : null}
                </SectionCard>
                <SectionCard
                    title="Adjudication provenance"
                    description="Gold decisions recorded for this corpus. Original coder judgments are preserved."
                >
                    <QueryBoundary
                        isLoading={adjudicationsQuery.isLoading}
                        isError={adjudicationsQuery.isError}
                        error={adjudicationsQuery.error}
                        onRetry={() => void adjudicationsQuery.refetch()}
                    >
                        {(adjudicationsQuery.data?.length ?? 0) === 0 ? (
                            <Typography variant="body2" color="text.secondary">
                                No adjudications saved yet.
                            </Typography>
                        ) : (
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Unit</TableCell>
                                        <TableCell>Label</TableCell>
                                        <TableCell>Codebook</TableCell>
                                        <TableCell>Adjudicator</TableCell>
                                        <TableCell>Final value</TableCell>
                                        <TableCell>Date</TableCell>
                                        <TableCell>Comment</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {adjudicationsQuery.data?.map((row) => (
                                        <TableRow key={row.id}>
                                            <TableCell>
                                                {row.text_unit_id.slice(0, 8)}…
                                            </TableCell>
                                            <TableCell>
                                                {ctx.labels.find((l) => l.id === row.label_id)
                                                    ?.name ?? row.label_id.slice(0, 8)}
                                            </TableCell>
                                            <TableCell>
                                                {row.codebook_version
                                                    ? `v${row.codebook_version}`
                                                    : "Legacy record"}
                                            </TableCell>
                                            <TableCell>
                                                {row.adjudicator_id.slice(0, 8)}…
                                            </TableCell>
                                            <TableCell>{row.final_value}</TableCell>
                                            <TableCell>
                                                {new Date(row.created_at).toLocaleString()}
                                            </TableCell>
                                            <TableCell>{row.comment ?? "—"}</TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        )}
                    </QueryBoundary>
                </SectionCard>
                </>
            ) : null}
        </Stack>
    );
}
