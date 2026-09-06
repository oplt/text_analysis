import { useEffect, useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    FormControlLabel,
    IconButton,
    List,
    ListItemButton,
    ListItemText,
    Radio,
    RadioGroup,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Tooltip,
    Typography,
} from "@mui/material";
import {
    HelpOutline as HelpIcon,
    PlaylistAddCheck as CodebookIcon,
    PlayArrow as RunIcon,
    RateReview as AdjudicateIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    computeReliability,
    getRun,
    getTextUnitContext,
    listAdjudications,
    listDisagreements,
    saveAdjudication,
    type DisagreementItem,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { MatrixHeatmap, MetricCards, ResultsInspector } from "../components/ResearchCharts";
import { RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import { activeRunRefetchInterval } from "../runPolling";
import type { AnnotationLabel } from "../types";

type LabelDecision = "yes" | "no" | "uncertain";

type CohensKappa = {
    observed_agreement?: number | null;
    expected_agreement?: number | null;
    kappa?: number | null;
    sample_size?: number | null;
};

type KrippendorffAlpha = {
    alpha?: number | null;
    n_units?: number | null;
    n_coders?: number | null;
    missingness?: number | null;
};

type CoderPairAgreement = {
    coders?: string[];
    matrix?: Record<string, Record<string, { agreement?: number | null; n_common_units?: number }>>;
};

type LabelReliability = {
    label_id?: string;
    n_coders?: number;
    cohens_kappa?: CohensKappa | null;
    krippendorff_alpha?: KrippendorffAlpha | null;
    coder_pair_agreement?: CoderPairAgreement | null;
    disagreement_count?: number;
    disagreements?: unknown;
};

const STAT_HELP: Record<string, string> = {
    kappa:
        "Cohen's kappa measures chance-corrected agreement between exactly two coders on the same units. Prefer it for two-rater designs with complete overlap.",
    alpha:
        "Krippendorff's alpha supports any number of coders and missing values. Prefer it when there are more than two coders or incomplete annotation overlap.",
    observed:
        "Observed agreement is the raw proportion of units where coders assign the same value, before correcting for chance.",
    expected:
        "Expected agreement is the agreement rate predicted by chance given each coder's category marginals. Used to compute kappa.",
    sample:
        "Sample size is the number of overlapping units used for the two-coder kappa calculation.",
    missingness:
        "Missingness is the share of coder×unit cells without a value. Higher missingness reduces the effective reliability sample for alpha.",
    n_coders:
        "Number of distinct annotators who contributed values for this label in the selected codebook version.",
};

function formatMetric(value: number | null | undefined, digits = 3): string {
    if (value == null || Number.isNaN(value)) return "—";
    return value.toFixed(digits);
}

function StatLabel({ label, helpKey }: { label: string; helpKey: keyof typeof STAT_HELP }) {
    return (
        <Stack direction="row" spacing={0.5} alignItems="center" component="span">
            <span>{label}</span>
            <Tooltip title={STAT_HELP[helpKey]} arrow>
                <IconButton size="small" aria-label={`About ${label}`} sx={{ p: 0.25 }}>
                    <HelpIcon fontSize="inherit" />
                </IconButton>
            </Tooltip>
        </Stack>
    );
}

function pairMatrixValues(pair: CoderPairAgreement | null | undefined): {
    coders: string[];
    values: number[][];
} {
    const coders = pair?.coders ?? [];
    const matrix = pair?.matrix ?? {};
    const values = coders.map((row) =>
        coders.map((col) => {
            const cell = matrix[row]?.[col];
            return typeof cell?.agreement === "number" ? cell.agreement : 0;
        })
    );
    return { coders, values };
}

function LabelReliabilityCard({
    labelName,
    row,
}: {
    labelName: string;
    row: LabelReliability;
}) {
    const kappa = row.cohens_kappa ?? null;
    const alpha = row.krippendorff_alpha ?? null;
    const { coders, values } = pairMatrixValues(row.coder_pair_agreement);

    return (
        <Box sx={{ p: 2, borderRadius: 1, bgcolor: "action.hover" }}>
            <Stack spacing={1.5}>
                <Typography variant="subtitle1">{labelName}</Typography>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell>
                                <StatLabel label="Cohen's κ" helpKey="kappa" />
                            </TableCell>
                            <TableCell>
                                <StatLabel label="Krippendorff's α" helpKey="alpha" />
                            </TableCell>
                            <TableCell>
                                <StatLabel label="Observed agr." helpKey="observed" />
                            </TableCell>
                            <TableCell>
                                <StatLabel label="Expected agr." helpKey="expected" />
                            </TableCell>
                            <TableCell>
                                <StatLabel label="Sample size" helpKey="sample" />
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
                            <TableCell>{formatMetric(kappa?.kappa)}</TableCell>
                            <TableCell>{formatMetric(alpha?.alpha)}</TableCell>
                            <TableCell>{formatMetric(kappa?.observed_agreement)}</TableCell>
                            <TableCell>{formatMetric(kappa?.expected_agreement)}</TableCell>
                            <TableCell>
                                {kappa?.sample_size != null ? String(kappa.sample_size) : "—"}
                            </TableCell>
                            <TableCell>{formatMetric(alpha?.missingness)}</TableCell>
                            <TableCell>
                                {row.n_coders != null ? String(row.n_coders) : "—"}
                            </TableCell>
                        </TableRow>
                    </TableBody>
                </Table>
                {row.disagreement_count != null ? (
                    <Typography variant="caption" color="text.secondary">
                        Disagreements: {row.disagreement_count}
                    </Typography>
                ) : null}
                {coders.length > 0 ? (
                    <Box>
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>
                            Coder-to-coder agreement
                        </Typography>
                        <MatrixHeatmap
                            rowLabels={coders}
                            colLabels={coders}
                            values={values}
                            formatCell={(v) => formatMetric(v, 2)}
                        />
                    </Box>
                ) : null}
            </Stack>
        </Box>
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
    const [finalValue, setFinalValue] = useState<LabelDecision | "">("");
    const [adjudicationComment, setAdjudicationComment] = useState("");

    const reliabilityReady =
        Boolean(ctx.selectedCorpusId) &&
        Boolean(ctx.selectedCodebookId) &&
        ctx.labels.length > 0;

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: () => getRun(runId!),
        enabled: Boolean(runId),
        refetchInterval: activeRunRefetchInterval,
    });

    const disagreementsQuery = useQuery({
        queryKey: queryKeys.textResearch.disagreements(
            ctx.selectedCorpusId,
            ctx.selectedCodebookId
        ),
        queryFn: () => listDisagreements(ctx.selectedCorpusId, ctx.selectedCodebookId),
        enabled: Boolean(ctx.selectedCorpusId && ctx.selectedCodebookId),
    });

    const adjudicationsQuery = useQuery({
        queryKey: queryKeys.textResearch.adjudications(ctx.selectedCorpusId),
        queryFn: () => listAdjudications(ctx.selectedCorpusId),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const disagreements = disagreementsQuery.data ?? [];

    const selectedDisagreement = useMemo(() => {
        if (!selectedDisagreementKey) return null;
        return (
            disagreements.find(
                (item) => `${item.text_unit_id}:${item.label_id}` === selectedDisagreementKey
            ) ?? null
        );
    }, [disagreements, selectedDisagreementKey]);

    const selectedIndex = selectedDisagreement
        ? disagreements.findIndex(
              (item) =>
                  item.text_unit_id === selectedDisagreement.text_unit_id &&
                  item.label_id === selectedDisagreement.label_id
          )
        : -1;

    useEffect(() => {
        if (!selectedDisagreementKey && disagreements[0]) {
            setSelectedDisagreementKey(
                `${disagreements[0].text_unit_id}:${disagreements[0].label_id}`
            );
        }
    }, [disagreements, selectedDisagreementKey]);

    useEffect(() => {
        setFinalValue("");
        setAdjudicationComment("");
    }, [selectedDisagreementKey]);

    const unitContextQuery = useQuery({
        queryKey: ["text-research", "unit-context", selectedDisagreement?.text_unit_id],
        queryFn: () => getTextUnitContext(selectedDisagreement!.text_unit_id, 2),
        enabled: Boolean(selectedDisagreement?.text_unit_id),
    });

    const selectedLabel = useMemo(() => {
        if (!selectedDisagreement) return null;
        return ctx.labels.find((label) => label.id === selectedDisagreement.label_id) ?? null;
    }, [ctx.labels, selectedDisagreement]);

    const reliabilityMutation = useMutation({
        mutationFn: () =>
            computeReliability(ctx.selectedCorpusId, {
                codebook_id: ctx.selectedCodebookId,
                label_ids: ctx.labels.map((l) => l.id),
            }),
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
        mutationFn: async (options: { advance: boolean }) => {
            if (!selectedDisagreement || !finalValue) {
                throw new Error("Select a disagreement and final value.");
            }
            await saveAdjudication({
                text_unit_id: selectedDisagreement.text_unit_id,
                label_id: selectedDisagreement.label_id,
                final_value: finalValue,
                comment: adjudicationComment.trim() || undefined,
            });
            return { ...options, saved: selectedDisagreement };
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
                setSelectedDisagreementKey(
                    next ? `${next.text_unit_id}:${next.label_id}` : null
                );
            }
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to save adjudication."),
                severity: "error",
            }),
    });

    function selectDisagreement(item: DisagreementItem) {
        setSelectedDisagreementKey(`${item.text_unit_id}:${item.label_id}`);
    }

    function goNextDisagreement() {
        const next = disagreements[selectedIndex + 1];
        if (next) {
            setSelectedDisagreementKey(`${next.text_unit_id}:${next.label_id}`);
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
            <SectionCard
                title="Inter-annotator reliability"
                description="Compute agreement metrics for the selected codebook labels."
            >
                {!reliabilityReady ? (
                    <EmptyState
                        icon={<CodebookIcon fontSize="large" />}
                        title={emptyTitle}
                        description={emptyDescription}
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
                            Labels included: {ctx.labels.map((l) => l.name).join(", ")}
                        </Typography>
                        <Button
                            variant="contained"
                            startIcon={<RunIcon />}
                            onClick={() => reliabilityMutation.mutate()}
                            disabled={reliabilityMutation.isPending}
                        >
                            Compute reliability
                        </Button>
                    </>
                )}
            </SectionCard>

            {runId ? (
                <SectionCard
                    title="Reliability results"
                    description="Per-label agreement statistics and coder pair matrices."
                >
                    <QueryBoundary
                        isLoading={runQuery.isLoading && !runQuery.data}
                        isError={runQuery.isError}
                        error={runQuery.error}
                        onRetry={() => void runQuery.refetch()}
                    >
                        {runQuery.data ? (
                            <Stack spacing={2}>
                                <Typography variant="body2">
                                    Run {runQuery.data.id} —{" "}
                                    <RunStatusChip status={runQuery.data.status} />
                                </Typography>
                                {runQuery.data.error_message ? (
                                    <Alert severity="error">{runQuery.data.error_message}</Alert>
                                ) : null}

                                {metrics ? (
                                    <MetricCards
                                        items={[
                                            {
                                                label: "Mean Krippendorff α",
                                                value: formatMetric(
                                                    metrics.mean_krippendorff_alpha as
                                                        | number
                                                        | null
                                                        | undefined
                                                ),
                                            },
                                            {
                                                label: "Mean Cohen's κ",
                                                value: formatMetric(
                                                    metrics.mean_cohens_kappa as
                                                        | number
                                                        | null
                                                        | undefined
                                                ),
                                            },
                                            {
                                                label: "Labels evaluated",
                                                value:
                                                    metrics.labels_evaluated != null
                                                        ? String(metrics.labels_evaluated)
                                                        : "—",
                                            },
                                        ]}
                                    />
                                ) : runQuery.data.status === "running" ||
                                  runQuery.data.status === "pending" ? (
                                    <Typography color="text.secondary">
                                        Computing reliability metrics…
                                    </Typography>
                                ) : null}

                                {byLabel
                                    ? Object.entries(byLabel).map(([labelName, row]) => (
                                          <LabelReliabilityCard
                                              key={labelName}
                                              labelName={labelName}
                                              row={row}
                                          />
                                      ))
                                    : null}

                                <ResultsInspector
                                    title="raw reliability data"
                                    data={{
                                        metrics: runQuery.data.metrics,
                                        results: runQuery.data.results,
                                    }}
                                />
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}

            <SectionCard
                title="Disagreement adjudication"
                description="Resolve coder disagreements into a gold label without changing original annotations."
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
                                    gridTemplateColumns: { xs: "1fr", md: "280px 1fr" },
                                }}
                            >
                                <List dense sx={{ maxHeight: 520, overflow: "auto" }}>
                                    {disagreements.map((item) => {
                                        const key = `${item.text_unit_id}:${item.label_id}`;
                                        return (
                                            <ListItemButton
                                                key={key}
                                                selected={key === selectedDisagreementKey}
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
                                                    Coder judgments
                                                </Typography>
                                                <Stack spacing={0.5}>
                                                    {Object.entries(
                                                        selectedDisagreement.judgements
                                                    ).map(([coderId, value]) => (
                                                        <Typography key={coderId} variant="body2">
                                                            {coderId.slice(0, 8)}… → {value}
                                                        </Typography>
                                                    ))}
                                                </Stack>
                                            </Box>

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

                                            <Box>
                                                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                                    Codebook definition
                                                </Typography>
                                                {selectedLabel ? (
                                                    <LabelDefinition label={selectedLabel} />
                                                ) : (
                                                    <Typography
                                                        variant="body2"
                                                        color="text.secondary"
                                                    >
                                                        Label {selectedDisagreement.label_name}{" "}
                                                        (definition not loaded in context).
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
                                                    onChange={(e) =>
                                                        setFinalValue(
                                                            e.target.value as LabelDecision
                                                        )
                                                    }
                                                >
                                                    <FormControlLabel
                                                        value="yes"
                                                        control={<Radio size="small" />}
                                                        label="Yes"
                                                    />
                                                    <FormControlLabel
                                                        value="no"
                                                        control={<Radio size="small" />}
                                                        label="No"
                                                    />
                                                    <FormControlLabel
                                                        value="uncertain"
                                                        control={<Radio size="small" />}
                                                        label="Uncertain"
                                                    />
                                                </RadioGroup>
                                                <TextField
                                                    label="Comment"
                                                    value={adjudicationComment}
                                                    onChange={(e) =>
                                                        setAdjudicationComment(e.target.value)
                                                    }
                                                    fullWidth
                                                    multiline
                                                    minRows={2}
                                                    sx={{ mt: 1 }}
                                                />
                                            </Box>

                                            <Stack direction="row" spacing={1}>
                                                <Button
                                                    variant="contained"
                                                    disabled={
                                                        !finalValue ||
                                                        saveAdjudicationMutation.isPending
                                                    }
                                                    onClick={() =>
                                                        saveAdjudicationMutation.mutate({
                                                            advance: false,
                                                        })
                                                    }
                                                >
                                                    Save
                                                </Button>
                                                <Button
                                                    variant="contained"
                                                    disabled={
                                                        !finalValue ||
                                                        saveAdjudicationMutation.isPending
                                                    }
                                                    onClick={() =>
                                                        saveAdjudicationMutation.mutate({
                                                            advance: true,
                                                        })
                                                    }
                                                >
                                                    Save + Next disagreement
                                                </Button>
                                                <Button
                                                    variant="outlined"
                                                    disabled={
                                                        selectedIndex < 0 ||
                                                        selectedIndex >=
                                                            disagreements.length - 1
                                                    }
                                                    onClick={goNextDisagreement}
                                                >
                                                    Next disagreement
                                                </Button>
                                            </Stack>
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

            {ctx.selectedCorpusId ? (
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
            ) : null}
        </Stack>
    );
}
