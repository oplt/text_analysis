import { useState } from "react";
import { Alert, Button, MenuItem, Stack, TextField, Typography } from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
    getRun,
    listDatasetSnapshots,
    runRobustnessSweep,
} from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    MetricCards,
    ResultsInspector,
    SimpleLineLikeBars,
    type RankedItem,
} from "../components/ResearchCharts";
import { RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import { activeRunRefetchInterval, isActiveRunStatus } from "../runPolling";

type EvalRow = Record<string, unknown>;

function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
}

function asRows(value: unknown): EvalRow[] {
    return Array.isArray(value) ? (value as EvalRow[]) : [];
}

function rowStatus(row: EvalRow): "ok" | "not_evaluable" | "legacy" {
    if (row.status === "not_evaluable") return "not_evaluable";
    if (row.status === "ok") return "ok";
    return "legacy";
}

function rowF1(row: EvalRow): number | null {
    const value = row.macro_f1;
    return typeof value === "number" ? value : null;
}

function formatF1(value: number | null | undefined): string {
    if (value == null || Number.isNaN(value)) return "—";
    return value.toFixed(3);
}

function chartableRows(
    rows: EvalRow[],
    labelFn: (row: EvalRow, index: number) => string
): { items: RankedItem[]; blocked: EvalRow[] } {
    const items: RankedItem[] = [];
    const blocked: EvalRow[] = [];
    rows.forEach((row, index) => {
        if (rowStatus(row) === "not_evaluable") {
            blocked.push(row);
            return;
        }
        const f1 = rowF1(row);
        if (f1 == null) {
            if (rowStatus(row) === "legacy") return;
            blocked.push(row);
            return;
        }
        items.push({ label: labelFn(row, index), value: f1 });
    });
    return { items, blocked };
}

function NotEvaluableAlerts({ rows }: { rows: EvalRow[] }) {
    if (!rows.length) return null;
    return (
        <Stack spacing={1}>
            {rows.map((row, index) => (
                <Alert key={index} severity="warning">
                    {typeof row.reason === "string" && row.reason
                        ? row.reason
                        : "This check was not evaluable."}
                </Alert>
            ))}
        </Stack>
    );
}

function SweepSection({
    title,
    description,
    items,
    blocked,
}: {
    title: string;
    description?: string;
    items: RankedItem[];
    blocked: EvalRow[];
}) {
    return (
        <Stack spacing={1}>
            <Typography variant="subtitle1">{title}</Typography>
            {description ? (
                <Typography variant="body2" color="text.secondary">
                    {description}
                </Typography>
            ) : null}
            <NotEvaluableAlerts rows={blocked} />
            {items.length ? <SimpleLineLikeBars items={items} /> : null}
            {!items.length && !blocked.length ? (
                <Typography variant="body2" color="text.secondary">
                    No results for this check.
                </Typography>
            ) : null}
        </Stack>
    );
}

export default function RobustnessView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const [snapshotId, setSnapshotId] = useState("");
    const [groupField, setGroupField] = useState("organization");
    const [runId, setRunId] = useState<string | null>(null);

    const snapshotsQuery = useQuery({
        queryKey: queryKeys.textResearch.datasetSnapshots(ctx.projectId, ctx.selectedCorpusId),
        queryFn: () => listDatasetSnapshots(ctx.projectId, ctx.selectedCorpusId),
        enabled: Boolean(ctx.projectId && ctx.selectedCorpusId),
    });

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: () => getRun(runId!),
        enabled: Boolean(runId),
        refetchInterval: activeRunRefetchInterval,
    });

    const sweepMutation = useMutation({
        mutationFn: () =>
            runRobustnessSweep({
                snapshot_id: snapshotId,
                group_field: groupField || "organization",
                run_async: true,
            }),
        onSuccess: (run) => {
            setRunId(run.id);
            showToast({ message: "Robustness sweep started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Robustness sweep failed."),
                severity: "error",
            }),
    });

    const results = asRecord(runQuery.data?.results);
    const seedStability = asRecord(results?.seed_stability);
    const seedRuns = asRows(seedStability?.runs);
    const seedsChart = chartableRows(seedRuns, (row) => `Seed ${String(row.seed ?? "?")}`);

    const groupCv = asRecord(results?.group_cross_validation);
    const foldRows = asRows(groupCv?.folds);
    const foldsChart = chartableRows(foldRows, (row, index) => `Fold ${String(row.fold ?? index)}`);

    const preprocessingRows = asRows(results?.preprocessing_sensitivity);
    const preprocessingChart = chartableRows(
        preprocessingRows,
        (row) => String(row.variant ?? "variant")
    );

    const classWeightRows = asRows(results?.class_weight_sensitivity);
    const classWeightChart = chartableRows(
        classWeightRows,
        (row) => String(row.class_weight ?? "none")
    );

    const groupOut = asRecord(results?.leave_one_group_out);
    const groupField = typeof groupOut?.group_field === "string" ? groupOut.group_field : "group";
    const groupRows = asRows(groupOut?.runs);
    const groupChart = chartableRows(
        groupRows,
        (row) => String(row.held_out_value ?? groupField)
    );

    const temporalOut = asRecord(results?.temporal_holdout);
    const temporalRows = asRows(temporalOut?.runs);
    const temporalChart = chartableRows(temporalRows, (row) => {
        if (row.train_period != null) {
            return `Train ${String(row.train_period)} · test ${String(row.test_period ?? "later")}`;
        }
        return "Temporal holdout";
    });

    const meanSeed =
        typeof seedStability?.mean_macro_f1 === "number" ? seedStability.mean_macro_f1 : null;
    const stdSeed =
        typeof seedStability?.std_macro_f1 === "number" ? seedStability.std_macro_f1 : null;
    const meanCv =
        typeof groupCv?.mean_macro_f1 === "number" ? groupCv.mean_macro_f1 : null;
    const stdCv = typeof groupCv?.std_macro_f1 === "number" ? groupCv.std_macro_f1 : null;

    const metricItems = [
        {
            label: "Seed mean F1",
            value: meanSeed == null ? "—" : `${formatF1(meanSeed)}${stdSeed == null ? "" : ` ± ${formatF1(stdSeed)}`}`,
        },
        { label: "Seed min F1", value: formatF1(seedStability?.min_macro_f1 as number | null) },
        { label: "Seed max F1", value: formatF1(seedStability?.max_macro_f1 as number | null) },
        {
            label: "Grouped CV mean F1",
            value: meanCv == null ? "—" : `${formatF1(meanCv)}${stdCv == null ? "" : ` ± ${formatF1(stdCv)}`}`,
        },
    ];

    const completed = runQuery.data?.status === "completed";

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Robustness testing"
                description="Evaluate the frozen training dataset across repeated grouped splits, cross-validation, preprocessing, class weights, organization, and temporal holdouts."
            >
                <QueryBoundary
                    isLoading={snapshotsQuery.isLoading}
                    isError={snapshotsQuery.isError}
                    error={snapshotsQuery.error}
                    onRetry={() => void snapshotsQuery.refetch()}
                >
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                        <TextField
                            select
                            size="small"
                            label="Frozen dataset snapshot"
                            value={snapshotId}
                            onChange={(event) => setSnapshotId(event.target.value)}
                            sx={{ minWidth: 300 }}
                        >
                            <MenuItem value="">Select snapshot</MenuItem>
                            {snapshotsQuery.data?.map((snapshot) => (
                                <MenuItem key={snapshot.id} value={snapshot.id}>
                                    {snapshot.name} · {snapshot.unit_type}
                                </MenuItem>
                            ))}
                        </TextField>
                        <TextField
                            size="small"
                            label="Group field (leave-one-out)"
                            value={groupField}
                            onChange={(event) => setGroupField(event.target.value)}
                            helperText="Any document metadata field, e.g. organization, region, country"
                            sx={{ minWidth: 260 }}
                        />
                        <Button
                            variant="contained"
                            onClick={() => sweepMutation.mutate()}
                            disabled={!snapshotId || sweepMutation.isPending}
                        >
                            Run sweep
                        </Button>
                    </Stack>
                </QueryBoundary>
            </SectionCard>

            {runId ? (
                <SectionCard
                    title="Robustness results"
                    description="Poor or unstable results remain visible. Checks that cannot be evaluated show a reason instead of a chart."
                >
                    <QueryBoundary
                        isLoading={runQuery.isLoading && !runQuery.data}
                        isError={runQuery.isError}
                        error={runQuery.error}
                        onRetry={() => void runQuery.refetch()}
                    >
                        {runQuery.data ? (
                            <Stack spacing={3}>
                                <Typography variant="body2">
                                    Run {runQuery.data.id} —{" "}
                                    <RunStatusChip status={runQuery.data.status} />
                                </Typography>
                                {runQuery.data.error_message ? (
                                    <Alert severity="error">{runQuery.data.error_message}</Alert>
                                ) : null}

                                {completed && results ? (
                                    <>
                                        <Stack spacing={1}>
                                            <Typography variant="subtitle1">Overall stability</Typography>
                                            <Typography variant="body2" color="text.secondary">
                                                Mean macro F1 ± standard deviation from repeated seeds.
                                            </Typography>
                                            <MetricCards items={metricItems} />
                                        </Stack>

                                        <SweepSection
                                            title="Repeated seeds"
                                            description="Seed vs macro F1"
                                            items={seedsChart.items}
                                            blocked={seedsChart.blocked}
                                        />
                                        <SweepSection
                                            title="Grouped cross-validation"
                                            description="Fold vs macro F1"
                                            items={foldsChart.items}
                                            blocked={foldsChart.blocked}
                                        />
                                        <SweepSection
                                            title="Preprocessing sensitivity"
                                            description="Variant vs macro F1"
                                            items={preprocessingChart.items}
                                            blocked={preprocessingChart.blocked}
                                        />
                                        <SweepSection
                                            title="Class weights"
                                            description="Configuration vs macro F1"
                                            items={classWeightChart.items}
                                            blocked={classWeightChart.blocked}
                                        />
                                        <SweepSection
                                            title={`Leave-one-${groupField}-out`}
                                            description={`Held-out ${groupField} value vs macro F1`}
                                            items={groupChart.items}
                                            blocked={groupChart.blocked}
                                        />
                                        <SweepSection
                                            title="Temporal holdout"
                                            description="Train/test period split vs macro F1"
                                            items={temporalChart.items}
                                            blocked={temporalChart.blocked}
                                        />

                                        <ResultsInspector
                                            title="robustness results"
                                            data={{
                                                metrics: runQuery.data.metrics,
                                                results: runQuery.data.results,
                                            }}
                                        />
                                    </>
                                ) : isActiveRunStatus(runQuery.data.status) ? (
                                    <Typography color="text.secondary">
                                        Computing controlled evaluations…
                                    </Typography>
                                ) : (
                                    <Typography color="text.secondary">No results yet.</Typography>
                                )}
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
