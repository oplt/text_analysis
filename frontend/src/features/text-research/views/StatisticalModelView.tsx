import { useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Checkbox,
    FormControlLabel,
    MenuItem,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import { PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { fitStatisticalModel } from "../../../api/textResearch";
import { SectionCard } from "../../../components/ui/SectionCard";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { MetricCards, ResultsInspector } from "../components/ResearchCharts";
import { RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import type { AnalysisRun } from "../types";

function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
}

function num(value: unknown): number | null {
    if (typeof value === "number" && !Number.isNaN(value)) return value;
    if (typeof value === "string" && value.trim() !== "") {
        const parsed = Number(value);
        return Number.isNaN(parsed) ? null : parsed;
    }
    return null;
}

function formatMetric(value: number | null | undefined, digits = 4): string {
    if (value == null || Number.isNaN(value)) return "—";
    return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

function parseRowsJson(text: string): Array<Record<string, unknown>> {
    const parsed = JSON.parse(text) as unknown;
    if (!Array.isArray(parsed) || !parsed.length) {
        throw new Error("Rows must be a non-empty JSON array of objects.");
    }
    return parsed.map((row, index) => {
        const record = asRecord(row);
        if (!record) throw new Error(`Row ${index} must be an object.`);
        return record;
    });
}

const SAMPLE_ROWS = `[
  {"x": 0, "z": 1, "y": 1},
  {"x": 1, "z": 1, "y": 3},
  {"x": 2, "z": 0, "y": 5},
  {"x": 3, "z": 1, "y": 7},
  {"x": 4, "z": 0, "y": 9},
  {"x": 5, "z": 1, "y": 11}
]`;

export default function StatisticalModelView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();

    const [model, setModel] = useState<"ols" | "logistic">("ols");
    const [dependentVar, setDependentVar] = useState("y");
    const [independentVarsText, setIndependentVarsText] = useState("x,z");
    const [rowsText, setRowsText] = useState(SAMPLE_ROWS);
    const [addIntercept, setAddIntercept] = useState(true);
    const [run, setRun] = useState<AnalysisRun | null>(null);

    const columnHints = useMemo(() => {
        try {
            const rows = parseRowsJson(rowsText);
            return Object.keys(rows[0] ?? {});
        } catch {
            return [];
        }
    }, [rowsText]);

    const fitMutation = useMutation({
        mutationFn: () => {
            const rows = parseRowsJson(rowsText);
            const independentVars = independentVarsText
                .split(/[,\s]+/)
                .map((part) => part.trim())
                .filter(Boolean);
            if (!dependentVar.trim()) throw new Error("Dependent variable is required.");
            if (!independentVars.length) throw new Error("Provide at least one independent variable.");
            return fitStatisticalModel(ctx.selectedCorpusId, {
                model,
                dependent_var: dependentVar.trim(),
                independent_vars: independentVars,
                rows,
                add_intercept: addIntercept,
            });
        },
        onSuccess: (next) => {
            setRun(next);
            showToast({ message: "Statistical model fitted.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to fit statistical model."),
                severity: "error",
            }),
    });

    const results = asRecord(run?.results);
    const coefficients = Array.isArray(results?.coefficients)
        ? results.coefficients.map((row) => asRecord(row)).filter(Boolean)
        : [];
    const notes = Array.isArray(results?.notes) ? results.notes.map(String) : [];

    return (
        <SectionCard
            title="Statistical model"
            description="Fit a user-specified OLS or logistic model on tabular rows you supply. No automatic model search or causal claims."
        >
            <Stack spacing={2}>
                <Alert severity="info">
                    You choose the outcome and predictors. Coefficients are associational unless your
                    research design supports stronger claims.
                </Alert>

                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                    <TextField
                        select
                        size="small"
                        label="Model"
                        value={model}
                        onChange={(e) => setModel(e.target.value as "ols" | "logistic")}
                        sx={{ minWidth: 160 }}
                    >
                        <MenuItem value="ols">OLS</MenuItem>
                        <MenuItem value="logistic">Logistic (0/1)</MenuItem>
                    </TextField>
                    <TextField
                        size="small"
                        label="Dependent variable"
                        value={dependentVar}
                        onChange={(e) => setDependentVar(e.target.value)}
                        sx={{ minWidth: 180 }}
                    />
                    <TextField
                        size="small"
                        label="Independent variables"
                        value={independentVarsText}
                        onChange={(e) => setIndependentVarsText(e.target.value)}
                        helperText="Comma-separated column names"
                        sx={{ minWidth: 240 }}
                    />
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={addIntercept}
                                onChange={(_, checked) => setAddIntercept(checked)}
                            />
                        }
                        label="Add intercept"
                    />
                </Stack>

                {columnHints.length ? (
                    <Typography variant="caption" color="text.secondary">
                        Detected columns: {columnHints.join(", ")}
                    </Typography>
                ) : null}

                <TextField
                    size="small"
                    label="Rows (JSON array of objects)"
                    value={rowsText}
                    onChange={(e) => setRowsText(e.target.value)}
                    multiline
                    minRows={8}
                    fullWidth
                    helperText="Numeric columns only for the selected variables. Logistic y must be 0/1."
                />

                <Button
                    variant="contained"
                    startIcon={<RunIcon />}
                    onClick={() => fitMutation.mutate()}
                    disabled={!ctx.selectedCorpusId || fitMutation.isPending}
                    sx={{ alignSelf: "flex-start" }}
                >
                    Fit model
                </Button>

                {run ? (
                    <Stack spacing={2}>
                        <Typography variant="body2">
                            Run {run.id} — <RunStatusChip status={run.status} />
                        </Typography>
                        {run.error_message ? (
                            <Alert severity="error">{run.error_message}</Alert>
                        ) : null}

                        <MetricCards
                            items={[
                                {
                                    label: "Observations",
                                    value: formatMetric(num(results?.n_observations), 0),
                                },
                                {
                                    label: "R² / pseudo-R²",
                                    value: formatMetric(
                                        num(results?.r_squared) ?? num(results?.pseudo_r_squared)
                                    ),
                                },
                                {
                                    label: "Adj. R²",
                                    value: formatMetric(num(results?.adj_r_squared)),
                                },
                                {
                                    label: "AIC",
                                    value: formatMetric(num(results?.aic)),
                                },
                            ]}
                        />

                        {coefficients.length ? (
                            <Box sx={{ overflowX: "auto" }}>
                                <Typography variant="subtitle2" gutterBottom>
                                    Coefficients
                                </Typography>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Term</TableCell>
                                            <TableCell align="right">Coef</TableCell>
                                            <TableCell align="right">SE</TableCell>
                                            <TableCell align="right">t / z</TableCell>
                                            <TableCell align="right">p</TableCell>
                                            <TableCell align="right">CI low</TableCell>
                                            <TableCell align="right">CI high</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {coefficients.map((row, index) => (
                                            <TableRow key={String(row?.term ?? index)}>
                                                <TableCell>{String(row?.term ?? "—")}</TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(num(row?.coefficient))}
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(num(row?.std_error))}
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(num(row?.t) ?? num(row?.z))}
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(num(row?.p_value))}
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(num(row?.ci_low))}
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(num(row?.ci_high))}
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </Box>
                        ) : null}

                        {notes.map((note) => (
                            <Alert key={note} severity="warning">
                                {note}
                            </Alert>
                        ))}

                        <ResultsInspector title="statistical model" data={run} />
                    </Stack>
                ) : null}
            </Stack>
        </SectionCard>
    );
}
