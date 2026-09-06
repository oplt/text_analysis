import { useMemo, useState } from "react";
import {
    Alert,
    Button,
    Checkbox,
    FormControl,
    FormControlLabel,
    InputLabel,
    MenuItem,
    Select,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import { PlaylistAdd as AssignIcon } from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { listUserDirectory } from "../../../api/users";
import { assignCorpusAnnotationTasks, getDashboardSummary } from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES } from "../../../config/queryTiming";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useAuth } from "../../../hooks/useAuth";
import { useResearchContext } from "../hooks/useResearchContext";
import { UNIT_TYPE_OPTIONS, type UnitType } from "../types";

type Strategy = "overlap" | "shared" | "disjoint";

function previewLoads(
    sampleSize: number,
    annotatorCount: number,
    strategy: Strategy,
    overlapCount: number
): { perAnnotator: number; unique: number; overlap: number } {
    if (annotatorCount < 1) return { perAnnotator: 0, unique: 0, overlap: 0 };
    if (strategy === "shared") {
        return { perAnnotator: sampleSize, unique: sampleSize, overlap: sampleSize };
    }
    if (strategy === "disjoint") {
        return {
            perAnnotator: Math.ceil(sampleSize / annotatorCount),
            unique: sampleSize,
            overlap: 0,
        };
    }
    const overlap = Math.min(overlapCount, sampleSize);
    const remainder = sampleSize - overlap;
    const perAnnotator = overlap + Math.ceil(remainder / annotatorCount);
    return { perAnnotator, unique: sampleSize, overlap };
}

export function AnnotationSetupPanel() {
    const ctx = useResearchContext();
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const { currentUser } = useAuth();

    const [unitType, setUnitType] = useState<UnitType>(ctx.unitType);
    const [sampleSize, setSampleSize] = useState(300);
    const [strategy, setStrategy] = useState<Strategy>("overlap");
    const [overlapCount, setOverlapCount] = useState(100);
    const [selectedAnnotators, setSelectedAnnotators] = useState<string[]>(
        currentUser ? [currentUser.id] : []
    );

    const usersQuery = useQuery({
        queryKey: queryKeys.users.directory,
        queryFn: listUserDirectory,
        staleTime: QUERY_STALE_TIMES.userDirectory,
    });

    const dashboardQuery = useQuery({
        queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId),
        queryFn: () => getDashboardSummary(ctx.selectedCorpusId),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const availableUnits = dashboardQuery.data?.text_unit_counts?.[unitType] ?? 0;
    const preview = useMemo(
        () => previewLoads(sampleSize, selectedAnnotators.length, strategy, overlapCount),
        [sampleSize, selectedAnnotators.length, strategy, overlapCount]
    );

    const assignMutation = useMutation({
        mutationFn: () =>
            assignCorpusAnnotationTasks(ctx.selectedCorpusId, {
                unit_type: unitType,
                sample_size: sampleSize,
                annotator_ids: selectedAnnotators,
                strategy,
                overlap_count: strategy === "overlap" ? overlapCount : undefined,
            }),
        onSuccess: (result) => {
            ctx.setUnitType(unitType);
            void client.invalidateQueries({
                queryKey: ["text-research", "annotation-queue"],
            });
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.annotationProgress(ctx.selectedCorpusId),
            });
            showToast({
                message: `Created ${result.assigned_count} task(s) across ${result.unique_units} units (${result.overlap_units} overlap).`,
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to create annotation tasks."),
                severity: "error",
            }),
    });

    function toggleAnnotator(userId: string) {
        setSelectedAnnotators((current) =>
            current.includes(userId)
                ? current.filter((id) => id !== userId)
                : [...current, userId]
        );
    }

    return (
        <Stack spacing={2}>
            <Typography variant="body2" color="text.secondary">
                Configure sampling and overlap, then create annotation tasks in bulk.
                Available {unitType} units: {availableUnits.toLocaleString()}.
            </Typography>

            <Stack direction={{ xs: "column", md: "row" }} spacing={1.5}>
                <FormControl size="small" sx={{ minWidth: 160 }}>
                    <InputLabel>Unit type</InputLabel>
                    <Select
                        label="Unit type"
                        value={unitType}
                        onChange={(e) => setUnitType(e.target.value as UnitType)}
                    >
                        {UNIT_TYPE_OPTIONS.map((option) => (
                            <MenuItem key={option.value} value={option.value}>
                                {option.label}
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <TextField
                    size="small"
                    type="number"
                    label="Sample size (unique units)"
                    value={sampleSize}
                    onChange={(e) => setSampleSize(Math.max(1, Number(e.target.value) || 1))}
                    sx={{ width: 200 }}
                />
                <FormControl size="small" sx={{ minWidth: 180 }}>
                    <InputLabel>Strategy</InputLabel>
                    <Select
                        label="Strategy"
                        value={strategy}
                        onChange={(e) => setStrategy(e.target.value as Strategy)}
                    >
                        <MenuItem value="overlap">Overlap (reliability)</MenuItem>
                        <MenuItem value="shared">Shared (full overlap)</MenuItem>
                        <MenuItem value="disjoint">Disjoint (no overlap)</MenuItem>
                    </Select>
                </FormControl>
                {strategy === "overlap" ? (
                    <TextField
                        size="small"
                        type="number"
                        label="Overlap count"
                        value={overlapCount}
                        onChange={(e) => setOverlapCount(Math.max(0, Number(e.target.value) || 0))}
                        sx={{ width: 150 }}
                    />
                ) : null}
            </Stack>

            <Stack spacing={0.5}>
                <Typography variant="subtitle2">Annotators</Typography>
                {(usersQuery.data ?? []).map((user) => (
                    <FormControlLabel
                        key={user.id}
                        control={
                            <Checkbox
                                size="small"
                                checked={selectedAnnotators.includes(user.id)}
                                onChange={() => toggleAnnotator(user.id)}
                            />
                        }
                        label={user.full_name || user.email}
                    />
                ))}
                {usersQuery.isError ? (
                    <Alert severity="warning">Could not load user directory.</Alert>
                ) : null}
            </Stack>

            <Alert severity="info">
                Preview: {preview.unique.toLocaleString()} unique units · about{" "}
                {preview.perAnnotator.toLocaleString()} per annotator ·{" "}
                {preview.overlap.toLocaleString()} overlap
                {selectedAnnotators.length === 2 && strategy === "overlap"
                    ? ` (e.g. Annotator A/B ≈ ${preview.perAnnotator} with ${preview.overlap} shared)`
                    : ""}
            </Alert>

            <Button
                variant="contained"
                startIcon={<AssignIcon />}
                onClick={() => assignMutation.mutate()}
                disabled={
                    !ctx.selectedCorpusId ||
                    selectedAnnotators.length === 0 ||
                    availableUnits === 0 ||
                    assignMutation.isPending
                }
            >
                Create annotation tasks
            </Button>
        </Stack>
    );
}
