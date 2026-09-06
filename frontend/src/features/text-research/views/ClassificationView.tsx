import { useState } from "react";
import {
    Alert,
    Box,
    Button,
    Checkbox,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import { PlayArrow as TrainIcon, Preview as PreviewIcon } from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { useAuth } from "../../../hooks/useAuth";
import {
    freezeDataset,
    getRun,
    assignUncertainPredictions,
    listUncertainPredictions,
    listClassifiers,
    predictClassifier,
    previewDataset,
    trainClassifier,
} from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { JsonBlock, RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";

export default function ClassificationView() {
    const ctx = useResearchContext();
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const { currentUser } = useAuth();
    const [snapshotName, setSnapshotName] = useState("Training snapshot");
    const [snapshotId, setSnapshotId] = useState<string | null>(null);
    const [trainRunId, setTrainRunId] = useState<string | null>(null);
    const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
    const [selectedUncertainIds, setSelectedUncertainIds] = useState<string[]>([]);

    const labelIds = ctx.labels.map((l) => l.id);

    const previewQuery = useQuery({
        queryKey: [
            "text-research",
            "dataset-preview",
            ctx.selectedCorpusId,
            ctx.selectedCodebookId,
            ctx.unitType,
            labelIds.join(","),
        ],
        queryFn: () =>
            previewDataset({
                corpus_id: ctx.selectedCorpusId,
                unit_type: ctx.unitType,
                codebook_id: ctx.selectedCodebookId,
                label_ids: labelIds,
            }),
        enabled: Boolean(ctx.selectedCorpusId && ctx.selectedCodebookId && labelIds.length > 0),
    });

    const classifiersQuery = useQuery({
        queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
        queryFn: () => listClassifiers(ctx.projectId, ctx.selectedCorpusId),
        enabled: Boolean(ctx.projectId),
    });

    const trainRunQuery = useQuery({
        queryKey: queryKeys.textResearch.run(trainRunId ?? ""),
        queryFn: () => getRun(trainRunId!),
        enabled: Boolean(trainRunId),
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            return status === "running" || status === "pending" ? 2000 : false;
        },
    });

    const uncertainQuery = useQuery({
        queryKey: ["text-research", "uncertain-predictions", selectedModelId],
        queryFn: () => listUncertainPredictions(selectedModelId!),
        enabled: Boolean(selectedModelId),
    });

    const freezeMutation = useMutation({
        mutationFn: () =>
            freezeDataset({
                name: snapshotName.trim(),
                corpus_id: ctx.selectedCorpusId,
                unit_type: ctx.unitType,
                codebook_id: ctx.selectedCodebookId,
                label_ids: labelIds,
            }),
        onSuccess: (snapshot) => {
            setSnapshotId(snapshot.id);
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.datasetSnapshots(ctx.projectId, ctx.selectedCorpusId),
            });
            showToast({ message: "Dataset snapshot frozen.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to freeze dataset."),
                severity: "error",
            }),
    });

    const trainMutation = useMutation({
        mutationFn: () =>
            trainClassifier({
                snapshot_id: snapshotId!,
                name: `Classifier ${new Date().toLocaleDateString()}`,
            }),
        onSuccess: (run) => {
            setTrainRunId(run.id);
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
            });
            showToast({ message: "Classifier training started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to train classifier."),
                severity: "error",
            }),
    });

    const predictMutation = useMutation({
        mutationFn: () =>
            predictClassifier(selectedModelId!, {
                unit_type: ctx.unitType,
                only_unannotated: true,
            }),
        onSuccess: () => {
            void uncertainQuery.refetch();
            showToast({ message: "Predictions saved. Review uncertain cases below.", severity: "success" });
        },
        onError: (error) =>
            showToast({ message: getQueryErrorMessage(error, "Prediction failed."), severity: "error" }),
    });

    const assignMutation = useMutation({
        mutationFn: () =>
            assignUncertainPredictions(selectedModelId!, selectedUncertainIds, [currentUser!.id]),
        onSuccess: () => {
            setSelectedUncertainIds([]);
            showToast({ message: "Selected cases were added to your annotation queue.", severity: "success" });
        },
        onError: (error) =>
            showToast({ message: getQueryErrorMessage(error, "Could not assign cases."), severity: "error" }),
    });

    const canPreview = Boolean(ctx.selectedCorpusId && ctx.selectedCodebookId && labelIds.length > 0);

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Dataset preview"
                description="Inspect labeled units before freezing a training snapshot."
                action={
                    <Button
                        variant="outlined"
                        startIcon={<PreviewIcon />}
                        onClick={() => void previewQuery.refetch()}
                        disabled={!canPreview}
                    >
                        Refresh preview
                    </Button>
                }
            >
                {!canPreview ? (
                    <Typography color="text.secondary">
                        Select corpus, codebook, and labels to preview the training dataset.
                    </Typography>
                ) : (
                    <QueryBoundary
                        isLoading={previewQuery.isLoading}
                        isError={previewQuery.isError}
                        error={previewQuery.error}
                        onRetry={() => void previewQuery.refetch()}
                    >
                        {previewQuery.data ? (
                            <Stack spacing={2}>
                                <Typography variant="body2">
                                    {previewQuery.data.unit_count} units across{" "}
                                    {previewQuery.data.document_count} documents
                                </Typography>
                                {previewQuery.data.warnings.map((warning) => (
                                    <Alert key={warning} severity="warning">
                                        {warning}
                                    </Alert>
                                ))}
                                <JsonBlock data={previewQuery.data.class_distribution} />
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                )}
            </SectionCard>

            <SectionCard title="Train classifier" description="Freeze a snapshot, then train a model.">
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1} sx={{ mb: 2 }}>
                    <TextField
                        size="small"
                        label="Snapshot name"
                        value={snapshotName}
                        onChange={(e) => setSnapshotName(e.target.value)}
                    />
                    <Button
                        variant="outlined"
                        onClick={() => freezeMutation.mutate()}
                        disabled={!canPreview || freezeMutation.isPending}
                    >
                        Freeze snapshot
                    </Button>
                    <Button
                        variant="contained"
                        startIcon={<TrainIcon />}
                        onClick={() => trainMutation.mutate()}
                        disabled={!snapshotId || trainMutation.isPending}
                    >
                        Train classifier
                    </Button>
                </Stack>
                {snapshotId ? (
                    <Typography variant="caption" color="text.secondary">
                        Active snapshot: {snapshotId}
                    </Typography>
                ) : null}
                {trainRunId && trainRunQuery.data ? (
                    <Box sx={{ mt: 2 }}>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                            Training run — <RunStatusChip status={trainRunQuery.data.status} />
                        </Typography>
                        {trainRunQuery.data.metrics ? (
                            <JsonBlock data={trainRunQuery.data.metrics} />
                        ) : null}
                    </Box>
                ) : null}
            </SectionCard>

            <SectionCard title="Trained models" description="Classifiers saved for this project/corpus.">
                <QueryBoundary
                    isLoading={classifiersQuery.isLoading}
                    isError={classifiersQuery.isError}
                    error={classifiersQuery.error}
                    onRetry={() => void classifiersQuery.refetch()}
                >
                    {classifiersQuery.data?.length ? (
                        <Box sx={{ overflowX: "auto" }}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Name</TableCell>
                                        <TableCell>Family</TableCell>
                                        <TableCell>Version</TableCell>
                                        <TableCell>Created</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {classifiersQuery.data.map((model) => (
                                        <TableRow key={model.id} selected={selectedModelId === model.id}>
                                            <TableCell>{model.name ?? model.id.slice(0, 8)}</TableCell>
                                            <TableCell>{model.model_family}</TableCell>
                                            <TableCell>{model.version}</TableCell>
                                            <TableCell>
                                                {new Date(model.created_at).toLocaleDateString()}
                                            </TableCell>
                                            <TableCell align="right">
                                                <Button
                                                    size="small"
                                                    variant={selectedModelId === model.id ? "contained" : "outlined"}
                                                    onClick={() => setSelectedModelId(model.id)}
                                                >
                                                    Select
                                                </Button>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </Box>
                    ) : (
                        <Typography color="text.secondary">No trained models yet.</Typography>
                    )}
                </QueryBoundary>
            </SectionCard>

            <SectionCard
                title="Apply model & uncertain cases"
                description="Predictions stay separate from human coding. Send difficult cases to annotation deliberately."
                action={
                    <Button
                        variant="contained"
                        onClick={() => predictMutation.mutate()}
                        disabled={!selectedModelId || predictMutation.isPending}
                    >
                        Apply to unannotated units
                    </Button>
                }
            >
                {!selectedModelId ? (
                    <Typography color="text.secondary">Select a trained model above.</Typography>
                ) : (
                    <QueryBoundary
                        isLoading={uncertainQuery.isLoading}
                        isError={uncertainQuery.isError}
                        error={uncertainQuery.error}
                        onRetry={() => void uncertainQuery.refetch()}
                    >
                        <Stack spacing={1}>
                            <Typography variant="body2" color="text.secondary">
                                Lowest uncertainty distance is shown first; these are candidates for human review, not automatic labels.
                            </Typography>
                            {uncertainQuery.data?.map((item) => {
                                const checked = selectedUncertainIds.includes(item.text_unit.id);
                                return (
                                    <Box key={item.prediction.id} sx={{ display: "flex", gap: 1, alignItems: "flex-start", py: 0.5 }}>
                                        <Checkbox
                                            checked={checked}
                                            onChange={(_, next) =>
                                                setSelectedUncertainIds((ids) =>
                                                    next ? [...ids, item.text_unit.id] : ids.filter((id) => id !== item.text_unit.id)
                                                )
                                            }
                                        />
                                        <Box sx={{ flex: 1 }}>
                                            <Typography variant="body2">{item.text_unit.text}</Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                Predicted: {item.prediction.predicted_labels.join(", ") || "none"} · uncertainty {item.prediction.uncertainty?.toFixed(3) ?? "n/a"}
                                            </Typography>
                                        </Box>
                                    </Box>
                                );
                            })}
                            <Button
                                sx={{ alignSelf: "flex-start" }}
                                onClick={() => assignMutation.mutate()}
                                disabled={!selectedUncertainIds.length || !currentUser || assignMutation.isPending}
                            >
                                Send selected to my annotation queue
                            </Button>
                        </Stack>
                    </QueryBoundary>
                )}
            </SectionCard>
        </Stack>
    );
}
