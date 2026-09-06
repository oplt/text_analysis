import { useState } from "react";
import {
    Box,
    Button,
    Checkbox,
    Chip,
    FormControlLabel,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import { Add as AddIcon, Save as SaveIcon } from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    addLabel,
    createCodebook,
    getAnnotationProgress,
    listAnnotationQueue,
    saveAnnotations,
} from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { JsonBlock } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";

export default function AnnotationView() {
    const ctx = useResearchContext();
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const [newCodebookName, setNewCodebookName] = useState("");
    const [newLabelName, setNewLabelName] = useState("");
    const [selectedUnitId, setSelectedUnitId] = useState<string | null>(null);
    const [labelValues, setLabelValues] = useState<Record<string, boolean>>({});

    const queueQuery = useQuery({
        queryKey: queryKeys.textResearch.annotationQueue("pending"),
        queryFn: () => listAnnotationQueue("pending"),
    });

    const progressQuery = useQuery({
        queryKey: queryKeys.textResearch.annotationProgress(ctx.selectedCorpusId),
        queryFn: () => getAnnotationProgress(ctx.selectedCorpusId),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const createCodebookMutation = useMutation({
        mutationFn: () => createCodebook(ctx.projectId, { name: newCodebookName.trim() }),
        onSuccess: (codebook) => {
            void client.invalidateQueries({ queryKey: queryKeys.textResearch.codebooks(ctx.projectId) });
            ctx.setSelectedCodebookId(codebook.id);
            setNewCodebookName("");
            showToast({ message: "Codebook created.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to create codebook."),
                severity: "error",
            }),
    });

    const addLabelMutation = useMutation({
        mutationFn: () => addLabel(ctx.selectedCodebookId, { name: newLabelName.trim() }),
        onSuccess: () => {
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.labels(ctx.selectedCodebookId),
            });
            setNewLabelName("");
            showToast({ message: "Label added.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to add label."),
                severity: "error",
            }),
    });

    const saveMutation = useMutation({
        mutationFn: () => {
            if (!selectedUnitId || !ctx.selectedCodebookId) {
                throw new Error("Select a text unit and codebook.");
            }
            return saveAnnotations({
                text_unit_id: selectedUnitId,
                codebook_id: ctx.selectedCodebookId,
                values: ctx.labels.map((label) => ({
                    label_id: label.id,
                    value: labelValues[label.id] ? "yes" : "no",
                })),
                mark_task_complete: true,
            });
        },
        onSuccess: () => {
            void client.invalidateQueries({ queryKey: queryKeys.textResearch.annotationQueue("pending") });
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.annotationProgress(ctx.selectedCorpusId),
            });
            setSelectedUnitId(null);
            setLabelValues({});
            showToast({ message: "Annotations saved.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to save annotations."),
                severity: "error",
            }),
    });

    const selectedQueueItem = queueQuery.data?.find((item) => item.text_unit?.id === selectedUnitId);

    return (
        <Stack spacing={2}>
            <SectionCard title="Codebook" description="Manage labels for manual coding.">
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1} sx={{ mb: 2 }}>
                    <TextField
                        size="small"
                        label="New codebook name"
                        value={newCodebookName}
                        onChange={(e) => setNewCodebookName(e.target.value)}
                    />
                    <Button
                        variant="outlined"
                        startIcon={<AddIcon />}
                        onClick={() => createCodebookMutation.mutate()}
                        disabled={!newCodebookName.trim() || createCodebookMutation.isPending}
                    >
                        Create codebook
                    </Button>
                </Stack>

                <Stack direction={{ xs: "column", sm: "row" }} spacing={1} sx={{ mb: 2 }}>
                    <TextField
                        size="small"
                        label="New label name"
                        value={newLabelName}
                        onChange={(e) => setNewLabelName(e.target.value)}
                        disabled={!ctx.selectedCodebookId}
                    />
                    <Button
                        variant="outlined"
                        startIcon={<AddIcon />}
                        onClick={() => addLabelMutation.mutate()}
                        disabled={!newLabelName.trim() || !ctx.selectedCodebookId || addLabelMutation.isPending}
                    >
                        Add label
                    </Button>
                </Stack>

                <QueryBoundary isLoading={ctx.labelsLoading}>
                    {ctx.labels.length ? (
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Label</TableCell>
                                    <TableCell>Description</TableCell>
                                    <TableCell>Placeholder</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {ctx.labels.map((label) => (
                                    <TableRow key={label.id}>
                                        <TableCell>{label.name}</TableCell>
                                        <TableCell>{label.description ?? "—"}</TableCell>
                                        <TableCell>
                                            {label.is_placeholder ? (
                                                <Chip label="demo" size="small" />
                                            ) : (
                                                "—"
                                            )}
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    ) : (
                        <Typography color="text.secondary">
                            No labels yet. Create a codebook or add labels.
                        </Typography>
                    )}
                </QueryBoundary>
            </SectionCard>

            {ctx.selectedCorpusId ? (
                <SectionCard title="Progress" description="Annotation completion for the selected corpus.">
                    <QueryBoundary
                        isLoading={progressQuery.isLoading}
                        isError={progressQuery.isError}
                        error={progressQuery.error}
                        onRetry={() => void progressQuery.refetch()}
                    >
                        {progressQuery.data ? <JsonBlock data={progressQuery.data} /> : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}

            <SectionCard title="Annotation workspace" description="Pick a queued text unit and apply codebook labels.">
                <QueryBoundary
                    isLoading={queueQuery.isLoading}
                    isError={queueQuery.isError}
                    error={queueQuery.error}
                    onRetry={() => void queueQuery.refetch()}
                >
                    <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" } }}>
                        <Box>
                            <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                Queue ({queueQuery.data?.length ?? 0} pending)
                            </Typography>
                            <Stack spacing={0.5} sx={{ maxHeight: 280, overflow: "auto" }}>
                                {(queueQuery.data ?? []).map((item) => {
                                    const unit = item.text_unit;
                                    if (!unit) return null;
                                    const active = unit.id === selectedUnitId;
                                    return (
                                        <Button
                                            key={item.task.id}
                                            variant={active ? "contained" : "outlined"}
                                            size="small"
                                            sx={{ justifyContent: "flex-start", textAlign: "left" }}
                                            onClick={() => {
                                                setSelectedUnitId(unit.id);
                                                setLabelValues({});
                                            }}
                                        >
                                            {unit.text.slice(0, 80)}
                                            {unit.text.length > 80 ? "…" : ""}
                                        </Button>
                                    );
                                })}
                                {!queueQuery.data?.length ? (
                                    <Typography variant="body2" color="text.secondary">
                                        No pending tasks in your queue.
                                    </Typography>
                                ) : null}
                            </Stack>
                        </Box>

                        <Box>
                            <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                Selected unit
                            </Typography>
                            {selectedQueueItem?.text_unit ? (
                                <>
                                    <Typography
                                        variant="body2"
                                        sx={{
                                            p: 1.5,
                                            borderRadius: 2,
                                            bgcolor: "action.hover",
                                            maxHeight: 160,
                                            overflow: "auto",
                                            mb: 2,
                                        }}
                                    >
                                        {selectedQueueItem.text_unit.text}
                                    </Typography>
                                    <Stack spacing={0.5} sx={{ mb: 2 }}>
                                        {ctx.labels.map((label) => (
                                            <FormControlLabel
                                                key={label.id}
                                                control={
                                                    <Checkbox
                                                        checked={Boolean(labelValues[label.id])}
                                                        onChange={(e) =>
                                                            setLabelValues((prev) => ({
                                                                ...prev,
                                                                [label.id]: e.target.checked,
                                                            }))
                                                        }
                                                    />
                                                }
                                                label={label.name}
                                            />
                                        ))}
                                    </Stack>
                                    <Button
                                        variant="contained"
                                        startIcon={<SaveIcon />}
                                        onClick={() => saveMutation.mutate()}
                                        disabled={
                                            !ctx.selectedCodebookId ||
                                            ctx.labels.length === 0 ||
                                            saveMutation.isPending
                                        }
                                    >
                                        Save annotations
                                    </Button>
                                </>
                            ) : (
                                <Typography variant="body2" color="text.secondary">
                                    Select a queued unit to annotate.
                                </Typography>
                            )}
                        </Box>
                    </Box>
                </QueryBoundary>
            </SectionCard>
        </Stack>
    );
}
