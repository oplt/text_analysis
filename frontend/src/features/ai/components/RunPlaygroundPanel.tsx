import { Box, Button, Chip, FormControlLabel, MenuItem, Stack, Switch, TextField, Typography } from "@mui/material";
import { AutoAwesome as AiIcon, PlayCircleOutline as RunIcon } from "@mui/icons-material";
import { EmptyState } from "../../../components/ui/EmptyState";
import { SectionCard } from "../../../components/ui/SectionCard";
import { formatDateTime } from "../../../utils/formatters";
import { aiRunDraftSchema, firstSchemaError, formatCostMicros, parseJsonObject } from "../studioUtils";
import type { AiStudioModel } from "../hooks/useAiStudioView";

export function RunPlaygroundPanel({ m }: { m: AiStudioModel }) {
    const { runForm, setRunForm, templateKeyOptions, documents, selectedDocumentIds, toggleDocument, createRunMutation, showToast, recentRuns, createReviewMutation, createFeedbackMutation, feedbackCommentByRunId, setFeedbackCommentByRunId, correctionsById, setCorrectionsById } = m;
    return (
    <SectionCard title="Run playground" description="Execute prompt versions with structured variables, retrieval context, and human-review routing.">
        <Stack spacing={2}>
            <TextField
                select
                label="Prompt template"
                value={runForm.prompt_template_key}
                onChange={(event) => setRunForm((current) => ({ ...current, prompt_template_key: event.target.value }))}
                fullWidth
            >
                {templateKeyOptions.map((template) => (
                    <MenuItem key={template.id} value={template.key}>
                        {template.name} ({template.key})
                    </MenuItem>
                ))}
            </TextField>
            <TextField
                label="Variables JSON"
                value={runForm.variables_json}
                onChange={(event) => setRunForm((current) => ({ ...current, variables_json: event.target.value }))}
                fullWidth
                multiline
                minRows={8}
            />
            <TextField
                label="Retrieval query"
                value={runForm.retrieval_query}
                onChange={(event) => setRunForm((current) => ({ ...current, retrieval_query: event.target.value }))}
                fullWidth
            />
            <TextField
                label="Top K chunks"
                value={runForm.top_k}
                onChange={(event) => setRunForm((current) => ({ ...current, top_k: event.target.value }))}
                fullWidth
            />
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {documents.map((document) => (
                    <Chip
                        key={document.id}
                        label={document.title}
                        color={selectedDocumentIds.includes(document.id) ? "primary" : "default"}
                        variant={selectedDocumentIds.includes(document.id) ? "filled" : "outlined"}
                        onClick={() => toggleDocument(document.id)}
                    />
                ))}
            </Stack>
            <FormControlLabel
                control={
                    <Switch
                        checked={runForm.review_required}
                        onChange={(event) => setRunForm((current) => ({ ...current, review_required: event.target.checked }))}
                    />
                }
                label="Request human review after this run"
            />
            <Button
                variant="contained"
                startIcon={<RunIcon />}
                disabled={createRunMutation.isPending || !runForm.prompt_template_key}
                onClick={() => {
                    const result = aiRunDraftSchema.safeParse(runForm);
                    if (!result.success) {
                        showToast({ message: firstSchemaError(result), severity: "error" });
                        return;
                    }
                    createRunMutation.mutate({
                        prompt_template_key: result.data.prompt_template_key,
                        variables: parseJsonObject(result.data.variables_json),
                        retrieval_query: result.data.retrieval_query || undefined,
                        document_ids: selectedDocumentIds,
                        top_k: Number(result.data.top_k),
                        review_required: result.data.review_required,
                    });
                }}
            >
                {createRunMutation.isPending ? "Running..." : "Run prompt"}
            </Button>
            {recentRuns.length > 0 ? (
                <Stack spacing={1.25}>
                    {recentRuns.map((run) => (
                        <Box key={run.id} sx={(theme) => ({ p: 2, borderRadius: 4, border: `1px solid ${theme.palette.divider}` })}>
                            <Stack spacing={1}>
                                <Stack direction={{ xs: "column", sm: "row" }} justifyContent="space-between" spacing={1}>
                                    <Typography variant="subtitle2">
                                        {run.provider_key}/{run.model_name}
                                    </Typography>
                                    <Chip label={run.status} size="small" color={run.status === "completed" ? "success" : "warning"} variant="outlined" />
                                </Stack>
                                <Typography variant="body2" color="text.secondary">
                                    {run.output_text?.slice(0, 280) || JSON.stringify(run.output_json, null, 2).slice(0, 280) || "No output"}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    {formatDateTime(run.created_at)} • {run.total_tokens} tokens • {formatCostMicros(run.estimated_cost_micros)}
                                </Typography>
                                {(run.retrieval_degraded || run.memory_degraded) && (
                                    <Chip
                                        size="small"
                                        color="warning"
                                        variant="outlined"
                                        label={`Context degraded${run.degradation_reason ? `: ${run.degradation_reason}` : ""}`}
                                        sx={{ alignSelf: "flex-start" }}
                                    />
                                )}
                                <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                                    <Button size="small" variant="outlined" onClick={() => createReviewMutation.mutate(run.id)}>
                                        Request review
                                    </Button>
                                    <Button
                                        size="small"
                                        variant="outlined"
                                        color="success"
                                        onClick={() => createFeedbackMutation.mutate({
                                            runId: run.id,
                                            rating: 1,
                                            comment: feedbackCommentByRunId[run.id],
                                        })}
                                    >
                                        Thumbs up
                                    </Button>
                                    <Button
                                        size="small"
                                        variant="outlined"
                                        color="warning"
                                        onClick={() => createFeedbackMutation.mutate({
                                            runId: run.id,
                                            rating: -1,
                                            comment: feedbackCommentByRunId[run.id],
                                            corrected_output: correctionsById[run.id],
                                        })}
                                    >
                                        Thumbs down
                                    </Button>
                                </Stack>
                                <TextField
                                    label="Feedback note"
                                    value={feedbackCommentByRunId[run.id] ?? ""}
                                    onChange={(event) =>
                                        setFeedbackCommentByRunId((current) => ({ ...current, [run.id]: event.target.value }))
                                    }
                                    fullWidth
                                    size="small"
                                />
                                <TextField
                                    label="Correction"
                                    value={correctionsById[run.id] ?? ""}
                                    onChange={(event) =>
                                        setCorrectionsById((current) => ({ ...current, [run.id]: event.target.value }))
                                    }
                                    fullWidth
                                    size="small"
                                    multiline
                                    minRows={2}
                                />
                            </Stack>
                        </Box>
                    ))}
                </Stack>
            ) : (
                <EmptyState icon={<AiIcon />} title="No AI runs yet" description="Run a prompt version to capture outputs, token usage, and review state." />
            )}
        </Stack>
    </SectionCard>
    );
}
