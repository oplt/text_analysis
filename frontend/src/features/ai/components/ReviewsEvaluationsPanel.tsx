import { Alert, Box, Button, Chip, MenuItem, Skeleton, Stack, TextField, Typography } from "@mui/material";
import { Approval as ReviewIcon } from "@mui/icons-material";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryErrorAlert } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { formatDateTime } from "../../../utils/formatters";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { datasetCaseDraftSchema, firstSchemaError, parseJsonObject } from "../studioUtils";
import type { AiStudioModel } from "../hooks/useAiStudioView";

export function ReviewsEvaluationsPanel({ m }: { m: AiStudioModel }) {
    const { reviewsIsError, reviewsError, refetchReviews, reviewsLoading, reviews, reviewNotesById, setReviewNotesById, correctionsById, setCorrectionsById, decideReviewMutation, datasetForm, setDatasetForm, createDatasetMutation, selectedDatasetId, setSelectedDatasetId, datasets, datasetCaseForm, setDatasetCaseForm, createDatasetCaseMutation, runForm, setRunForm, selectedTemplateVersions, runEvaluationMutation, datasetCasesIsError, datasetCasesError, refetchDatasetCases, datasetCasesLoading, selectedDatasetCases, evaluationRunsIsError, evaluationRunsError, refetchEvaluationRuns, evaluationRunsLoading, evaluationRuns, showToast } = m;
    return (
    <SectionCard title="Reviews and evaluations" description="Route sensitive outputs through human approval and keep reusable benchmark datasets for prompt regression testing.">
        <Stack spacing={2}>
            <Stack spacing={1.25}>
                <Typography variant="subtitle2">Review queue</Typography>
                {reviewsIsError && (
                    <QueryErrorAlert
                        error={reviewsError}
                        fallback="Failed to load review queue."
                        onRetry={() => void refetchReviews()}
                    />
                )}
                {reviewsLoading ? (
                    <Stack spacing={1.25}>
                        {Array.from({ length: 2 }).map((_, index) => (
                            <Skeleton key={index} variant="rounded" height={140} sx={{ borderRadius: 3 }} />
                        ))}
                    </Stack>
                ) : reviews.length > 0 ? (
                    reviews.map((review) => (
                        <Box key={review.id} sx={(theme) => ({ p: 2, borderRadius: 4, border: `1px solid ${theme.palette.divider}` })}>
                            <Stack spacing={1}>
                                <Stack direction={{ xs: "column", sm: "row" }} justifyContent="space-between" spacing={1}>
                                    <Typography variant="body2">Run {review.run_id.slice(0, 8)}</Typography>
                                    <Chip label={review.status} size="small" variant="outlined" />
                                </Stack>
                                <TextField
                                    label="Reviewer notes"
                                    value={reviewNotesById[review.id] ?? ""}
                                    onChange={(event) => setReviewNotesById((current) => ({ ...current, [review.id]: event.target.value }))}
                                    fullWidth
                                    size="small"
                                />
                                <TextField
                                    label="Corrected output"
                                    value={correctionsById[review.id] ?? ""}
                                    onChange={(event) => setCorrectionsById((current) => ({ ...current, [review.id]: event.target.value }))}
                                    fullWidth
                                    size="small"
                                    multiline
                                    minRows={2}
                                />
                                <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                                    <Button size="small" variant="outlined" color="success" onClick={() => decideReviewMutation.mutate({ reviewId: review.id, status: "approved", reviewer_notes: reviewNotesById[review.id], corrected_output: correctionsById[review.id] })}>
                                        Approve
                                    </Button>
                                    <Button size="small" variant="outlined" color="warning" onClick={() => decideReviewMutation.mutate({ reviewId: review.id, status: "changes_requested", reviewer_notes: reviewNotesById[review.id], corrected_output: correctionsById[review.id] })}>
                                        Request changes
                                    </Button>
                                    <Button size="small" variant="outlined" color="error" onClick={() => decideReviewMutation.mutate({ reviewId: review.id, status: "rejected", reviewer_notes: reviewNotesById[review.id] })}>
                                        Reject
                                    </Button>
                                </Stack>
                            </Stack>
                        </Box>
                    ))
                ) : (
                    <EmptyState icon={<ReviewIcon />} title="No reviews queued" description="Review requests created from runs will appear here." />
                )}
            </Stack>

            <Stack spacing={1.5}>
                <Typography variant="subtitle2">Evaluation datasets</Typography>
                <TextField label="Dataset name" value={datasetForm.name} onChange={(event) => setDatasetForm((current) => ({ ...current, name: event.target.value }))} fullWidth />
                <TextField label="Dataset description" value={datasetForm.description} onChange={(event) => setDatasetForm((current) => ({ ...current, description: event.target.value }))} fullWidth />
                <Button variant="outlined" disabled={createDatasetMutation.isPending || !datasetForm.name.trim()} onClick={() => createDatasetMutation.mutate(datasetForm)}>
                    {createDatasetMutation.isPending ? "Creating..." : "Create evaluation dataset"}
                </Button>
                <TextField select label="Selected dataset" value={selectedDatasetId} onChange={(event) => setSelectedDatasetId(event.target.value)} fullWidth>
                    {datasets.map((dataset) => (
                        <MenuItem key={dataset.id} value={dataset.id}>
                            {dataset.name}
                        </MenuItem>
                    ))}
                </TextField>
                <TextField label="Case input variables JSON" value={datasetCaseForm.input_variables_json} onChange={(event) => setDatasetCaseForm((current) => ({ ...current, input_variables_json: event.target.value }))} fullWidth multiline minRows={4} />
                <TextField label="Retrieval query" value={datasetCaseForm.retrieval_query} onChange={(event) => setDatasetCaseForm((current) => ({ ...current, retrieval_query: event.target.value }))} fullWidth />
                <TextField label="Expected chunk IDs" helperText="Comma-separated chunk IDs used to calculate citation hit rate." value={datasetCaseForm.expected_chunk_ids} onChange={(event) => setDatasetCaseForm((current) => ({ ...current, expected_chunk_ids: event.target.value }))} fullWidth />
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    {m.documents.map((document) => (
                        <Chip key={document.id} label={document.title} color={m.selectedEvaluationDocumentIds.includes(document.id) ? "primary" : "default"} variant={m.selectedEvaluationDocumentIds.includes(document.id) ? "filled" : "outlined"} onClick={() => m.toggleEvaluationDocument(document.id)} />
                    ))}
                </Stack>
                <TextField label="Expected output text" value={datasetCaseForm.expected_output_text} onChange={(event) => setDatasetCaseForm((current) => ({ ...current, expected_output_text: event.target.value }))} fullWidth multiline minRows={2} />
                <TextField label="Expected output JSON" value={datasetCaseForm.expected_output_json} onChange={(event) => setDatasetCaseForm((current) => ({ ...current, expected_output_json: event.target.value }))} fullWidth multiline minRows={2} />
                <TextField label="Notes" value={datasetCaseForm.notes} onChange={(event) => setDatasetCaseForm((current) => ({ ...current, notes: event.target.value }))} fullWidth />
                <Button
                    variant="outlined"
                    disabled={createDatasetCaseMutation.isPending || !selectedDatasetId}
                    onClick={() => {
                        const result = datasetCaseDraftSchema.safeParse(datasetCaseForm);
                        if (!result.success) {
                            showToast({ message: firstSchemaError(result), severity: "error" });
                            return;
                        }
                        try {
                            createDatasetCaseMutation.mutate({
                                datasetId: selectedDatasetId,
                                payload: {
                                    input_variables: parseJsonObject(result.data.input_variables_json),
                                    retrieval_query: result.data.retrieval_query || null,
                                    document_ids: m.selectedEvaluationDocumentIds,
                                    expected_chunk_ids: result.data.expected_chunk_ids.split(",").map((id) => id.trim()).filter(Boolean),
                                    expected_output_text: result.data.expected_output_text || null,
                                    expected_output_json: result.data.expected_output_json.trim()
                                        ? parseJsonObject(result.data.expected_output_json)
                                        : null,
                                    notes: result.data.notes || null,
                                },
                            });
                        } catch (error) {
                            showToast({
                                message: getQueryErrorMessage(error, "Invalid dataset case JSON."),
                                severity: "error",
                            });
                        }
                    }}
                >
                    {createDatasetCaseMutation.isPending ? "Saving..." : "Add dataset case"}
                </Button>
                <TextField
                    select
                    label="Prompt version for evaluation"
                    value={runForm.prompt_version_id}
                    onChange={(event) => setRunForm((current) => ({ ...current, prompt_version_id: event.target.value }))}
                    fullWidth
                >
                    {selectedTemplateVersions.map((version) => (
                        <MenuItem key={version.id} value={version.id}>
                            v{version.version_number} • {version.model_name}
                        </MenuItem>
                    ))}
                </TextField>
                <Button
                    variant="contained"
                    disabled={runEvaluationMutation.isPending || !selectedDatasetId || !runForm.prompt_version_id}
                    onClick={() => runEvaluationMutation.mutate({ datasetId: selectedDatasetId, promptVersionId: runForm.prompt_version_id })}
                >
                    {runEvaluationMutation.isPending ? "Starting..." : "Run evaluation"}
                </Button>
                {selectedDatasetId && datasetCasesIsError && (
                    <QueryErrorAlert
                        error={datasetCasesError}
                        fallback="Failed to load dataset cases."
                        onRetry={() => void refetchDatasetCases()}
                    />
                )}
                {selectedDatasetId && datasetCasesLoading ? (
                    <Stack spacing={1}>
                        {Array.from({ length: 2 }).map((_, index) => (
                            <Skeleton key={index} variant="rounded" height={72} sx={{ borderRadius: 2 }} />
                        ))}
                    </Stack>
                ) : selectedDatasetCases.length > 0 ? (
                    <Stack spacing={1}>
                        {selectedDatasetCases.map((item) => (
                            <Box key={item.id} sx={(theme) => ({ p: 1.5, borderRadius: 3, border: `1px solid ${theme.palette.divider}` })}>
                                <Typography variant="body2" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>
                                    {JSON.stringify(item.input_variables)}
                                </Typography>
                                {item.expected_output_text && (
                                    <Typography variant="caption" color="text.secondary">
                                        Expected: {item.expected_output_text}
                                    </Typography>
                                )}
                                {item.expected_chunk_ids.length > 0 && (
                                    <Typography variant="caption" color="text.secondary" display="block">
                                        Expected chunks: {item.expected_chunk_ids.length}
                                    </Typography>
                                )}
                            </Box>
                        ))}
                    </Stack>
                ) : null}
                {evaluationRunsIsError && (
                    <QueryErrorAlert
                        error={evaluationRunsError}
                        fallback="Failed to load evaluation runs."
                        onRetry={() => void refetchEvaluationRuns()}
                    />
                )}
                {evaluationRunsLoading ? (
                    <Stack spacing={1}>
                        {Array.from({ length: 2 }).map((_, index) => (
                            <Skeleton key={index} variant="rounded" height={56} sx={{ borderRadius: 2 }} />
                        ))}
                    </Stack>
                ) : evaluationRuns.length > 0 ? (
                    <Stack spacing={1}>
                        {evaluationRuns.map((run) => (
                            <Alert
                                key={run.id}
                                severity={
                                    run.status === "running"
                                        ? "info"
                                        : run.status === "failed"
                                          ? "error"
                                          : run.passed_cases === run.total_cases
                                            ? "success"
                                            : "warning"
                                }
                            >
                                {formatDateTime(run.created_at)}:{" "}
                                {run.status === "running"
                                    ? `Running (${run.total_cases} cases)...`
                                    : run.status === "failed"
                                      ? "Evaluation failed"
                                      : `${run.passed_cases}/${run.total_cases} passed, average score ${run.average_score}`}
                            </Alert>
                        ))}
                    </Stack>
                ) : null}
            </Stack>
        </Stack>
    </SectionCard>
    );
}
