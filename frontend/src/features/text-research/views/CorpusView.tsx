import { useState } from "react";
import {
    Alert,
    Box,
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import { Add as AddIcon, PlayArrow as SegmentIcon, Science as SeedIcon } from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    createCorpus,
    listDocuments,
    segmentCorpus,
    seedDemoCorpus,
} from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useResearchContext } from "../hooks/useResearchContext";

export default function CorpusView() {
    const ctx = useResearchContext();
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const [createOpen, setCreateOpen] = useState(false);
    const [corpusName, setCorpusName] = useState("");
    const [corpusDescription, setCorpusDescription] = useState("");

    const documentsQuery = useQuery({
        queryKey: queryKeys.textResearch.documents(ctx.selectedCorpusId),
        queryFn: () => listDocuments(ctx.selectedCorpusId, { limit: 100, offset: 0 }),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const seedMutation = useMutation({
        mutationFn: () => seedDemoCorpus(ctx.projectId),
        onSuccess: (corpus) => {
            void client.invalidateQueries({ queryKey: queryKeys.textResearch.corpora(ctx.projectId) });
            ctx.setSelectedCorpusId(corpus.id);
            showToast({ message: "Demo corpus seeded.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to seed demo corpus."),
                severity: "error",
            }),
    });

    const createMutation = useMutation({
        mutationFn: () =>
            createCorpus(ctx.projectId, {
                name: corpusName.trim(),
                description: corpusDescription.trim() || undefined,
            }),
        onSuccess: (corpus) => {
            void client.invalidateQueries({ queryKey: queryKeys.textResearch.corpora(ctx.projectId) });
            ctx.setSelectedCorpusId(corpus.id);
            setCreateOpen(false);
            setCorpusName("");
            setCorpusDescription("");
            showToast({ message: "Corpus created.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to create corpus."),
                severity: "error",
            }),
    });

    const segmentMutation = useMutation({
        mutationFn: () => segmentCorpus(ctx.selectedCorpusId, ctx.unitType),
        onSuccess: (run) => {
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId),
            });
            showToast({
                message: `Segmentation started (${run.status}).`,
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to start segmentation."),
                severity: "error",
            }),
    });

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Corpus management"
                description="Create a corpus, seed synthetic demo data, or segment into text units."
                action={
                    <Stack direction="row" spacing={1}>
                        <Button
                            variant="outlined"
                            startIcon={<SeedIcon />}
                            onClick={() => seedMutation.mutate()}
                            disabled={seedMutation.isPending}
                        >
                            Seed demo
                        </Button>
                        <Button variant="contained" startIcon={<AddIcon />} onClick={() => setCreateOpen(true)}>
                            New corpus
                        </Button>
                    </Stack>
                }
            >
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Active corpus: {ctx.selectedCorpus?.name ?? "None selected"}
                </Typography>
                <Button
                    variant="contained"
                    color="secondary"
                    startIcon={<SegmentIcon />}
                    onClick={() => segmentMutation.mutate()}
                    disabled={!ctx.selectedCorpusId || segmentMutation.isPending}
                >
                    Segment into {ctx.unitType}s
                </Button>
                {segmentMutation.data ? (
                    <Alert severity="info" sx={{ mt: 2 }}>
                        Segmentation run {segmentMutation.data.id} — status: {segmentMutation.data.status}
                    </Alert>
                ) : null}
            </SectionCard>

            <SectionCard title="Documents" description="Documents linked to the selected corpus.">
                {!ctx.selectedCorpusId ? (
                    <Typography color="text.secondary">Select a corpus to list documents.</Typography>
                ) : (
                    <QueryBoundary
                        isLoading={documentsQuery.isLoading}
                        isError={documentsQuery.isError}
                        error={documentsQuery.error}
                        onRetry={() => void documentsQuery.refetch()}
                    >
                        {documentsQuery.data?.items.length ? (
                            <Box sx={{ overflowX: "auto" }}>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Title</TableCell>
                                            <TableCell>Organization</TableCell>
                                            <TableCell>Year</TableCell>
                                            <TableCell>Country</TableCell>
                                            <TableCell>Language</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {documentsQuery.data.items.map((doc) => (
                                            <TableRow key={doc.id}>
                                                <TableCell>{doc.title ?? doc.id.slice(0, 8)}</TableCell>
                                                <TableCell>{doc.organization ?? "—"}</TableCell>
                                                <TableCell>{doc.publication_year ?? "—"}</TableCell>
                                                <TableCell>{doc.country ?? "—"}</TableCell>
                                                <TableCell>{doc.language ?? "—"}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
                                    Showing {documentsQuery.data.items.length} of {documentsQuery.data.total} documents
                                </Typography>
                            </Box>
                        ) : (
                            <Typography color="text.secondary">
                                No documents yet. Seed the demo corpus or add documents via the API.
                            </Typography>
                        )}
                    </QueryBoundary>
                )}
            </SectionCard>

            <Dialog open={createOpen} onClose={() => setCreateOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle>Create corpus</DialogTitle>
                <DialogContent>
                    <Stack spacing={2} sx={{ mt: 1 }}>
                        <TextField
                            label="Name"
                            value={corpusName}
                            onChange={(e) => setCorpusName(e.target.value)}
                            fullWidth
                            required
                        />
                        <TextField
                            label="Description"
                            value={corpusDescription}
                            onChange={(e) => setCorpusDescription(e.target.value)}
                            fullWidth
                            multiline
                            minRows={2}
                        />
                    </Stack>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
                    <Button
                        variant="contained"
                        onClick={() => createMutation.mutate()}
                        disabled={!corpusName.trim() || createMutation.isPending}
                    >
                        Create
                    </Button>
                </DialogActions>
            </Dialog>
        </Stack>
    );
}
