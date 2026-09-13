import { Button, Stack } from "@mui/material";
import {
    Add as AddIcon,
    Science as SeedIcon,
    UploadFile as UploadIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { createCorpus, seedDemoCorpus } from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { queryKeys } from "../../../config/queryKeys";
import { useResearchContext } from "../hooks/useResearchContext";

/**
 * Shared empty corpus CTA used across research entry views.
 * Context selectors live in ResearchContextBar; run status uses components/ui/RunStatusChip.
 */
export function NoCorpusEmptyState() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const client = useQueryClient();
    const { showToast } = useSnackbar();

    const seedMutation = useMutation({
        mutationFn: () => seedDemoCorpus(ctx.projectId),
        onSuccess: (corpus) => {
            void client.invalidateQueries({ queryKey: queryKeys.textResearch.corpora(ctx.projectId) });
            ctx.setSelectedCorpusId(corpus.id);
            showToast({ message: "Demo corpus seeded.", severity: "success" });
            navigate(`/research/${ctx.projectId}/corpus`);
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to seed demo corpus."),
                severity: "error",
            }),
    });

    const createMutation = useMutation({
        mutationFn: (intent: "upload" | "empty") =>
            createCorpus(ctx.projectId, {
                name: "Research corpus",
                description:
                    intent === "upload"
                        ? "Created to upload documents from the Text Research empty state."
                        : "Created from the Text Research empty state.",
            }),
        onSuccess: (corpus, intent) => {
            void client.invalidateQueries({ queryKey: queryKeys.textResearch.corpora(ctx.projectId) });
            ctx.setSelectedCorpusId(corpus.id);
            showToast({
                message:
                    intent === "upload"
                        ? "Corpus created. Upload your documents."
                        : "Empty corpus created.",
                severity: "success",
            });
            navigate(`/research/${ctx.projectId}/corpus`);
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to create corpus."),
                severity: "error",
            }),
    });

    return (
        <EmptyState
            icon={<SeedIcon fontSize="large" />}
            title="No corpus yet"
            description="Start by uploading source documents, creating an empty corpus, or loading the synthetic demo."
            action={
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1} justifyContent="center">
                    <Button
                        variant="contained"
                        startIcon={<UploadIcon />}
                        onClick={() => createMutation.mutate("upload")}
                        disabled={createMutation.isPending}
                    >
                        Upload documents
                    </Button>
                    <Button
                        variant="outlined"
                        startIcon={<AddIcon />}
                        onClick={() => createMutation.mutate("empty")}
                        disabled={createMutation.isPending}
                    >
                        Create empty corpus
                    </Button>
                    <Button
                        variant="outlined"
                        startIcon={<SeedIcon />}
                        onClick={() => seedMutation.mutate()}
                        disabled={seedMutation.isPending}
                    >
                        Load synthetic demo
                    </Button>
                </Stack>
            }
        />
    );
}
