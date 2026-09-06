import {
    Box,
    Button,
    FormControl,
    InputLabel,
    MenuItem,
    Select,
    Stack,
    Typography,
} from "@mui/material";
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
import { UNIT_TYPE_OPTIONS } from "../types";

export function CorpusSelector() {
    const ctx = useResearchContext();
    return (
        <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel id="research-corpus-label">Corpus</InputLabel>
            <Select
                labelId="research-corpus-label"
                label="Corpus"
                value={ctx.selectedCorpusId || ""}
                onChange={(e) => ctx.setSelectedCorpusId(e.target.value)}
                disabled={ctx.corpora.length === 0}
            >
                {ctx.corpora.map((corpus) => (
                    <MenuItem key={corpus.id} value={corpus.id}>
                        {corpus.name}
                    </MenuItem>
                ))}
            </Select>
        </FormControl>
    );
}

export function CodebookSelector() {
    const ctx = useResearchContext();
    return (
        <FormControl size="small" sx={{ minWidth: 220 }}>
            <InputLabel id="research-codebook-label">Codebook</InputLabel>
            <Select
                labelId="research-codebook-label"
                label="Codebook"
                value={ctx.selectedCodebookId || ""}
                onChange={(e) => ctx.setSelectedCodebookId(e.target.value)}
                disabled={ctx.codebooks.length === 0}
            >
                {ctx.codebooks.map((codebook) => (
                    <MenuItem key={codebook.id} value={codebook.id}>
                        {codebook.name} (v{codebook.version}
                        {codebook.is_frozen ? " · frozen" : ""})
                    </MenuItem>
                ))}
            </Select>
        </FormControl>
    );
}

export function UnitTypeSelector() {
    const ctx = useResearchContext();
    return (
        <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel id="research-unit-type-label">Unit type</InputLabel>
            <Select
                labelId="research-unit-type-label"
                label="Unit type"
                value={ctx.unitType}
                onChange={(e) => ctx.setUnitType(e.target.value as typeof ctx.unitType)}
            >
                {UNIT_TYPE_OPTIONS.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                        {option.label}
                    </MenuItem>
                ))}
            </Select>
        </FormControl>
    );
}

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
                        ? "Created to upload documents from the Policy Text Lab empty state."
                        : "Created from the Policy Text Lab empty state.",
            }),
        onSuccess: (corpus, intent) => {
            void client.invalidateQueries({ queryKey: queryKeys.textResearch.corpora(ctx.projectId) });
            ctx.setSelectedCorpusId(corpus.id);
            showToast({
                message: intent === "upload" ? "Corpus created. Upload your documents." : "Empty corpus created.",
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

export function ResearchContextBar() {
    const ctx = useResearchContext();
    const navigate = useNavigate();

    if (!ctx.corporaLoading && ctx.corpora.length === 0) {
        return <NoCorpusEmptyState />;
    }

    return (
        <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={2}
            alignItems={{ xs: "stretch", md: "center" }}
            flexWrap="wrap"
        >
            <CorpusSelector />
            <CodebookSelector />
            <UnitTypeSelector />
            {ctx.codebooks.length === 0 ? (
                <Button
                    size="small"
                    variant="outlined"
                    onClick={() => navigate(`/research/${ctx.projectId}/codebook`)}
                >
                    Create codebook
                </Button>
            ) : null}
        </Stack>
    );
}

export function JsonBlock({ data }: { data: unknown }) {
    return (
        <Box
            component="pre"
            sx={{
                m: 0,
                p: 2,
                borderRadius: 2,
                overflow: "auto",
                maxHeight: 420,
                fontSize: 12,
                bgcolor: "action.hover",
            }}
        >
            {JSON.stringify(data, null, 2)}
        </Box>
    );
}

export function RunStatusChip({ status }: { status: string }) {
    const color =
        status === "completed"
            ? "success"
            : status === "failed"
              ? "error"
              : status === "running"
                ? "warning"
                : "default";
    return (
        <Typography component="span" variant="caption" color={`${color}.main`} sx={{ fontWeight: 600 }}>
            {status}
        </Typography>
    );
}
