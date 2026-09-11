import { useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Chip,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
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
import {
    Add as AddIcon,
    ContentCopy as VersionIcon,
    Edit as EditIcon,
    MenuBook as DictIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    createDictionary,
    createDictionaryVersion,
    listDictionaries,
    listRuns,
    updateDictionary,
    type ResearchDictionary,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { AskAboutThisButton } from "../components/assistant/AskAboutThisButton";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useResearchContext } from "../hooks/useResearchContext";

type DictionaryDraft = {
    name: string;
    description: string;
    version: string;
    language: string;
    termsText: string;
    hierarchyText: string;
    exclusionsText: string;
};

const EMPTY_DRAFT: DictionaryDraft = {
    name: "",
    description: "",
    version: "1",
    language: "",
    termsText: "",
    hierarchyText: "",
    exclusionsText: "",
};

function parseTerms(text: string): string[] {
    return text
        .split(/[\n,]+/)
        .map((part) => part.trim())
        .filter(Boolean);
}

function parseJsonObject(text: string, label: string): Record<string, unknown> | undefined {
    const trimmed = text.trim();
    if (!trimmed) return undefined;
    const parsed: unknown = JSON.parse(trimmed);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error(`${label} must be a JSON object.`);
    }
    return parsed as Record<string, unknown>;
}

function parseJsonArray(text: string, label: string): unknown[] | undefined {
    const trimmed = text.trim();
    if (!trimmed) return undefined;
    const parsed: unknown = JSON.parse(trimmed);
    if (!Array.isArray(parsed)) throw new Error(`${label} must be a JSON array.`);
    return parsed;
}

function draftFromDictionary(dictionary: ResearchDictionary): DictionaryDraft {
    return {
        name: dictionary.name,
        description: dictionary.description ?? "",
        version: dictionary.version,
        language: dictionary.language ?? "",
        termsText: dictionary.terms.join("\n"),
        hierarchyText: dictionary.hierarchy
            ? JSON.stringify(dictionary.hierarchy, null, 2)
            : "",
        exclusionsText: dictionary.exclusions?.length
            ? JSON.stringify(dictionary.exclusions, null, 2)
            : "",
    };
}

export default function DictionaryManagerView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const client = useQueryClient();
    const { showToast } = useSnackbar();

    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [editorOpen, setEditorOpen] = useState(false);
    const [draft, setDraft] = useState<DictionaryDraft>(EMPTY_DRAFT);
    const [editingId, setEditingId] = useState<string | null>(null);

    const dictionariesQuery = useQuery({
        queryKey: queryKeys.textResearch.dictionaries(ctx.projectId),
        queryFn: () => listDictionaries(ctx.projectId),
        enabled: Boolean(ctx.projectId),
    });

    const usageQuery = useQuery({
        queryKey: queryKeys.textResearch.runs(ctx.projectId, ctx.selectedCorpusId, "dictionary_analysis"),
        queryFn: () =>
            listRuns(ctx.projectId, {
                corpus_id: ctx.selectedCorpusId || undefined,
                run_type: "dictionary_analysis",
                limit: 20,
            }),
        enabled: Boolean(ctx.projectId),
    });

    const selected = useMemo(
        () => (dictionariesQuery.data ?? []).find((d) => d.id === selectedId) ?? null,
        [dictionariesQuery.data, selectedId]
    );

    const saveMutation = useMutation({
        mutationFn: async () => {
            const terms = parseTerms(draft.termsText);
            const hierarchy = parseJsonObject(draft.hierarchyText, "Hierarchy");
            const exclusions = parseJsonArray(draft.exclusionsText, "Exclusions");
            const payload = {
                name: draft.name.trim(),
                description: draft.description.trim() || undefined,
                version: draft.version.trim() || "1",
                language: draft.language.trim() || undefined,
                terms,
                hierarchy,
                exclusions,
            };
            if (editingId) {
                return updateDictionary(editingId, payload);
            }
            return createDictionary(ctx.projectId, payload);
        },
        onSuccess: async (dictionary) => {
            await client.invalidateQueries({
                queryKey: queryKeys.textResearch.dictionaries(ctx.projectId),
            });
            setSelectedId(dictionary.id);
            setEditorOpen(false);
            showToast({
                message: editingId ? "Dictionary updated." : "Dictionary created.",
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Dictionary save failed."),
                severity: "error",
            }),
    });

    const versionMutation = useMutation({
        mutationFn: () => createDictionaryVersion(selectedId!),
        onSuccess: async (dictionary) => {
            await client.invalidateQueries({
                queryKey: queryKeys.textResearch.dictionaries(ctx.projectId),
            });
            setSelectedId(dictionary.id);
            showToast({
                message: `Created version ${dictionary.version}.`,
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Could not create version."),
                severity: "error",
            }),
    });

    function openCreate() {
        setEditingId(null);
        setDraft(EMPTY_DRAFT);
        setEditorOpen(true);
    }

    function openEdit(dictionary: ResearchDictionary) {
        setEditingId(dictionary.id);
        setDraft(draftFromDictionary(dictionary));
        setEditorOpen(true);
    }

    const usageList = usageQuery.data?.items ?? [];

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Dictionary manager"
                description="Administer versioned dictionaries, hierarchies, exclusions, and languages. Run dictionary analysis from Analysis."
                action={
                    <Stack direction="row" spacing={1}>
                        <Button
                            size="small"
                            variant="outlined"
                            onClick={() =>
                                navigate(`/research/${ctx.projectId}/analysis?tab=dictionaries`)
                            }
                        >
                            Run analysis
                        </Button>
                        <Button
                            size="small"
                            variant="contained"
                            startIcon={<AddIcon />}
                            onClick={openCreate}
                        >
                            New dictionary
                        </Button>
                    </Stack>
                }
            >
                <QueryBoundary
                    isLoading={dictionariesQuery.isLoading}
                    isError={dictionariesQuery.isError}
                    error={dictionariesQuery.error}
                    onRetry={() => void dictionariesQuery.refetch()}
                >
                    {(dictionariesQuery.data ?? []).length === 0 ? (
                        <EmptyState
                            icon={<DictIcon fontSize="large" />}
                            title="No dictionaries yet"
                            description="Create a user-defined dictionary with terms or a hierarchy for analysis."
                            action={
                                <Button variant="contained" startIcon={<AddIcon />} onClick={openCreate}>
                                    Create dictionary
                                </Button>
                            }
                        />
                    ) : (
                        <Box sx={{ overflowX: "auto" }}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Name</TableCell>
                                        <TableCell>Version</TableCell>
                                        <TableCell>Language</TableCell>
                                        <TableCell>Terms</TableCell>
                                        <TableCell>Created</TableCell>
                                        <TableCell align="right">Actions</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {(dictionariesQuery.data ?? []).map((dictionary) => (
                                        <TableRow
                                            key={dictionary.id}
                                            selected={selectedId === dictionary.id}
                                            hover
                                            onClick={() => setSelectedId(dictionary.id)}
                                            sx={{ cursor: "pointer" }}
                                        >
                                            <TableCell>{dictionary.name}</TableCell>
                                            <TableCell>{dictionary.version}</TableCell>
                                            <TableCell>{dictionary.language || "—"}</TableCell>
                                            <TableCell>{dictionary.terms.length}</TableCell>
                                            <TableCell>
                                                {new Date(dictionary.created_at).toLocaleDateString()}
                                            </TableCell>
                                            <TableCell align="right">
                                                <Button
                                                    size="small"
                                                    startIcon={<EditIcon fontSize="small" />}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        openEdit(dictionary);
                                                    }}
                                                >
                                                    Edit
                                                </Button>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </Box>
                    )}
                </QueryBoundary>
            </SectionCard>

            {selected ? (
                <SectionCard
                    title={selected.name}
                    description={`Version ${selected.version}${selected.language ? ` · ${selected.language}` : ""}`}
                    action={
                        <Stack direction="row" spacing={1}>
                            <AskAboutThisButton
                                label="Suggest corpus-derived terms"
                                intent="semantic_search"
                                question={`Suggest corpus-derived candidate terms/expressions related to dictionary "${selected.name}" (existing terms: ${selected.terms.slice(0, 30).join(", ")}). Provide candidates for researcher review only; do not add them automatically.`}
                                onAsk={ctx.askAbout}
                            />
                            <Button
                                size="small"
                                variant="outlined"
                                startIcon={<VersionIcon />}
                                disabled={versionMutation.isPending}
                                onClick={() => versionMutation.mutate()}
                            >
                                New version
                            </Button>
                        </Stack>
                    }
                >
                    <Stack spacing={1.5}>
                        <Typography variant="body2" color="text.secondary">
                            {selected.description || "No description."}
                        </Typography>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                            <Chip size="small" label={`${selected.terms.length} terms`} />
                            <Chip
                                size="small"
                                variant="outlined"
                                label={
                                    selected.hierarchy
                                        ? "Has hierarchy"
                                        : "Flat term list"
                                }
                            />
                            <Chip
                                size="small"
                                variant="outlined"
                                label={`${selected.exclusions?.length ?? 0} exclusions`}
                            />
                            {selected.format ? (
                                <Chip size="small" variant="outlined" label={selected.format} />
                            ) : null}
                        </Stack>
                        <Box>
                            <Typography variant="subtitle2" gutterBottom>
                                Terms
                            </Typography>
                            <Typography
                                variant="body2"
                                sx={{ whiteSpace: "pre-wrap", fontFamily: "monospace" }}
                            >
                                {selected.terms.slice(0, 40).join(", ")}
                                {selected.terms.length > 40
                                    ? ` … (+${selected.terms.length - 40} more)`
                                    : ""}
                            </Typography>
                        </Box>
                        {selected.hierarchy ? (
                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Hierarchy
                                </Typography>
                                <Box
                                    component="pre"
                                    sx={{
                                        m: 0,
                                        p: 1.5,
                                        borderRadius: 1,
                                        bgcolor: "action.hover",
                                        overflow: "auto",
                                        fontSize: 12,
                                    }}
                                >
                                    {JSON.stringify(selected.hierarchy, null, 2)}
                                </Box>
                            </Box>
                        ) : null}
                        {(selected.exclusions?.length ?? 0) > 0 ? (
                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Exclusions
                                </Typography>
                                <Box
                                    component="pre"
                                    sx={{
                                        m: 0,
                                        p: 1.5,
                                        borderRadius: 1,
                                        bgcolor: "action.hover",
                                        overflow: "auto",
                                        fontSize: 12,
                                    }}
                                >
                                    {JSON.stringify(selected.exclusions, null, 2)}
                                </Box>
                            </Box>
                        ) : null}
                    </Stack>
                </SectionCard>
            ) : null}

            <SectionCard
                title="Dictionary usage in analysis"
                description="Recent dictionary analysis runs for this project/corpus."
            >
                {usageList.length === 0 ? (
                    <Typography variant="body2" color="text.secondary">
                        No dictionary analysis runs yet. Execute from Analysis → Dictionaries.
                    </Typography>
                ) : (
                    <Box sx={{ overflowX: "auto" }}>
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Run</TableCell>
                                    <TableCell>Status</TableCell>
                                    <TableCell>Dictionary</TableCell>
                                    <TableCell>Created</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {usageList.map((run) => {
                                    const params =
                                        run.parameters && typeof run.parameters === "object"
                                            ? (run.parameters as Record<string, unknown>)
                                            : {};
                                    const dictId =
                                        typeof params.dictionary_id === "string"
                                            ? params.dictionary_id
                                            : null;
                                    return (
                                        <TableRow key={run.id}>
                                            <TableCell sx={{ fontFamily: "monospace" }}>
                                                {run.id.slice(0, 8)}…
                                            </TableCell>
                                            <TableCell>{run.status}</TableCell>
                                            <TableCell>
                                                {dictId
                                                    ? (dictionariesQuery.data ?? []).find(
                                                          (d) => d.id === dictId
                                                      )?.name ?? dictId.slice(0, 8)
                                                    : "Custom terms"}
                                            </TableCell>
                                            <TableCell>
                                                {new Date(run.created_at).toLocaleString()}
                                            </TableCell>
                                        </TableRow>
                                    );
                                })}
                            </TableBody>
                        </Table>
                    </Box>
                )}
            </SectionCard>

            <Dialog
                open={editorOpen}
                onClose={() => setEditorOpen(false)}
                fullWidth
                maxWidth="md"
            >
                <DialogTitle>{editingId ? "Edit dictionary" : "Create dictionary"}</DialogTitle>
                <DialogContent>
                    <Stack spacing={2} sx={{ mt: 1 }}>
                        <Alert severity="info">
                            Administration only. Execute matching from Analysis → Dictionaries.
                        </Alert>
                        <TextField
                            label="Name"
                            size="small"
                            value={draft.name}
                            onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))}
                            required
                        />
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                            <TextField
                                label="Version"
                                size="small"
                                value={draft.version}
                                onChange={(e) =>
                                    setDraft((d) => ({ ...d, version: e.target.value }))
                                }
                                sx={{ minWidth: 120 }}
                            />
                            <TextField
                                select
                                label="Language"
                                size="small"
                                value={draft.language}
                                onChange={(e) =>
                                    setDraft((d) => ({ ...d, language: e.target.value }))
                                }
                                sx={{ minWidth: 160 }}
                            >
                                <MenuItem value="">Unspecified</MenuItem>
                                {["en", "de", "fr", "es", "tr", "other"].map((lang) => (
                                    <MenuItem key={lang} value={lang}>
                                        {lang}
                                    </MenuItem>
                                ))}
                            </TextField>
                        </Stack>
                        <TextField
                            label="Description"
                            size="small"
                            value={draft.description}
                            onChange={(e) =>
                                setDraft((d) => ({ ...d, description: e.target.value }))
                            }
                            multiline
                            minRows={2}
                        />
                        <TextField
                            label="Terms (one per line or comma-separated)"
                            size="small"
                            value={draft.termsText}
                            onChange={(e) =>
                                setDraft((d) => ({ ...d, termsText: e.target.value }))
                            }
                            multiline
                            minRows={4}
                        />
                        <TextField
                            label="Hierarchy JSON (optional)"
                            size="small"
                            value={draft.hierarchyText}
                            onChange={(e) =>
                                setDraft((d) => ({ ...d, hierarchyText: e.target.value }))
                            }
                            multiline
                            minRows={4}
                            placeholder='{"positive": ["good", "great"], "negative": ["bad"]}'
                        />
                        <TextField
                            label="Exclusions JSON array (optional)"
                            size="small"
                            value={draft.exclusionsText}
                            onChange={(e) =>
                                setDraft((d) => ({ ...d, exclusionsText: e.target.value }))
                            }
                            multiline
                            minRows={3}
                            placeholder='[{"term": "not good", "reason": "negation"}]'
                        />
                    </Stack>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setEditorOpen(false)}>Cancel</Button>
                    <Button
                        variant="contained"
                        disabled={!draft.name.trim() || saveMutation.isPending}
                        onClick={() => saveMutation.mutate()}
                    >
                        Save
                    </Button>
                </DialogActions>
            </Dialog>
        </Stack>
    );
}
