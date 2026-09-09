import { useEffect, useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Checkbox,
    Chip,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    FormControlLabel,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import {
    AcUnit as FreezeIcon,
    Add as AddIcon,
    ContentCopy as VersionIcon,
    Lock as LockedIcon,
    MenuBook as CodebookIcon,
} from "@mui/icons-material";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    addLabel,
    createCodebook,
    createCodebookVersion,
    freezeCodebook,
    updateLabel,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useResearchContext } from "../hooks/useResearchContext";
import type { AnnotationLabel, Codebook } from "../types";

type LabelDraft = {
    name: string;
    description: string;
    inclusion_criteria: string;
    exclusion_criteria: string;
    positive_examples: string;
    negative_examples: string;
    is_placeholder: boolean;
};

const emptyDraft = (): LabelDraft => ({
    name: "",
    description: "",
    inclusion_criteria: "",
    exclusion_criteria: "",
    positive_examples: "",
    negative_examples: "",
    is_placeholder: false,
});

function draftFromLabel(label: AnnotationLabel): LabelDraft {
    return {
        name: label.name,
        description: label.description ?? "",
        inclusion_criteria: label.inclusion_criteria ?? "",
        exclusion_criteria: label.exclusion_criteria ?? "",
        positive_examples: (label.positive_examples ?? []).join("\n"),
        negative_examples: (label.negative_examples ?? []).join("\n"),
        is_placeholder: label.is_placeholder,
    };
}

function parseExamples(value: string): string[] {
    return value
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
}

function suggestNextVersion(version: string): string {
    const match = version.trim().match(/^(\d+)(?:\.(\d+))?$/);
    if (!match) return `${version}-next`;
    const major = Number(match[1]);
    const minor = match[2] != null ? Number(match[2]) + 1 : 1;
    return `${major}.${minor}`;
}

function CodebookChip({ codebook }: { codebook: Codebook }) {
    return (
        <Stack direction="row" spacing={0.75} alignItems="center" flexWrap="wrap" useFlexGap>
            <Typography variant="body2">
                {codebook.name} · v{codebook.version}
            </Typography>
            {codebook.is_frozen ? (
                <Chip size="small" color="warning" icon={<LockedIcon />} label="Frozen" />
            ) : (
                <Chip size="small" color="success" label="Editable" />
            )}
        </Stack>
    );
}

export default function CodebookView() {
    const ctx = useResearchContext();
    const client = useQueryClient();
    const { showToast } = useSnackbar();

    const [createOpen, setCreateOpen] = useState(false);
    const [newName, setNewName] = useState("");
    const [newDescription, setNewDescription] = useState("");
    const [versionOpen, setVersionOpen] = useState(false);
    const [newVersion, setNewVersion] = useState("");
    const [selectedLabelId, setSelectedLabelId] = useState<string | null>(null);
    const [draft, setDraft] = useState<LabelDraft>(emptyDraft());
    const [creatingLabel, setCreatingLabel] = useState(false);

    const selected = ctx.selectedCodebook;
    const frozen = Boolean(selected?.is_frozen);
    const selectedLabel = useMemo(
        () => ctx.labels.find((label) => label.id === selectedLabelId) ?? null,
        [ctx.labels, selectedLabelId]
    );

    useEffect(() => {
        if (!ctx.labels.length) {
            setSelectedLabelId(null);
            setDraft(emptyDraft());
            setCreatingLabel(false);
            return;
        }
        if (!selectedLabelId || !ctx.labels.some((label) => label.id === selectedLabelId)) {
            setSelectedLabelId(ctx.labels[0].id);
        }
    }, [ctx.labels, selectedLabelId]);

    useEffect(() => {
        if (creatingLabel) {
            setDraft(emptyDraft());
            return;
        }
        if (selectedLabel) setDraft(draftFromLabel(selectedLabel));
    }, [selectedLabel, creatingLabel]);

    function invalidateCodebooks(nextId?: string) {
        void client.invalidateQueries({ queryKey: queryKeys.textResearch.codebooks(ctx.projectId) });
        if (nextId || ctx.selectedCodebookId) {
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.labels(nextId ?? ctx.selectedCodebookId),
            });
        }
        ctx.refetchCodebooks();
    }

    const createMutation = useMutation({
        mutationFn: () =>
            createCodebook(ctx.projectId, {
                name: newName.trim(),
                description: newDescription.trim() || undefined,
                seed_demo_labels: false,
            }),
        onSuccess: (codebook) => {
            invalidateCodebooks(codebook.id);
            ctx.setSelectedCodebookId(codebook.id);
            setCreateOpen(false);
            setNewName("");
            setNewDescription("");
            showToast({ message: "Codebook created. Add your own labels next.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to create codebook."),
                severity: "error",
            }),
    });

    const freezeMutation = useMutation({
        mutationFn: () => freezeCodebook(ctx.selectedCodebookId),
        onSuccess: () => {
            invalidateCodebooks();
            showToast({
                message: "Codebook frozen. Create a new version to edit labels later.",
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to freeze codebook."),
                severity: "error",
            }),
    });

    const versionMutation = useMutation({
        mutationFn: () => createCodebookVersion(ctx.selectedCodebookId, newVersion.trim()),
        onSuccess: (codebook) => {
            invalidateCodebooks(codebook.id);
            ctx.setSelectedCodebookId(codebook.id);
            setVersionOpen(false);
            setCreatingLabel(false);
            showToast({
                message: `Created codebook version ${codebook.version}.`,
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to create codebook version."),
                severity: "error",
            }),
    });

    const saveLabelMutation = useMutation({
        mutationFn: async () => {
            if (!draft.name.trim()) throw new Error("Label name is required.");
            const payload = {
                name: draft.name.trim(),
                description: draft.description.trim() || null,
                inclusion_criteria: draft.inclusion_criteria.trim() || null,
                exclusion_criteria: draft.exclusion_criteria.trim() || null,
                positive_examples: parseExamples(draft.positive_examples),
                negative_examples: parseExamples(draft.negative_examples),
                is_placeholder: draft.is_placeholder,
            };
            if (creatingLabel) {
                return addLabel(ctx.selectedCodebookId, payload);
            }
            if (!selectedLabelId) throw new Error("Select a label to edit.");
            return updateLabel(selectedLabelId, payload);
        },
        onSuccess: (label) => {
            invalidateCodebooks();
            setCreatingLabel(false);
            setSelectedLabelId(label.id);
            showToast({
                message: creatingLabel ? "Label created." : "Label updated.",
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to save label."),
                severity: "error",
            }),
    });

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Codebook management"
                description="Define coding schemes, freeze versions used in annotation, and clone new editable versions."
                action={
                    <Button variant="contained" startIcon={<AddIcon />} onClick={() => setCreateOpen(true)}>
                        New codebook
                    </Button>
                }
            >
                {!ctx.codebooks.length ? (
                    <EmptyState
                        icon={<CodebookIcon fontSize="large" />}
                        title="No codebooks yet"
                        description="Create a codebook, then define labels with inclusion/exclusion criteria and examples."
                        action={
                            <Button variant="contained" onClick={() => setCreateOpen(true)}>
                                Create codebook
                            </Button>
                        }
                    />
                ) : (
                    <Stack spacing={2}>
                        <Stack direction={{ xs: "column", md: "row" }} spacing={1} flexWrap="wrap" useFlexGap>
                            {ctx.codebooks.map((codebook) => (
                                <Button
                                    key={codebook.id}
                                    size="small"
                                    variant={
                                        codebook.id === ctx.selectedCodebookId ? "contained" : "outlined"
                                    }
                                    onClick={() => {
                                        ctx.setSelectedCodebookId(codebook.id);
                                        setCreatingLabel(false);
                                    }}
                                >
                                    {codebook.name} v{codebook.version}
                                    {codebook.is_frozen ? " · frozen" : ""}
                                </Button>
                            ))}
                        </Stack>

                        {selected ? (
                            <Stack spacing={1.5}>
                                <CodebookChip codebook={selected} />
                                {selected.description ? (
                                    <Typography variant="body2" color="text.secondary">
                                        {selected.description}
                                    </Typography>
                                ) : null}
                                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                    <Button
                                        variant="outlined"
                                        startIcon={<FreezeIcon />}
                                        disabled={frozen || freezeMutation.isPending}
                                        onClick={() => freezeMutation.mutate()}
                                    >
                                        Freeze version
                                    </Button>
                                    <Button
                                        variant="outlined"
                                        startIcon={<VersionIcon />}
                                        onClick={() => {
                                            setNewVersion(suggestNextVersion(selected.version));
                                            setVersionOpen(true);
                                        }}
                                    >
                                        Create new version
                                    </Button>
                                </Stack>
                                {frozen ? (
                                    <Alert severity="warning">
                                        This codebook version is frozen. Label edits are blocked — create a new
                                        version to change definitions.
                                    </Alert>
                                ) : null}
                            </Stack>
                        ) : null}
                    </Stack>
                )}
            </SectionCard>

            {selected ? (
                <SectionCard
                    title="Labels"
                    description="Edit definition, inclusion/exclusion criteria, and examples for the selected codebook version."
                    action={
                        <Button
                            startIcon={<AddIcon />}
                            disabled={frozen}
                            onClick={() => {
                                setCreatingLabel(true);
                                setSelectedLabelId(null);
                                setDraft(emptyDraft());
                            }}
                        >
                            Add label
                        </Button>
                    }
                >
                    <Box
                        sx={{
                            display: "grid",
                            gap: 2,
                            gridTemplateColumns: { xs: "1fr", md: "220px 1fr" },
                        }}
                    >
                        <Stack spacing={1}>
                            {ctx.labels.map((label) => (
                                <Button
                                    key={label.id}
                                    size="small"
                                    variant={
                                        !creatingLabel && label.id === selectedLabelId
                                            ? "contained"
                                            : "outlined"
                                    }
                                    onClick={() => {
                                        setCreatingLabel(false);
                                        setSelectedLabelId(label.id);
                                    }}
                                    sx={{ justifyContent: "flex-start" }}
                                >
                                    {label.name}
                                    {label.is_placeholder ? " · demo" : ""}
                                </Button>
                            ))}
                            {!ctx.labels.length ? (
                                <Typography variant="body2" color="text.secondary">
                                    No labels yet.
                                </Typography>
                            ) : null}
                        </Stack>

                        <Stack spacing={1.5}>
                            {creatingLabel || selectedLabel ? (
                                <>
                                    <TextField
                                        label="Name"
                                        size="small"
                                        value={draft.name}
                                        disabled={frozen && !creatingLabel}
                                        onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                                        fullWidth
                                    />
                                    <TextField
                                        label="Definition"
                                        size="small"
                                        value={draft.description}
                                        disabled={frozen && !creatingLabel}
                                        onChange={(e) =>
                                            setDraft({ ...draft, description: e.target.value })
                                        }
                                        fullWidth
                                        multiline
                                        minRows={2}
                                    />
                                    <TextField
                                        label="Inclusion criteria"
                                        size="small"
                                        value={draft.inclusion_criteria}
                                        disabled={frozen && !creatingLabel}
                                        onChange={(e) =>
                                            setDraft({ ...draft, inclusion_criteria: e.target.value })
                                        }
                                        fullWidth
                                        multiline
                                        minRows={2}
                                    />
                                    <TextField
                                        label="Exclusion criteria"
                                        size="small"
                                        value={draft.exclusion_criteria}
                                        disabled={frozen && !creatingLabel}
                                        onChange={(e) =>
                                            setDraft({ ...draft, exclusion_criteria: e.target.value })
                                        }
                                        fullWidth
                                        multiline
                                        minRows={2}
                                    />
                                    <TextField
                                        label="Positive examples (one per line)"
                                        size="small"
                                        value={draft.positive_examples}
                                        disabled={frozen && !creatingLabel}
                                        onChange={(e) =>
                                            setDraft({ ...draft, positive_examples: e.target.value })
                                        }
                                        fullWidth
                                        multiline
                                        minRows={2}
                                    />
                                    <TextField
                                        label="Negative examples (one per line)"
                                        size="small"
                                        value={draft.negative_examples}
                                        disabled={frozen && !creatingLabel}
                                        onChange={(e) =>
                                            setDraft({ ...draft, negative_examples: e.target.value })
                                        }
                                        fullWidth
                                        multiline
                                        minRows={2}
                                    />
                                    <FormControlLabel
                                        control={
                                            <Checkbox
                                                size="small"
                                                checked={draft.is_placeholder}
                                                disabled={frozen}
                                                onChange={(event) =>
                                                    setDraft({
                                                        ...draft,
                                                        is_placeholder: event.target.checked,
                                                    })
                                                }
                                            />
                                        }
                                        label="Placeholder / demo label"
                                    />
                                    {(creatingLabel || selectedLabel?.is_placeholder) && (
                                        <Chip
                                            size="small"
                                            label={
                                                selectedLabel?.is_placeholder || draft.is_placeholder
                                                    ? "Placeholder / demo label"
                                                    : "Research label"
                                            }
                                        />
                                    )}
                                    <Stack direction="row" spacing={1}>
                                        <Button
                                            variant="contained"
                                            disabled={frozen || saveLabelMutation.isPending}
                                            onClick={() => saveLabelMutation.mutate()}
                                        >
                                            {creatingLabel ? "Create label" : "Save label"}
                                        </Button>
                                        {creatingLabel ? (
                                            <Button
                                                onClick={() => {
                                                    setCreatingLabel(false);
                                                    if (ctx.labels[0]) {
                                                        setSelectedLabelId(ctx.labels[0].id);
                                                    }
                                                }}
                                            >
                                                Cancel
                                            </Button>
                                        ) : null}
                                    </Stack>
                                </>
                            ) : (
                                <Typography color="text.secondary">Select a label to edit.</Typography>
                            )}
                        </Stack>
                    </Box>
                </SectionCard>
            ) : null}

            <Dialog open={createOpen} onClose={() => setCreateOpen(false)} fullWidth maxWidth="sm">
                <DialogTitle>Create codebook</DialogTitle>
                <DialogContent>
                    <Stack spacing={2} sx={{ mt: 1 }}>
                        <TextField
                            label="Name"
                            value={newName}
                            onChange={(e) => setNewName(e.target.value)}
                            fullWidth
                            required
                        />
                        <TextField
                            label="Description"
                            value={newDescription}
                            onChange={(e) => setNewDescription(e.target.value)}
                            fullWidth
                            multiline
                            minRows={2}
                            helperText="Codebooks start empty. Add labels that match your research design."
                        />
                    </Stack>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
                    <Button
                        variant="contained"
                        disabled={!newName.trim() || createMutation.isPending}
                        onClick={() => createMutation.mutate()}
                    >
                        Create
                    </Button>
                </DialogActions>
            </Dialog>

            <Dialog open={versionOpen} onClose={() => setVersionOpen(false)} fullWidth maxWidth="xs">
                <DialogTitle>Create codebook version</DialogTitle>
                <DialogContent>
                    <TextField
                        sx={{ mt: 1 }}
                        label="New version"
                        value={newVersion}
                        onChange={(e) => setNewVersion(e.target.value)}
                        fullWidth
                        helperText="Clones labels into a new editable codebook version."
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setVersionOpen(false)}>Cancel</Button>
                    <Button
                        variant="contained"
                        disabled={!newVersion.trim() || versionMutation.isPending}
                        onClick={() => versionMutation.mutate()}
                    >
                        Create version
                    </Button>
                </DialogActions>
            </Dialog>
        </Stack>
    );
}
