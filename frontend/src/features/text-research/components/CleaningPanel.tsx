import { useEffect, useMemo, useState } from "react";
import {
    Alert,
    Button,
    Checkbox,
    FormControlLabel,
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
import { Delete as DeleteIcon, PlayArrow as ApplyIcon, Preview as PreviewIcon, Save as SaveIcon } from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
    applyCleaning,
    createCleaningProfile,
    deleteCleaningProfile,
    listCleaningProfiles,
    previewCleaning,
    updateCleaningProfile,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useSnackbar } from "../../../app/snackbarContext";
import { useResearchContext } from "../hooks/useResearchContext";
import type { CleaningPreview, CleaningProfile } from "../types";

const DEFAULT_CONFIG: Record<string, unknown> = {
    unicode_normalization: null,
    normalize_whitespace: true,
    dehyphenate_line_breaks: false,
    remove_headers: false,
    remove_footers: false,
    remove_page_numbers: false,
    exclude_bibliography: false,
    exclude_tables: false,
    custom_regex: [],
};

const TOGGLES = [
    ["normalize_whitespace", "Normalize whitespace"],
    ["dehyphenate_line_breaks", "Join line-break hyphenation"],
    ["remove_headers", "Remove repeated headers"],
    ["remove_footers", "Remove repeated footers"],
    ["remove_page_numbers", "Remove page-number lines"],
    ["exclude_bibliography", "Exclude bibliography"],
    ["exclude_tables", "Exclude pipe-formatted tables"],
] as const;

export function CleaningPreviewDiff({ profileId }: { profileId: string | null }) {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const [sample, setSample] = useState("Paste representative extracted text to compare it with the cleaned result.");
    const [preview, setPreview] = useState<CleaningPreview | null>(null);
    const previewMutation = useMutation({
        mutationFn: () => previewCleaning({
            project_id: ctx.projectId,
            texts: sample.trim() ? [sample.trim()] : undefined,
            cleaning_profile_id: profileId ?? undefined,
        }),
        onSuccess: setPreview,
        onError: (error) => showToast({ message: getQueryErrorMessage(error, "Cleaning preview failed."), severity: "error" }),
    });
    const row = preview?.rows[0];

    return (
        <SectionCard title="Cleaning preview" description="Compare a sample before applying a profile to the corpus.">
            <Stack spacing={1.5}>
                <TextField
                    label="Sample extracted text"
                    multiline
                    minRows={4}
                    value={sample}
                    onChange={(event) => setSample(event.target.value)}
                />
                <Button
                    variant="outlined"
                    startIcon={<PreviewIcon />}
                    onClick={() => previewMutation.mutate()}
                    disabled={!profileId || !sample.trim() || previewMutation.isPending}
                    sx={{ alignSelf: "flex-start" }}
                >
                    {previewMutation.isPending ? "Generating preview…" : "Preview cleaning"}
                </Button>
                {!profileId ? <Alert severity="info">Save or select a cleaning profile to preview it.</Alert> : null}
                {row ? (
                    <Table size="small" aria-label="Cleaning preview comparison">
                        <TableHead><TableRow><TableCell>Original</TableCell><TableCell>Cleaned</TableCell></TableRow></TableHead>
                        <TableBody><TableRow>
                            <TableCell sx={{ whiteSpace: "pre-wrap", verticalAlign: "top" }}>{row.raw_preview}</TableCell>
                            <TableCell sx={{ whiteSpace: "pre-wrap", verticalAlign: "top" }}>{row.cleaned_preview}</TableCell>
                        </TableRow></TableBody>
                    </Table>
                ) : null}
                {row?.steps.length ? (
                    <Typography variant="body2" color="text.secondary">
                        Applied: {row.steps.map((step) => `${step.name} (${step.discarded_chars} characters removed)`).join(", ")}.
                    </Typography>
                ) : null}
            </Stack>
        </SectionCard>
    );
}

export function CleaningRunPanel({ profileId }: { profileId: string | null }) {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const client = useQueryClient();
    const applyMutation = useMutation({
        mutationFn: () => applyCleaning(ctx.selectedCorpusId, { cleaning_profile_id: profileId! }),
        onSuccess: (run) => {
            void Promise.all([
                client.invalidateQueries({ queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId) }),
                client.invalidateQueries({ queryKey: ["text-research", "corpus", ctx.selectedCorpusId, "documents"] }),
                client.invalidateQueries({ queryKey: queryKeys.textResearch.run(run.id) }),
            ]);
            showToast({ message: "Cleaning completed. Raw extracts were preserved.", severity: "success" });
        },
        onError: (error) => showToast({ message: getQueryErrorMessage(error, "Cleaning failed."), severity: "error" }),
    });
    return (
        <SectionCard title="Apply cleaning" description="Rebuild canonical text for every document. Raw extracts are never overwritten.">
            <Stack spacing={1.5}>
                <Alert severity="warning">Applying cleaning changes canonical corpus text. Segment the corpus again afterwards to refresh text units.</Alert>
                <Button
                    variant="contained"
                    color="primary"
                    startIcon={<ApplyIcon />}
                    onClick={() => applyMutation.mutate()}
                    disabled={!profileId || !ctx.selectedCorpusId || applyMutation.isPending}
                    sx={{ alignSelf: "flex-start" }}
                >
                    {applyMutation.isPending ? "Applying profile…" : "Apply to corpus"}
                </Button>
                {!ctx.selectedCorpusId ? <Alert severity="info">Select a corpus before applying this profile.</Alert> : null}
            </Stack>
        </SectionCard>
    );
}

function cleaningProfileDraft(profile: CleaningProfile) {
    return {
        name: profile.name,
        description: profile.description ?? "",
        version: profile.version,
        config: { ...DEFAULT_CONFIG, ...profile.config },
        rulesText: JSON.stringify(profile.config.custom_regex ?? [], null, 2),
    };
}

function CleaningProfileFields({
    profile,
    onSave,
    onDelete,
    savePending,
    deletePending,
}: {
    profile: CleaningProfile;
    onSave: (draft: {
        name: string;
        description: string;
        version: string;
        config: Record<string, unknown>;
        rulesText: string;
    }) => void;
    onDelete: () => void;
    savePending: boolean;
    deletePending: boolean;
}) {
    const initial = cleaningProfileDraft(profile);
    const [name, setName] = useState(initial.name);
    const [description, setDescription] = useState(initial.description);
    const [version, setVersion] = useState(initial.version);
    const [config, setConfig] = useState<Record<string, unknown>>(initial.config);
    const [rulesText, setRulesText] = useState(initial.rulesText);

    return (
        <>
            <Stack direction={{ xs: "column", md: "row" }} spacing={1}>
                <Button
                    variant="contained"
                    startIcon={<SaveIcon />}
                    onClick={() => onSave({ name, description, version, config, rulesText })}
                    disabled={savePending}
                >
                    Save
                </Button>
                <Button
                    color="error"
                    startIcon={<DeleteIcon />}
                    onClick={onDelete}
                    disabled={deletePending}
                >
                    Delete
                </Button>
            </Stack>
            <Stack direction={{ xs: "column", md: "row" }} spacing={1}>
                <TextField label="Name" size="small" value={name} onChange={(event) => setName(event.target.value)} fullWidth />
                <TextField label="Version" size="small" value={version} onChange={(event) => setVersion(event.target.value)} sx={{ minWidth: 130 }} />
            </Stack>
            <TextField label="Description" size="small" value={description} onChange={(event) => setDescription(event.target.value)} fullWidth />
            <TextField select label="Unicode normalization" size="small" value={String(config.unicode_normalization ?? "")} onChange={(event) => setConfig({ ...config, unicode_normalization: event.target.value || null })} sx={{ maxWidth: 250 }}>
                <MenuItem value="">None</MenuItem>{["NFC", "NFKC", "NFD", "NFKD"].map((form) => <MenuItem key={form} value={form}>{form}</MenuItem>)}
            </TextField>
            <Stack direction={{ xs: "column", sm: "row" }} flexWrap="wrap" useFlexGap>
                {TOGGLES.map(([key, label]) => <FormControlLabel key={key} control={<Checkbox checked={Boolean(config[key])} onChange={(event) => setConfig({ ...config, [key]: event.target.checked })} />} label={label} />)}
            </Stack>
            <TextField label="Custom replacement rules (JSON array)" value={rulesText} onChange={(event) => setRulesText(event.target.value)} multiline minRows={3} helperText='Each rule uses {"pattern":"…","replacement":"…","count":0}.' />
        </>
    );
}

export function CleaningProfileEditor({ onProfileChange }: { onProfileChange: (profileId: string | null) => void }) {
    const ctx = useResearchContext();
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const [profileId, setProfileId] = useState("");
    const profilesQuery = useQuery({
        queryKey: queryKeys.textResearch.cleaningProfiles(ctx.projectId),
        queryFn: () => listCleaningProfiles(ctx.projectId),
        enabled: Boolean(ctx.projectId),
    });
    const profiles = useMemo(() => profilesQuery.data ?? [], [profilesQuery.data]);
    const effectiveProfileId = useMemo(() => {
        if (!profiles.length) return "";
        if (profileId && profiles.some((profile) => profile.id === profileId)) return profileId;
        return profiles[0].id;
    }, [profileId, profiles]);
    const selected = useMemo(
        () => profiles.find((profile) => profile.id === effectiveProfileId) ?? null,
        [profiles, effectiveProfileId]
    );

    useEffect(() => onProfileChange(effectiveProfileId || null), [onProfileChange, effectiveProfileId]);

    const invalidate = () => client.invalidateQueries({ queryKey: queryKeys.textResearch.cleaningProfiles(ctx.projectId) });
    const saveMutation = useMutation({
        mutationFn: async (draft: {
            name: string;
            description: string;
            version: string;
            config: Record<string, unknown>;
            rulesText: string;
        }) => {
            let customRegex: unknown;
            try { customRegex = JSON.parse(draft.rulesText); } catch { throw new Error("Custom rules must be valid JSON."); }
            if (!Array.isArray(customRegex)) throw new Error("Custom rules must be a JSON array.");
            const payload = {
                name: draft.name.trim(),
                description: draft.description.trim() || undefined,
                version: draft.version.trim() || "1.0",
                config: { ...draft.config, custom_regex: customRegex },
            };
            if (!payload.name) throw new Error("Profile name is required.");
            return effectiveProfileId
                ? updateCleaningProfile(effectiveProfileId, payload)
                : createCleaningProfile(ctx.projectId, payload);
        },
        onSuccess: (profile) => {
            void invalidate();
            setProfileId(profile.id);
            showToast({ message: "Cleaning profile saved.", severity: "success" });
        },
        onError: (error) => showToast({ message: getQueryErrorMessage(error, "Could not save cleaning profile."), severity: "error" }),
    });
    const createMutation = useMutation({
        mutationFn: () => createCleaningProfile(ctx.projectId, { name: "New cleaning profile", config: DEFAULT_CONFIG }),
        onSuccess: (profile) => { void invalidate(); setProfileId(profile.id); },
    });
    const deleteMutation = useMutation({
        mutationFn: () => deleteCleaningProfile(effectiveProfileId),
        onSuccess: () => { setProfileId(""); void invalidate(); showToast({ message: "Cleaning profile deleted.", severity: "success" }); },
        onError: (error) => showToast({ message: getQueryErrorMessage(error, "Could not delete cleaning profile."), severity: "error" }),
    });

    return (
        <SectionCard title="Cleaning profiles" description="Define deterministic source-text transformations before segmentation.">
            <QueryBoundary isLoading={profilesQuery.isLoading} isError={profilesQuery.isError} error={profilesQuery.error} onRetry={() => void profilesQuery.refetch()} variant="inline">
                <Stack spacing={1.5}>
                    {!profiles.length ? <EmptyState icon={<SaveIcon fontSize="large" />} title="No cleaning profiles" description="Create a profile to normalize extracted source text." /> : null}
                    <Stack direction={{ xs: "column", md: "row" }} spacing={1}>
                        <TextField select label="Profile" size="small" value={effectiveProfileId} onChange={(event) => setProfileId(event.target.value)} sx={{ minWidth: 220 }}>
                            {profiles.map((profile) => <MenuItem key={profile.id} value={profile.id}>{profile.name}</MenuItem>)}
                        </TextField>
                        <Button variant="outlined" onClick={() => createMutation.mutate()} disabled={createMutation.isPending}>New profile</Button>
                    </Stack>
                    {selected ? (
                        <CleaningProfileFields
                            key={selected.id}
                            profile={selected}
                            onSave={(draft) => saveMutation.mutate(draft)}
                            onDelete={() => deleteMutation.mutate()}
                            savePending={saveMutation.isPending}
                            deletePending={deleteMutation.isPending}
                        />
                    ) : null}
                </Stack>
            </QueryBoundary>
        </SectionCard>
    );
}

export function CleaningPanel() {
    const [profileId, setProfileId] = useState<string | null>(null);
    return <Stack spacing={2}><CleaningProfileEditor onProfileChange={setProfileId} /><CleaningPreviewDiff profileId={profileId} /><CleaningRunPanel profileId={profileId} /></Stack>;
}
