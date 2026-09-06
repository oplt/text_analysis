import { useEffect, useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Checkbox,
    FormControlLabel,
    FormGroup,
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
    Preview as PreviewIcon,
    Save as SaveIcon,
    Tune as PreprocessIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    createPreprocessingProfile,
    listPreprocessingProfiles,
    previewPreprocessing,
    updatePreprocessingProfile,
    type PreprocessingConfigPayload,
    type PreprocessingPreview,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useResearchContext } from "../hooks/useResearchContext";
import { DEFAULT_PREPROCESSING_CONFIG } from "./preprocessingConfig";

const DEFAULT_SAMPLE =
    "Students should not be excluded from the universal education policy in 2024.";

function configFromProfile(config: Record<string, unknown> | undefined): PreprocessingConfigPayload {
    return {
        ...DEFAULT_PREPROCESSING_CONFIG,
        ...(config as PreprocessingConfigPayload),
        lemmatization: false,
    };
}

function toggle(
    config: PreprocessingConfigPayload,
    key: keyof PreprocessingConfigPayload,
    value: boolean
): PreprocessingConfigPayload {
    return { ...config, [key]: value, lemmatization: false };
}

export function PreprocessingPanel() {
    const ctx = useResearchContext();
    const client = useQueryClient();
    const { showToast } = useSnackbar();

    const [selectedProfileId, setSelectedProfileId] = useState("");
    const [profileName, setProfileName] = useState("Default profile");
    const [config, setConfig] = useState<PreprocessingConfigPayload>(DEFAULT_PREPROCESSING_CONFIG);
    const [customStopwordsText, setCustomStopwordsText] = useState("");
    const [sampleText, setSampleText] = useState(DEFAULT_SAMPLE);
    const [preview, setPreview] = useState<PreprocessingPreview | null>(null);

    const profilesQuery = useQuery({
        queryKey: queryKeys.textResearch.preprocessingProfiles(ctx.projectId),
        queryFn: () => listPreprocessingProfiles(ctx.projectId),
        enabled: Boolean(ctx.projectId),
    });

    const profiles = profilesQuery.data ?? [];
    const selectedProfile = useMemo(
        () => profiles.find((profile) => profile.id === selectedProfileId) ?? null,
        [profiles, selectedProfileId]
    );

    useEffect(() => {
        if (!profiles.length) {
            setSelectedProfileId("");
            return;
        }
        if (!selectedProfileId || !profiles.some((p) => p.id === selectedProfileId)) {
            setSelectedProfileId(profiles[0].id);
        }
    }, [profiles, selectedProfileId]);

    useEffect(() => {
        if (!selectedProfile) {
            setProfileName("Default profile");
            setConfig(DEFAULT_PREPROCESSING_CONFIG);
            setCustomStopwordsText("");
            return;
        }
        setProfileName(selectedProfile.name);
        const next = configFromProfile(selectedProfile.config);
        setConfig(next);
        setCustomStopwordsText((next.custom_stopwords ?? []).join("\n"));
    }, [selectedProfile]);

    const workingConfig = useMemo(
        (): PreprocessingConfigPayload => ({
            ...config,
            lemmatization: false,
            custom_stopwords: customStopwordsText
                .split(/[\n,]/)
                .map((item) => item.trim())
                .filter(Boolean),
        }),
        [config, customStopwordsText]
    );

    const previewMutation = useMutation({
        mutationFn: () =>
            previewPreprocessing({
                project_id: ctx.projectId,
                corpus_id: ctx.selectedCorpusId || undefined,
                unit_type: ctx.unitType,
                texts: sampleText.trim() ? [sampleText.trim()] : undefined,
                config: workingConfig,
                sample_size: 5,
            }),
        onSuccess: (data) => setPreview(data),
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to preview preprocessing."),
                severity: "error",
            }),
    });

    const saveMutation = useMutation({
        mutationFn: async () => {
            if (!profileName.trim()) {
                throw new Error("Profile name is required.");
            }
            if (selectedProfileId) {
                return updatePreprocessingProfile(selectedProfileId, {
                    name: profileName.trim(),
                    config: workingConfig,
                });
            }
            return createPreprocessingProfile(ctx.projectId, {
                name: profileName.trim(),
                config: workingConfig,
            });
        },
        onSuccess: (profile) => {
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.preprocessingProfiles(ctx.projectId),
            });
            setSelectedProfileId(profile.id);
            showToast({ message: "Preprocessing profile saved.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to save preprocessing profile."),
                severity: "error",
            }),
    });

    const createMutation = useMutation({
        mutationFn: () =>
            createPreprocessingProfile(ctx.projectId, {
                name: "New preprocessing profile",
                config: DEFAULT_PREPROCESSING_CONFIG,
            }),
        onSuccess: (profile) => {
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.preprocessingProfiles(ctx.projectId),
            });
            setSelectedProfileId(profile.id);
            showToast({ message: "Created a new preprocessing profile.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to create profile."),
                severity: "error",
            }),
    });

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Preprocessing profiles"
                description="Configure reproducible text cleaning for analysis, topics, and classifiers."
                action={
                    <Stack direction="row" spacing={1}>
                        <Button
                            size="small"
                            variant="outlined"
                            onClick={() => createMutation.mutate()}
                            disabled={createMutation.isPending}
                        >
                            New profile
                        </Button>
                        <Button
                            size="small"
                            variant="contained"
                            startIcon={<SaveIcon />}
                            onClick={() => saveMutation.mutate()}
                            disabled={saveMutation.isPending}
                        >
                            Save
                        </Button>
                    </Stack>
                }
            >
                <QueryBoundary
                    isLoading={profilesQuery.isLoading}
                    isError={profilesQuery.isError}
                    error={profilesQuery.error}
                    onRetry={() => void profilesQuery.refetch()}
                    variant="inline"
                >
                    {!profiles.length ? (
                        <EmptyState
                            icon={<PreprocessIcon fontSize="large" />}
                            title="No preprocessing profiles yet"
                            description="Create a profile, tune options, preview tokens, then save for analysis runs."
                            action={
                                <Button variant="contained" onClick={() => createMutation.mutate()}>
                                    Create profile
                                </Button>
                            }
                        />
                    ) : (
                        <Stack spacing={2}>
                            <Stack direction={{ xs: "column", md: "row" }} spacing={1.5}>
                                <TextField
                                    select
                                    size="small"
                                    label="Active profile"
                                    value={selectedProfileId}
                                    onChange={(e) => setSelectedProfileId(e.target.value)}
                                    sx={{ minWidth: 220 }}
                                >
                                    {profiles.map((profile) => (
                                        <MenuItem key={profile.id} value={profile.id}>
                                            {profile.name}
                                        </MenuItem>
                                    ))}
                                </TextField>
                                <TextField
                                    size="small"
                                    label="Profile name"
                                    value={profileName}
                                    onChange={(e) => setProfileName(e.target.value)}
                                    sx={{ flex: 1 }}
                                />
                            </Stack>

                            <Typography variant="body2" color="text.secondary">
                                Active: <strong>{selectedProfile?.name ?? profileName}</strong>
                                {selectedProfile
                                    ? ` · updated ${new Date(selectedProfile.updated_at).toLocaleString()}`
                                    : ""}
                                {" · stemmer: Snowball English"}
                            </Typography>

                            <FormGroup
                                sx={{
                                    display: "grid",
                                    gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
                                    gap: 0.5,
                                }}
                            >
                                {(
                                    [
                                        ["lowercase", "Lowercase"],
                                        ["remove_punctuation", "Remove punctuation"],
                                        ["remove_numbers", "Remove numbers"],
                                        ["remove_stopwords", "Remove stopwords"],
                                        ["preserve_negation", "Preserve negation"],
                                        ["stemming", "Stemming (Snowball English)"],
                                    ] as const
                                ).map(([key, label]) => (
                                    <FormControlLabel
                                        key={key}
                                        control={
                                            <Checkbox
                                                checked={Boolean(config[key])}
                                                onChange={(e) =>
                                                    setConfig(toggle(config, key, e.target.checked))
                                                }
                                            />
                                        }
                                        label={label}
                                    />
                                ))}
                            </FormGroup>

                            <Alert severity="info">
                                Lemmatization is not available yet and cannot be enabled. Profiles never
                                claim lemmatization occurred.
                            </Alert>

                            <Stack direction={{ xs: "column", md: "row" }} spacing={1.5}>
                                <TextField
                                    size="small"
                                    label="N-gram min"
                                    type="number"
                                    value={config.ngram_min ?? 1}
                                    onChange={(e) =>
                                        setConfig({
                                            ...config,
                                            ngram_min: Number(e.target.value) || 1,
                                            lemmatization: false,
                                        })
                                    }
                                    sx={{ width: 140 }}
                                />
                                <TextField
                                    size="small"
                                    label="N-gram max"
                                    type="number"
                                    value={config.ngram_max ?? 1}
                                    onChange={(e) =>
                                        setConfig({
                                            ...config,
                                            ngram_max: Number(e.target.value) || 1,
                                            lemmatization: false,
                                        })
                                    }
                                    sx={{ width: 140 }}
                                />
                                <TextField
                                    size="small"
                                    label="min_df"
                                    type="number"
                                    value={config.min_df ?? 1}
                                    onChange={(e) =>
                                        setConfig({
                                            ...config,
                                            min_df: Number(e.target.value) || 1,
                                            lemmatization: false,
                                        })
                                    }
                                    sx={{ width: 120 }}
                                />
                                <TextField
                                    size="small"
                                    label="max_df"
                                    type="number"
                                    inputProps={{ step: 0.05, min: 0, max: 1 }}
                                    value={config.max_df ?? 1}
                                    onChange={(e) =>
                                        setConfig({
                                            ...config,
                                            max_df: Number(e.target.value) || 1,
                                            lemmatization: false,
                                        })
                                    }
                                    sx={{ width: 120 }}
                                />
                                <TextField
                                    size="small"
                                    label="max_features"
                                    type="number"
                                    value={config.max_features ?? ""}
                                    onChange={(e) =>
                                        setConfig({
                                            ...config,
                                            max_features: e.target.value
                                                ? Number(e.target.value)
                                                : null,
                                            lemmatization: false,
                                        })
                                    }
                                    sx={{ width: 140 }}
                                    helperText="Leave blank for unlimited"
                                />
                            </Stack>

                            <TextField
                                label="Custom stopwords"
                                value={customStopwordsText}
                                onChange={(e) => setCustomStopwordsText(e.target.value)}
                                multiline
                                minRows={2}
                                helperText="One term per line (or comma-separated)."
                            />
                        </Stack>
                    )}
                </QueryBoundary>
            </SectionCard>

            <SectionCard
                title="Live preview"
                description="Compare original text with the processed token stream for the active config."
                action={
                    <Button
                        variant="contained"
                        startIcon={<PreviewIcon />}
                        onClick={() => previewMutation.mutate()}
                        disabled={previewMutation.isPending}
                    >
                        Refresh preview
                    </Button>
                }
            >
                <Stack spacing={2}>
                    <TextField
                        label="Sample text"
                        value={sampleText}
                        onChange={(e) => setSampleText(e.target.value)}
                        multiline
                        minRows={2}
                        helperText={
                            ctx.selectedCorpusId
                                ? "Uses this sample when provided; otherwise samples segmented corpus units."
                                : "Paste sample text to preview without a segmented corpus."
                        }
                    />

                    {preview ? (
                        <Stack spacing={1.5}>
                            <Typography variant="body2" color="text.secondary">
                                Tokens {preview.token_count_before} → {preview.token_count_after}
                                {" · "}vocabulary {preview.vocabulary_size}
                                {preview.profile_name
                                    ? ` · profile ${preview.profile_name}`
                                    : ` · draft ${profileName}`}
                                {" · "}
                                {preview.stemmer}
                            </Typography>

                            <Box sx={{ overflowX: "auto" }}>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Original</TableCell>
                                            <TableCell>Processed</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {preview.rows.map((row, index) => (
                                            <TableRow key={`${index}-${row.original.slice(0, 24)}`}>
                                                <TableCell sx={{ verticalAlign: "top", width: "50%" }}>
                                                    {row.original}
                                                </TableCell>
                                                <TableCell sx={{ verticalAlign: "top", width: "50%" }}>
                                                    {row.processed || (
                                                        <Typography
                                                            component="span"
                                                            color="text.secondary"
                                                            variant="body2"
                                                        >
                                                            (empty)
                                                        </Typography>
                                                    )}
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </Box>

                            {preview.most_frequently_removed_terms.length ? (
                                <Typography variant="body2" color="text.secondary">
                                    Most removed:{" "}
                                    {preview.most_frequently_removed_terms
                                        .slice(0, 12)
                                        .map((item) => `${item.term} (${item.count})`)
                                        .join(" · ")}
                                </Typography>
                            ) : (
                                <Typography variant="body2" color="text.secondary">
                                    No terms removed relative to the baseline token stream.
                                </Typography>
                            )}
                        </Stack>
                    ) : (
                        <Typography variant="body2" color="text.secondary">
                            Run a preview to inspect token changes before saving the profile.
                        </Typography>
                    )}
                </Stack>
            </SectionCard>
        </Stack>
    );
}
