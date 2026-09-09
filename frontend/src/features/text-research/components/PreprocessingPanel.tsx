import { useCallback, useEffect, useMemo, useState } from "react";
import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
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
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
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
import type { PreprocessingProfile } from "../types";
import {
    DEFAULT_PREPROCESSING_CONFIG,
    LANGUAGE_OPTIONS,
    SPACY_MODEL_OPTIONS,
} from "./preprocessingConfig";

const DEFAULT_SAMPLE =
    "Students should not be excluded from the universal education policy in 2024.";

function configFromProfile(config: Record<string, unknown> | undefined): PreprocessingConfigPayload {
    return {
        ...DEFAULT_PREPROCESSING_CONFIG,
        ...(config as PreprocessingConfigPayload),
    };
}

function patchConfig(
    config: PreprocessingConfigPayload,
    partial: Partial<PreprocessingConfigPayload>
): PreprocessingConfigPayload {
    const next = { ...config, ...partial };
    if (next.stemming && next.lemmatization) {
        if (partial.stemming) next.lemmatization = false;
        if (partial.lemmatization) next.stemming = false;
    }
    if (next.pos_lemmatization) {
        next.stemming = false;
        next.lemmatization = false;
    }
    if (partial.stemming || partial.lemmatization) {
        next.pos_lemmatization = false;
    }
    if (next.multilingual) {
        next.language_mode = "per_unit";
        next.auto_detect_language = true;
    } else if (next.auto_detect_language && next.language_mode === "manual") {
        next.language_mode = "auto";
    }
    return next;
}

function preprocessingProfileDraft(profile: PreprocessingProfile) {
    const config = configFromProfile(profile.config);
    return {
        profileName: profile.name,
        config,
        customStopwordsText: (config.custom_stopwords ?? []).join("\n"),
    };
}

function PreprocessingProfileEditor({
    profile,
    onSave,
    onDraftChange,
    savePending,
}: {
    profile: PreprocessingProfile;
    onSave: (draft: {
        profileName: string;
        config: PreprocessingConfigPayload;
        customStopwordsText: string;
    }) => void;
    onDraftChange: (draft: {
        profileName: string;
        config: PreprocessingConfigPayload;
    }) => void;
    savePending: boolean;
}) {
    const initial = preprocessingProfileDraft(profile);
    const [profileName, setProfileName] = useState(initial.profileName);
    const [config, setConfig] = useState<PreprocessingConfigPayload>(initial.config);
    const [customStopwordsText, setCustomStopwordsText] = useState(initial.customStopwordsText);

    const workingConfig = useMemo(
        (): PreprocessingConfigPayload => ({
            ...config,
            custom_stopwords: customStopwordsText
                .split(/[\n,]/)
                .map((item) => item.trim())
                .filter(Boolean),
        }),
        [config, customStopwordsText]
    );

    const needsSpacy =
        Boolean(config.pos_lemmatization) ||
        Boolean(config.entity_masking) ||
        Boolean(config.phrase_detection) ||
        Boolean(config.enable_ner);

    useEffect(() => {
        onDraftChange({ profileName, config: workingConfig });
    }, [profileName, workingConfig, onDraftChange]);

    return (
        <Stack spacing={2}>
            <Stack direction={{ xs: "column", md: "row" }} spacing={1.5} alignItems={{ sm: "center" }}>
                <TextField
                    size="small"
                    label="Profile name"
                    value={profileName}
                    onChange={(e) => setProfileName(e.target.value)}
                    sx={{ flex: 1 }}
                />
                <Button
                    size="small"
                    variant="contained"
                    startIcon={<SaveIcon />}
                    onClick={() =>
                        onSave({ profileName, config: workingConfig, customStopwordsText })
                    }
                    disabled={savePending}
                >
                    Save
                </Button>
            </Stack>

            <Typography variant="body2" color="text.secondary">
                Active: <strong>{profileName}</strong>
                {` · updated ${new Date(profile.updated_at).toLocaleString()}`}
            </Typography>

            <Accordion defaultExpanded disableGutters>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                    <Typography variant="subtitle2">Language & classical cleaning</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Stack spacing={2}>
                                        <Stack
                                            direction={{ xs: "column", sm: "row" }}
                                            spacing={2}
                                            flexWrap="wrap"
                                            useFlexGap
                                        >
                                            <TextField
                                                select
                                                size="small"
                                                label="Language"
                                                value={config.language ?? "en"}
                                                onChange={(e) =>
                                                    setConfig(patchConfig(config, { language: e.target.value }))
                                                }
                                                sx={{ minWidth: 160 }}
                                                disabled={Boolean(config.multilingual)}
                                            >
                                                {LANGUAGE_OPTIONS.map((opt) => (
                                                    <MenuItem key={opt.value} value={opt.value}>
                                                        {opt.label}
                                                    </MenuItem>
                                                ))}
                                            </TextField>
                                            <TextField
                                                select
                                                size="small"
                                                label="Language mode"
                                                value={config.language_mode ?? "manual"}
                                                onChange={(e) =>
                                                    setConfig(
                                                        patchConfig(config, {
                                                            language_mode: e.target.value,
                                                            auto_detect_language: e.target.value !== "manual",
                                                            multilingual: e.target.value === "per_unit",
                                                        })
                                                    )
                                                }
                                                sx={{ minWidth: 180 }}
                                            >
                                                <MenuItem value="manual">Manual</MenuItem>
                                                <MenuItem value="auto">Auto-detect (corpus)</MenuItem>
                                                <MenuItem value="per_unit">Per-unit multilingual</MenuItem>
                                            </TextField>
                                            <FormControlLabel
                                                control={
                                                    <Checkbox
                                                        checked={Boolean(config.auto_detect_language)}
                                                        onChange={(_, checked) =>
                                                            setConfig(
                                                                patchConfig(config, {
                                                                    auto_detect_language: checked,
                                                                    language_mode: checked
                                                                        ? config.language_mode === "per_unit"
                                                                            ? "per_unit"
                                                                            : "auto"
                                                                        : "manual",
                                                                })
                                                            )
                                                        }
                                                    />
                                                }
                                                label="Automatic language detection"
                                            />
                                            <FormControlLabel
                                                control={
                                                    <Checkbox
                                                        checked={Boolean(config.multilingual)}
                                                        onChange={(_, checked) =>
                                                            setConfig(
                                                                patchConfig(config, {
                                                                    multilingual: checked,
                                                                })
                                                            )
                                                        }
                                                    />
                                                }
                                                label="Multilingual (per-unit routing)"
                                            />
                                        </Stack>

                                        <FormGroup
                                            sx={{
                                                display: "grid",
                                                gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
                                                gap: 0.5,
                                            }}
                                        >
                                            {(
                                                [
                                                    ["fix_encoding", "Fix encoding (ftfy)"],
                                                    ["lowercase", "Lowercase"],
                                                    ["remove_punctuation", "Remove punctuation"],
                                                    ["remove_numbers", "Remove numbers"],
                                                    ["remove_stopwords", "Remove stopwords"],
                                                    ["preserve_negation", "Preserve negation"],
                                                    ["stemming", "Stemming (Snowball)"],
                                                    ["lemmatization", "Lemmatization (simplemma)"],
                                                ] as const
                                            ).map(([key, label]) => (
                                                <FormControlLabel
                                                    key={key}
                                                    control={
                                                        <Checkbox
                                                            checked={Boolean(config[key])}
                                                            disabled={
                                                                Boolean(config.pos_lemmatization) &&
                                                                (key === "stemming" || key === "lemmatization")
                                                            }
                                                            onChange={(e) =>
                                                                setConfig(
                                                                    patchConfig(config, {
                                                                        [key]: e.target.checked,
                                                                    })
                                                                )
                                                            }
                                                        />
                                                    }
                                                    label={label}
                                                />
                                            ))}
                                        </FormGroup>
                                    </Stack>
                                </AccordionDetails>
                            </Accordion>

                            <Accordion disableGutters>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                    <Typography variant="subtitle2">Optional spaCy NLP</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Stack spacing={2}>
                                        <Alert severity="info">
                                            Requires optional spaCy + a downloaded model. Classical
                                            preprocessing stays available when these are off.
                                        </Alert>
                                        <TextField
                                            select
                                            size="small"
                                            label="spaCy model"
                                            value={config.spacy_model ?? "en_core_web_sm"}
                                            onChange={(e) =>
                                                setConfig(
                                                    patchConfig(config, { spacy_model: e.target.value })
                                                )
                                            }
                                            sx={{ minWidth: 220, maxWidth: 320 }}
                                        >
                                            {SPACY_MODEL_OPTIONS.map((opt) => (
                                                <MenuItem key={opt.value} value={opt.value}>
                                                    {opt.label}
                                                </MenuItem>
                                            ))}
                                        </TextField>
                                        <FormGroup
                                            sx={{
                                                display: "grid",
                                                gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
                                                gap: 0.5,
                                            }}
                                        >
                                            {(
                                                [
                                                    ["pos_lemmatization", "POS-aware lemmatization"],
                                                    ["enable_ner", "NER (store in run provenance)"],
                                                    ["entity_masking", "Entity masking"],
                                                    ["phrase_detection", "Phrase detection (noun chunks)"],
                                                ] as const
                                            ).map(([key, label]) => (
                                                <FormControlLabel
                                                    key={key}
                                                    control={
                                                        <Checkbox
                                                            checked={Boolean(config[key])}
                                                            onChange={(e) =>
                                                                setConfig(
                                                                    patchConfig(config, {
                                                                        [key]: e.target.checked,
                                                                    })
                                                                )
                                                            }
                                                        />
                                                    }
                                                    label={label}
                                                />
                                            ))}
                                        </FormGroup>
                                        {needsSpacy ? (
                                            <Alert severity="warning">
                                                Preview/save will fail honestly if spaCy or the selected
                                                model is not installed in the backend environment.
                                            </Alert>
                                        ) : null}
                                    </Stack>
                                </AccordionDetails>
                            </Accordion>

                            <Accordion disableGutters>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                    <Typography variant="subtitle2">Vectorizer defaults</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Stack spacing={2}>
                                        <Stack direction={{ xs: "column", md: "row" }} spacing={1.5}>
                                            <TextField
                                                size="small"
                                                label="N-gram min"
                                                type="number"
                                                value={config.ngram_min ?? 1}
                                                onChange={(e) =>
                                                    setConfig(
                                                        patchConfig(config, {
                                                            ngram_min: Number(e.target.value) || 1,
                                                        })
                                                    )
                                                }
                                                sx={{ width: 140 }}
                                            />
                                            <TextField
                                                size="small"
                                                label="N-gram max"
                                                type="number"
                                                value={config.ngram_max ?? 1}
                                                onChange={(e) =>
                                                    setConfig(
                                                        patchConfig(config, {
                                                            ngram_max: Number(e.target.value) || 1,
                                                        })
                                                    )
                                                }
                                                sx={{ width: 140 }}
                                            />
                                            <TextField
                                                size="small"
                                                label="min_df"
                                                type="number"
                                                value={config.min_df ?? 1}
                                                onChange={(e) =>
                                                    setConfig(
                                                        patchConfig(config, {
                                                            min_df: Number(e.target.value) || 1,
                                                        })
                                                    )
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
                                                    setConfig(
                                                        patchConfig(config, {
                                                            max_df: Number(e.target.value) || 1,
                                                        })
                                                    )
                                                }
                                                sx={{ width: 120 }}
                                            />
                                            <TextField
                                                size="small"
                                                label="max_features"
                                                type="number"
                                                value={config.max_features ?? ""}
                                                onChange={(e) =>
                                                    setConfig(
                                                        patchConfig(config, {
                                                            max_features: e.target.value
                                                                ? Number(e.target.value)
                                                                : null,
                                                        })
                                                    )
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
                                </AccordionDetails>
                            </Accordion>
        </Stack>
    );
}

export function PreprocessingPanel() {
    const ctx = useResearchContext();
    const client = useQueryClient();
    const { showToast } = useSnackbar();

    const [selectedProfileId, setSelectedProfileId] = useState("");
    const [sampleText, setSampleText] = useState(DEFAULT_SAMPLE);
    const [preview, setPreview] = useState<PreprocessingPreview | null>(null);
    const [previewConfig, setPreviewConfig] = useState<PreprocessingConfigPayload>(
        DEFAULT_PREPROCESSING_CONFIG
    );
    const [previewProfileName, setPreviewProfileName] = useState("Default profile");

    const profilesQuery = useQuery({
        queryKey: queryKeys.textResearch.preprocessingProfiles(ctx.projectId),
        queryFn: () => listPreprocessingProfiles(ctx.projectId),
        enabled: Boolean(ctx.projectId),
    });

    const profiles = useMemo(() => profilesQuery.data ?? [], [profilesQuery.data]);
    const effectiveProfileId = useMemo(() => {
        if (!profiles.length) return "";
        if (selectedProfileId && profiles.some((profile) => profile.id === selectedProfileId)) {
            return selectedProfileId;
        }
        return profiles[0].id;
    }, [profiles, selectedProfileId]);
    const selectedProfile = useMemo(
        () => profiles.find((profile) => profile.id === effectiveProfileId) ?? null,
        [profiles, effectiveProfileId]
    );

    const previewMutation = useMutation({
        mutationFn: () =>
            previewPreprocessing({
                project_id: ctx.projectId,
                corpus_id: ctx.selectedCorpusId || undefined,
                unit_type: ctx.unitType,
                texts: sampleText.trim() ? [sampleText.trim()] : undefined,
                config: previewConfig,
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
        mutationFn: async (draft: {
            profileName: string;
            config: PreprocessingConfigPayload;
        }) => {
            if (!draft.profileName.trim()) {
                throw new Error("Profile name is required.");
            }
            if (!effectiveProfileId) {
                return createPreprocessingProfile(ctx.projectId, {
                    name: draft.profileName.trim(),
                    config: draft.config,
                });
            }
            return updatePreprocessingProfile(effectiveProfileId, {
                name: draft.profileName.trim(),
                config: draft.config,
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

    const handleDraftChange = useCallback(
        (draft: { profileName: string; config: PreprocessingConfigPayload }) => {
            setPreviewConfig(draft.config);
            setPreviewProfileName(draft.profileName);
        },
        []
    );

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
                description="Classical Unicode preprocessing is the default. Optional spaCy NLP extras stay off unless enabled."
                action={
                    <Button
                        size="small"
                        variant="outlined"
                        onClick={() => createMutation.mutate()}
                        disabled={createMutation.isPending}
                    >
                        New profile
                    </Button>
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
                            <TextField
                                select
                                size="small"
                                label="Active profile"
                                value={effectiveProfileId}
                                onChange={(e) => setSelectedProfileId(e.target.value)}
                                sx={{ minWidth: 220 }}
                            >
                                {profiles.map((profile) => (
                                    <MenuItem key={profile.id} value={profile.id}>
                                        {profile.name}
                                    </MenuItem>
                                ))}
                            </TextField>
                            {selectedProfile ? (
                                <PreprocessingProfileEditor
                                    key={selectedProfile.id}
                                    profile={selectedProfile}
                                    onDraftChange={handleDraftChange}
                                    onSave={(draft) => {
                                        saveMutation.mutate({
                                            profileName: draft.profileName,
                                            config: draft.config,
                                        });
                                    }}
                                    savePending={saveMutation.isPending}
                                />
                            ) : null}
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
                                    : ` · draft ${previewProfileName}`}
                                {" · "}
                                {preview.stemmer || "no stemmer"}
                                {preview.lemmatization_supported ? " · lemmatization available" : ""}
                                {preview.spacy_available ? " · spaCy available" : " · spaCy unavailable"}
                            </Typography>

                            {preview.implementation ? (
                                <Alert severity="info">
                                    Provenance:{" "}
                                    {String(
                                        preview.implementation.preprocessing_implementation ??
                                            "text_research.preprocessing"
                                    )}
                                    {" v"}
                                    {String(
                                        preview.implementation.preprocessing_implementation_version ??
                                            "?"
                                    )}
                                    {preview.implementation.model_name
                                        ? ` · model ${String(preview.implementation.model_name)}`
                                        : ""}
                                    {preview.implementation.model_version
                                        ? ` (${String(preview.implementation.model_version)})`
                                        : ""}
                                    {preview.implementation.lemmatizer
                                        ? ` · lemmatizer ${String(preview.implementation.lemmatizer)}`
                                        : ""}
                                </Alert>
                            ) : null}

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
