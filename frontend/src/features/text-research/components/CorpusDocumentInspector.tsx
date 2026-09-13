import { useEffect, useRef, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Chip,
    Link,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { deleteDocument, getSourceText, updateDocument } from "../../../api/textResearch";
import { getRagDocument } from "../../../api/rag";
import { AdvancedSettings } from "../../../components/ui/AdvancedSettings";
import { FormGrid } from "../../../components/ui/FormGrid";
import { KeyValueList } from "../../../components/ui/KeyValueList";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import type { CorpusDocument, UnitType } from "../types";
import {
    resolveCitationHighlightRange,
    type PageProvenance,
} from "./citationOffsetResolver";

export type MetadataDraft = {
    title: string;
    organization: string;
    organization_type: string;
    publication_year: string;
    publication_type: string;
    country: string;
    region: string;
    cultural_sphere: string;
    language: string;
    education_level: string;
    source_url: string;
    research_notes: string;
};

export function toMetadataDraft(doc: CorpusDocument): MetadataDraft {
    return {
        title: doc.title ?? "",
        organization: doc.organization ?? "",
        organization_type: doc.organization_type ?? "",
        publication_year: doc.publication_year != null ? String(doc.publication_year) : "",
        publication_type: doc.publication_type ?? "",
        country: doc.country ?? "",
        region: doc.region ?? "",
        cultural_sphere: doc.cultural_sphere ?? "",
        language: doc.language ?? "",
        education_level: doc.education_level ?? "",
        source_url: doc.source_url ?? "",
        research_notes: doc.research_notes ?? "",
    };
}

type CitationCoordinate = Pick<
    import("../../../api/textResearch").AssistantCitation,
    "offset_scope_id" | "offset_coordinate_system" | "source_spans"
> | null;

export type CorpusDocumentInspectorProps = {
    document: CorpusDocument | null;
    corpusId: string;
    unitType?: UnitType;
    unitCountForType?: number;
    onRemoved?: () => void;
    onSaved?: (document: CorpusDocument) => void;
    highlightSnippet?: string | null;
    pageHint?: number | null;
    charStart?: number | null;
    charEnd?: number | null;
    sourceSpanIds?: string[] | null;
    offsetScope?: "parsed_document" | "page" | "canonical_document" | null;
    sourceCoordinate?: CitationCoordinate;
    /** Fetch source text when the inspector is visible. */
    active?: boolean;
    compactActions?: boolean;
};

export function CorpusDocumentInspector({
    document,
    corpusId,
    unitType,
    unitCountForType,
    onRemoved,
    onSaved,
    highlightSnippet,
    pageHint,
    charStart,
    charEnd,
    sourceSpanIds,
    offsetScope,
    sourceCoordinate,
    active = true,
    compactActions = false,
}: CorpusDocumentInspectorProps) {
    return (
        <CorpusDocumentInspectorContent
            key={document?.id ?? "none"}
            document={document}
            corpusId={corpusId}
            unitType={unitType}
            unitCountForType={unitCountForType}
            onRemoved={onRemoved}
            onSaved={onSaved}
            highlightSnippet={highlightSnippet}
            pageHint={pageHint}
            charStart={charStart}
            charEnd={charEnd}
            sourceSpanIds={sourceSpanIds}
            offsetScope={offsetScope}
            sourceCoordinate={sourceCoordinate}
            active={active}
            compactActions={compactActions}
        />
    );
}

function CorpusDocumentInspectorContent({
    document,
    corpusId,
    unitType,
    unitCountForType,
    onRemoved,
    onSaved,
    highlightSnippet,
    pageHint,
    charStart,
    charEnd,
    sourceSpanIds,
    offsetScope,
    sourceCoordinate,
    active,
    compactActions,
}: CorpusDocumentInspectorProps) {
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const [draft, setDraft] = useState<MetadataDraft | null>(() =>
        document ? toMetadataDraft(document) : null
    );

    const sourceQuery = useQuery({
        queryKey: ["text-research", "source-text", document?.id],
        queryFn: ({ signal }) => getSourceText(document!.id, signal),
        enabled: active && Boolean(document?.id),
    });

    const ragQuery = useQuery({
        queryKey: ["rag", "document", document?.rag_document_id],
        queryFn: () => getRagDocument(document!.rag_document_id!),
        enabled: active && Boolean(document?.rag_document_id),
        retry: false,
    });

    const saveMutation = useMutation({
        mutationFn: () => {
            if (!document || !draft) throw new Error("No document selected.");
            const year = draft.publication_year.trim();
            return updateDocument(document.id, {
                title: draft.title.trim() || undefined,
                organization: draft.organization.trim() || undefined,
                organization_type: draft.organization_type.trim() || undefined,
                publication_year: year ? Number(year) : null,
                publication_type: draft.publication_type.trim() || undefined,
                country: draft.country.trim() || undefined,
                region: draft.region.trim() || undefined,
                cultural_sphere: draft.cultural_sphere.trim() || undefined,
                language: draft.language.trim() || undefined,
                education_level: draft.education_level.trim() || undefined,
                source_url: draft.source_url.trim() || undefined,
                research_notes: draft.research_notes.trim() || undefined,
            });
        },
        onSuccess: (updated) => {
            void client.invalidateQueries({
                queryKey: ["text-research", "corpus", corpusId, "documents"],
            });
            showToast({ message: "Document metadata saved.", severity: "success" });
            onSaved?.(updated);
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to save metadata."),
                severity: "error",
            }),
    });

    const deleteMutation = useMutation({
        mutationFn: () => deleteDocument(document!.id),
        onSuccess: () => {
            void client.invalidateQueries({
                queryKey: ["text-research", "corpus", corpusId, "documents"],
            });
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.dashboard(corpusId),
            });
            showToast({ message: "Document removed from corpus.", severity: "success" });
            onRemoved?.();
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to remove document."),
                severity: "error",
            }),
    });

    function field(key: keyof MetadataDraft, label: string, multiline = false) {
        if (!draft) return null;
        return (
            <TextField
                size="small"
                label={label}
                value={draft[key]}
                onChange={(e) => setDraft({ ...draft, [key]: e.target.value })}
                fullWidth
                multiline={multiline}
                minRows={multiline ? 3 : undefined}
            />
        );
    }

    if (!document || !draft) {
        return (
            <Typography variant="body2" color="text.secondary">
                Select a document to inspect title, metadata, ingestion, and source text.
            </Typography>
        );
    }

    const ingestionStatus = ragQuery.data?.status ?? (document.rag_document_id ? "…" : "Not linked");
    const sourceLabel = draft.source_url.trim() || ragQuery.data?.original_filename || "—";

    return (
        <Stack spacing={1.5} sx={{ minHeight: 0 }}>
            <Stack spacing={1.25} sx={{ overflowY: "auto", flex: 1, pr: 0.5 }}>
                {field("title", "Title")}
                {field("organization", "Organization / author")}
                <FormGrid columns="6-6">
                    {field("publication_year", "Year")}
                    {field("language", "Language")}
                </FormGrid>
                {field("source_url", "Source")}
                {field("research_notes", "Research notes", true)}

                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                    <Typography variant="body2" color="text.secondary">
                        Ingestion
                    </Typography>
                    <Chip
                        size="small"
                        label={ingestionStatus}
                        color={
                            ingestionStatus === "indexed" || ingestionStatus === "completed"
                                ? "success"
                                : ingestionStatus === "failed"
                                  ? "error"
                                  : "default"
                        }
                        variant="outlined"
                    />
                </Stack>

                <KeyValueList
                    dense
                    showHelp={false}
                    items={[
                        {
                            key: "segmentation",
                            label: "Segmentation",
                            value:
                                unitType != null
                                    ? `${unitCountForType ?? 0} ${unitType}${
                                          (unitCountForType ?? 0) === 1 ? "" : "s"
                                      } in corpus`
                                    : "—",
                        },
                        {
                            key: "source_file",
                            label: "Source file",
                            value: sourceLabel,
                        },
                    ]}
                />

                <AdvancedSettings title="More metadata" description="Region, type, education, and related fields">
                    <Stack spacing={1.25}>
                        {field("organization_type", "Organization type")}
                        {field("publication_type", "Publication type")}
                        {field("country", "Country")}
                        {field("region", "Region")}
                        {field("cultural_sphere", "Cultural sphere")}
                        {field("education_level", "Education level")}
                    </Stack>
                </AdvancedSettings>

                <AdvancedSettings title="Provenance" description="Identifiers and timestamps">
                    <KeyValueList
                        dense
                        showHelp={false}
                        items={[
                            {
                                key: "doc_id",
                                label: "Document ID",
                                value: document.id,
                            },
                            {
                                key: "rag_id",
                                label: "RAG document",
                                value: document.rag_document_id ?? "—",
                            },
                            {
                                key: "created",
                                label: "Created",
                                value: new Date(document.created_at).toLocaleString(),
                            },
                            {
                                key: "updated",
                                label: "Updated",
                                value: new Date(document.updated_at).toLocaleString(),
                            },
                            ...(pageHint != null
                                ? [
                                      {
                                          key: "page",
                                          label: "Cited page",
                                          value: String(pageHint),
                                      },
                                  ]
                                : []),
                        ]}
                    />
                </AdvancedSettings>

                <Box>
                    <Typography variant="subtitle2" gutterBottom>
                        Source text
                    </Typography>
                    {sourceQuery.isError && highlightSnippet ? (
                        <Stack spacing={1}>
                            <Alert severity="warning">
                                Full source text unavailable (
                                {getQueryErrorMessage(sourceQuery.error, "canonical source missing")}
                                ). Showing citation snippet instead.
                            </Alert>
                            <HighlightedSourceText text={highlightSnippet} snippet={highlightSnippet} />
                        </Stack>
                    ) : (
                        <QueryBoundary
                            isLoading={sourceQuery.isLoading}
                            isError={sourceQuery.isError}
                            error={sourceQuery.error}
                            onRetry={() => void sourceQuery.refetch()}
                            variant="inline"
                        >
                            <HighlightedSourceText
                                text={
                                    sourceQuery.data?.text ||
                                    highlightSnippet ||
                                    "No source text available yet."
                                }
                                snippet={highlightSnippet}
                                charStart={charStart}
                                charEnd={charEnd}
                                pageNumber={pageHint}
                                pageProvenance={sourceQuery.data?.page_provenance}
                                sourceSpanIds={sourceSpanIds}
                                offsetScope={offsetScope}
                                sourceCoordinate={sourceCoordinate}
                            />
                        </QueryBoundary>
                    )}
                    {draft.source_url.trim() ? (
                        <Link href={draft.source_url.trim()} target="_blank" rel="noreferrer" variant="caption">
                            Open source URL
                        </Link>
                    ) : null}
                </Box>
            </Stack>

            <Stack direction={compactActions ? "column" : "row"} spacing={1}>
                <Button
                    variant="contained"
                    onClick={() => saveMutation.mutate()}
                    disabled={saveMutation.isPending}
                    fullWidth={compactActions}
                >
                    Save metadata
                </Button>
                <Button
                    color="error"
                    onClick={() => {
                        if (window.confirm("Remove this document from the corpus?")) {
                            deleteMutation.mutate();
                        }
                    }}
                    disabled={deleteMutation.isPending}
                    fullWidth={compactActions}
                >
                    Remove
                </Button>
            </Stack>
        </Stack>
    );
}

export function HighlightedSourceText({
    text,
    snippet,
    charStart,
    charEnd,
    pageNumber,
    pageProvenance,
    sourceSpanIds,
    offsetScope,
    sourceCoordinate,
}: {
    text: string;
    snippet?: string | null;
    charStart?: number | null;
    charEnd?: number | null;
    pageNumber?: number | null;
    pageProvenance?: PageProvenance[];
    sourceSpanIds?: string[] | null;
    offsetScope?: "parsed_document" | "page" | "canonical_document" | null;
    sourceCoordinate?: CitationCoordinate;
}) {
    const markRef = useRef<HTMLElement | null>(null);
    const needle = (snippet ?? "").trim();
    const lowerText = text.toLowerCase();
    const lowerNeedle = needle.toLowerCase();
    const exactRange = resolveCitationHighlightRange({
        text,
        charStart,
        charEnd,
        pageNumber,
        pageProvenance,
        sourceSpanIds,
        offsetScope,
        offsetScopeId: sourceCoordinate?.offset_scope_id,
        offsetCoordinateSystem: sourceCoordinate?.offset_coordinate_system,
        sourceSpans: sourceCoordinate?.source_spans,
    });
    const index = exactRange?.start ?? (needle ? lowerText.indexOf(lowerNeedle) : -1);
    const matchEnd = exactRange?.end ?? (index >= 0 ? index + needle.length : -1);

    useEffect(() => {
        markRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
    }, [text, snippet, exactRange?.start, exactRange?.end]);

    const preSx = {
        m: 0,
        p: 1.5,
        borderRadius: 1,
        bgcolor: "action.hover",
        fontSize: 12,
        whiteSpace: "pre-wrap",
        maxHeight: 240,
        overflow: "auto",
    } as const;

    if (index < 0) {
        return (
            <Box component="pre" sx={preSx}>
                {text}
            </Box>
        );
    }

    const before = text.slice(0, index);
    const match = text.slice(index, matchEnd);
    const after = text.slice(matchEnd);

    return (
        <Box component="pre" sx={preSx}>
            {before}
            <Box
                component="mark"
                ref={markRef}
                sx={{
                    bgcolor: "warning.light",
                    color: "warning.contrastText",
                    px: 0.25,
                    borderRadius: 0.5,
                }}
            >
                {match}
            </Box>
            {after}
        </Box>
    );
}
