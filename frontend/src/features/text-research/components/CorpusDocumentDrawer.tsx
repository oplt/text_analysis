import { useEffect, useRef, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Divider,
    Drawer,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { deleteDocument, getSourceText, updateDocument } from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import type { CorpusDocument } from "../types";
import {
    resolveCitationHighlightRange,
    type PageProvenance,
} from "./citationOffsetResolver";

type MetadataDraft = {
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

function toDraft(doc: CorpusDocument): MetadataDraft {
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

type CorpusDocumentDrawerProps = {
    open: boolean;
    document: CorpusDocument | null;
    corpusId: string;
    onClose: () => void;
    highlightSnippet?: string | null;
    /** Optional page number hint from a citation. */
    pageHint?: number | null;
    charStart?: number | null;
    charEnd?: number | null;
    sourceSpanIds?: string[] | null;
    offsetScope?: "parsed_document" | "page" | "canonical_document" | null;
};

export function CorpusDocumentDrawer({
    document,
    ...props
}: CorpusDocumentDrawerProps) {
    return <CorpusDocumentDrawerContent key={document?.id ?? "none"} document={document} {...props} />;
}

function CorpusDocumentDrawerContent({
    open,
    document,
    corpusId,
    onClose,
    highlightSnippet,
    pageHint,
    charStart,
    charEnd,
    sourceSpanIds,
    offsetScope,
}: CorpusDocumentDrawerProps) {
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const [draft, setDraft] = useState<MetadataDraft | null>(() =>
        document ? toDraft(document) : null
    );

    const sourceQuery = useQuery({
        queryKey: ["text-research", "source-text", document?.id],
        queryFn: () => getSourceText(document!.id),
        enabled: open && Boolean(document?.id),
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
        onSuccess: () => {
            void client.invalidateQueries({
                queryKey: ["text-research", "corpus", corpusId, "documents"],
            });
            showToast({ message: "Document metadata saved.", severity: "success" });
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
            onClose();
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to remove document."),
                severity: "error",
            }),
    });

    function field(
        key: keyof MetadataDraft,
        label: string,
        multiline = false
    ) {
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

    return (
        <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx: { width: { xs: "100%", sm: 420 } } }}>
            <Box sx={{ p: 2.5, display: "flex", flexDirection: "column", gap: 2, height: "100%" }}>
                <Typography variant="h6">Document details</Typography>
                {!document || !draft ? (
                    <Typography color="text.secondary">Select a document from the table.</Typography>
                ) : (
                    <>
                        <Stack spacing={1.5} sx={{ overflowY: "auto", flex: 1 }}>
                            {field("title", "Title")}
                            {field("organization", "Organization")}
                            {field("organization_type", "Organization type")}
                            {field("publication_year", "Publication year")}
                            {field("publication_type", "Publication type")}
                            {field("country", "Country")}
                            {field("region", "Region")}
                            {field("cultural_sphere", "Cultural sphere")}
                            {field("language", "Language")}
                            {field("education_level", "Education level")}
                            {field("source_url", "Source URL")}
                            {field("research_notes", "Research notes", true)}
                            <Typography variant="caption" color="text.secondary">
                                RAG document: {document.rag_document_id}
                                {pageHint != null ? ` · Cited page ~${pageHint}` : ""}
                            </Typography>
                            <Divider />
                            <Typography variant="subtitle2">Source text</Typography>
                            {sourceQuery.isError && highlightSnippet ? (
                                <Stack spacing={1}>
                                    <Alert severity="warning">
                                        Full source text unavailable (
                                        {getQueryErrorMessage(
                                            sourceQuery.error,
                                            "canonical source missing"
                                        )}
                                        ). Showing citation snippet instead.
                                    </Alert>
                                    <HighlightedSourceText
                                        text={highlightSnippet}
                                        snippet={highlightSnippet}
                                    />
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
                                    />
                                </QueryBoundary>
                            )}
                        </Stack>
                        <Stack direction="row" spacing={1}>
                            <Button
                                variant="contained"
                                onClick={() => saveMutation.mutate()}
                                disabled={saveMutation.isPending}
                            >
                                Save metadata
                            </Button>
                            <Button
                                color="error"
                                onClick={() => deleteMutation.mutate()}
                                disabled={deleteMutation.isPending}
                            >
                                Remove
                            </Button>
                            <Button onClick={onClose}>Close</Button>
                        </Stack>
                    </>
                )}
            </Box>
        </Drawer>
    );
}

function HighlightedSourceText({
    text,
    snippet,
    charStart,
    charEnd,
    pageNumber,
    pageProvenance,
    sourceSpanIds,
    offsetScope,
}: {
    text: string;
    snippet?: string | null;
    charStart?: number | null;
    charEnd?: number | null;
    pageNumber?: number | null;
    pageProvenance?: PageProvenance[];
    sourceSpanIds?: string[] | null;
    offsetScope?: "parsed_document" | "page" | "canonical_document" | null;
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
    });
    const index = exactRange?.start ?? (needle ? lowerText.indexOf(lowerNeedle) : -1);
    const matchEnd = exactRange?.end ?? (index >= 0 ? index + needle.length : -1);

    useEffect(() => {
        markRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
    }, [text, snippet, exactRange?.start, exactRange?.end]);

    if (index < 0) {
        return (
            <Box
                component="pre"
                sx={{
                    m: 0,
                    p: 1.5,
                    borderRadius: 1,
                    bgcolor: "action.hover",
                    fontSize: 12,
                    whiteSpace: "pre-wrap",
                    maxHeight: 240,
                    overflow: "auto",
                }}
            >
                {text}
            </Box>
        );
    }

    const before = text.slice(0, index);
    const match = text.slice(index, matchEnd);
    const after = text.slice(matchEnd);

    return (
        <Box
            component="pre"
            sx={{
                m: 0,
                p: 1.5,
                borderRadius: 1,
                bgcolor: "action.hover",
                fontSize: 12,
                whiteSpace: "pre-wrap",
                maxHeight: 240,
                overflow: "auto",
            }}
        >
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
