import { useState } from "react";
import { Alert, Box, Button, Divider, Link, Stack, Typography } from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { archiveResearchMemo, listResearchMemos, type AssistantCitation } from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useResearchContext } from "../hooks/useResearchContext";

type Props = { onOpenCitation?: (citation: AssistantCitation) => void };

export function ResearchMemosPanel({ onOpenCitation }: Props) {
    const ctx = useResearchContext();
    const projectId = ctx.projectId ?? "";
    const corpusId = ctx.selectedCorpus?.id;
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const queryClient = useQueryClient();
    const memosQuery = useQuery({
        queryKey: queryKeys.textResearch.memos(projectId, corpusId),
        queryFn: () => listResearchMemos(projectId, corpusId),
        enabled: Boolean(projectId),
    });
    const archiveMutation = useMutation({
        mutationFn: (memoId: string) => archiveResearchMemo(memoId),
        onSuccess: () => {
            void queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.memos(projectId, corpusId),
            });
            setSelectedId(null);
        },
    });

    const memos = memosQuery.data ?? [];
    const selected = memos.find((memo) => memo.id === selectedId) ?? memos[0] ?? null;
    if (!projectId) return null;

    return (
        <Box sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom>Research memos</Typography>
            {memosQuery.isLoading ? <Typography variant="body2">Loading memos…</Typography> : null}
            {memosQuery.isError ? (
                <Alert severity="error">Could not load memos: {getQueryErrorMessage(memosQuery.error)}</Alert>
            ) : null}
            {!memosQuery.isLoading && !memos.length ? (
                <Alert severity="info">Save an Ask Corpus answer or synthesis to create a memo.</Alert>
            ) : null}
            {memos.length ? (
                <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
                    <Stack spacing={0.5} sx={{ minWidth: 220 }}>
                        {memos.map((memo) => (
                            <Button
                                key={memo.id}
                                variant={memo.id === selected?.id ? "contained" : "text"}
                                onClick={() => setSelectedId(memo.id)}
                                sx={{ justifyContent: "flex-start", textAlign: "left" }}
                            >
                                {memo.title}
                            </Button>
                        ))}
                    </Stack>
                    {selected ? (
                        <Box sx={{ minWidth: 0, flex: 1 }}>
                            <Typography variant="subtitle1" fontWeight={600}>{selected.title}</Typography>
                            <Typography variant="caption" color="text.secondary">
                                {selected.source_type.replaceAll("_", " ")} · evidence revision {selected.evidence_revision_hash?.slice(0, 12) ?? "—"}
                            </Typography>
                            <Typography sx={{ whiteSpace: "pre-wrap", mt: 1 }}>{selected.body}</Typography>
                            {selected.citations.length ? (
                                <>
                                    <Divider sx={{ my: 1 }} />
                                    <Typography variant="subtitle2">Citations</Typography>
                                    {selected.citations.map((rawCitation, index) => {
                                        const citation = rawCitation as unknown as AssistantCitation;
                                        return (
                                            <Link
                                                key={`${String(rawCitation.chunk_id ?? index)}`}
                                                component="button"
                                                type="button"
                                                variant="body2"
                                                onClick={() => onOpenCitation?.(citation)}
                                                sx={{ display: "block", textAlign: "left" }}
                                            >
                                                [{citation.citation_number ?? index + 1}] {citation.filename ?? "Source"}
                                                {citation.page_number != null ? ` · p.${citation.page_number}` : ""}
                                            </Link>
                                        );
                                    })}
                                </>
                            ) : null}
                            <Button
                                size="small"
                                color="error"
                                sx={{ mt: 1 }}
                                onClick={() => archiveMutation.mutate(selected.id)}
                                disabled={archiveMutation.isPending}
                            >
                                Archive memo
                            </Button>
                        </Box>
                    ) : null}
                </Stack>
            ) : null}
        </Box>
    );
}
