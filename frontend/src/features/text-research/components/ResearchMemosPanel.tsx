import { useState } from "react";
import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Alert,
    Box,
    Button,
    Checkbox,
    Divider,
    FormControlLabel,
    Link,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
    archiveResearchMemo,
    listResearchMemos,
    updateResearchMemo,
    type AssistantCitation,
} from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useResearchContext } from "../hooks/useResearchContext";

type Props = { onOpenCitation?: (citation: AssistantCitation) => void };

export function ResearchMemosPanel({ onOpenCitation }: Props) {
    const ctx = useResearchContext();
    const projectId = ctx.projectId ?? "";
    const corpusId = ctx.selectedCorpus?.id;
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [includeArchived, setIncludeArchived] = useState(false);
    const [editing, setEditing] = useState(false);
    const [draftTitle, setDraftTitle] = useState("");
    const [draftBody, setDraftBody] = useState("");
    const queryClient = useQueryClient();
    const memosQuery = useQuery({
        queryKey: [...queryKeys.textResearch.memos(projectId, corpusId), includeArchived],
        queryFn: () => listResearchMemos(projectId, corpusId, includeArchived),
        enabled: Boolean(projectId),
    });
    const invalidateMemos = () =>
        queryClient.invalidateQueries({ queryKey: queryKeys.textResearch.memos(projectId, corpusId) });
    const archiveMutation = useMutation({
        mutationFn: (memoId: string) => archiveResearchMemo(memoId),
        onSuccess: () => {
            void invalidateMemos();
            setSelectedId(null);
            setEditing(false);
        },
    });
    const updateMutation = useMutation({
        mutationFn: (memoId: string) =>
            updateResearchMemo(memoId, { title: draftTitle.trim(), body: draftBody }),
        onSuccess: () => {
            void invalidateMemos();
            setEditing(false);
        },
    });

    const memos = memosQuery.data ?? [];
    const selected = memos.find((memo) => memo.id === selectedId) ?? memos[0] ?? null;
    if (!projectId) return null;

    return (
        <Accordion disableGutters sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Research memos ({memos.length})</Typography>
            </AccordionSummary>
            <AccordionDetails>
                <Stack spacing={1}>
                    <FormControlLabel
                        control={
                            <Checkbox
                                size="small"
                                checked={includeArchived}
                                onChange={(event) => setIncludeArchived(event.target.checked)}
                            />
                        }
                        label={<Typography variant="caption">Show archived memos</Typography>}
                    />
                    {memosQuery.isLoading ? <Typography variant="body2">Loading memos…</Typography> : null}
                    {memosQuery.isError ? (
                        <Alert severity="error">
                            Could not load memos: {getQueryErrorMessage(memosQuery.error)}
                        </Alert>
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
                                        onClick={() => {
                                            setSelectedId(memo.id);
                                            setEditing(false);
                                        }}
                                        sx={{ justifyContent: "flex-start", textAlign: "left" }}
                                    >
                                        {memo.title}{memo.archived_at ? " · archived" : ""}
                                    </Button>
                                ))}
                            </Stack>
                            {selected ? (
                                <Box sx={{ minWidth: 0, flex: 1 }}>
                                    {editing ? (
                                        <Stack spacing={1}>
                                            <TextField
                                                label="Title"
                                                size="small"
                                                value={draftTitle}
                                                onChange={(event) => setDraftTitle(event.target.value)}
                                            />
                                            <TextField
                                                label="Memo"
                                                multiline
                                                minRows={5}
                                                value={draftBody}
                                                onChange={(event) => setDraftBody(event.target.value)}
                                            />
                                            <Stack direction="row" spacing={1}>
                                                <Button
                                                    variant="contained"
                                                    onClick={() => updateMutation.mutate(selected.id)}
                                                    disabled={!draftTitle.trim() || updateMutation.isPending}
                                                >
                                                    Save edits
                                                </Button>
                                                <Button onClick={() => setEditing(false)}>Cancel</Button>
                                            </Stack>
                                        </Stack>
                                    ) : (
                                        <>
                                            <Typography variant="subtitle1" fontWeight={600}>
                                                {selected.title}
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                {selected.source_type.replaceAll("_", " ")} · evidence revision {selected.evidence_revision_hash?.slice(0, 12) ?? "—"}
                                            </Typography>
                                            <Typography sx={{ whiteSpace: "pre-wrap", mt: 1 }}>
                                                {selected.body}
                                            </Typography>
                                        </>
                                    )}
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
                                                    </Link>
                                                );
                                            })}
                                        </>
                                    ) : null}
                                    {!editing ? (
                                        <Stack direction="row" spacing={1} mt={1}>
                                            {!selected.archived_at ? (
                                                <>
                                                    <Button
                                                        size="small"
                                                        onClick={() => {
                                                            setDraftTitle(selected.title);
                                                            setDraftBody(selected.body);
                                                            setEditing(true);
                                                        }}
                                                    >
                                                        Edit memo
                                                    </Button>
                                                    <Button
                                                        size="small"
                                                        color="error"
                                                        onClick={() => archiveMutation.mutate(selected.id)}
                                                        disabled={archiveMutation.isPending}
                                                    >
                                                        Archive memo
                                                    </Button>
                                                </>
                                            ) : null}
                                        </Stack>
                                    ) : null}
                                </Box>
                            ) : null}
                        </Stack>
                    ) : null}
                </Stack>
            </AccordionDetails>
        </Accordion>
    );
}
