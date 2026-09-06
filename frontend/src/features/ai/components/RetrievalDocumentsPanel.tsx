import { Box, Button, Stack, TextField, Typography } from "@mui/material";
import { Description as DocumentIcon } from "@mui/icons-material";
import { EmptyState } from "../../../components/ui/EmptyState";
import { SectionCard } from "../../../components/ui/SectionCard";
import { formatDateTime } from "../../../utils/formatters";
import type { AiStudioModel } from "../hooks/useAiStudioView";

export function RetrievalDocumentsPanel({ m }: { m: AiStudioModel }) {
    const { textDocumentForm, setTextDocumentForm, createTextDocumentMutation, uploadDocumentMutation, uploadDescription, setUploadDescription, documents } = m;
    return (
    <SectionCard title="Retrieval documents" description="Ingest source files or direct text, chunk them, and use them as retrieval context in prompt runs.">
        <Stack spacing={2}>
            <TextField label="Document title" value={textDocumentForm.title} onChange={(event) => setTextDocumentForm((current) => ({ ...current, title: event.target.value }))} fullWidth />
            <TextField label="Description" value={textDocumentForm.description} onChange={(event) => setTextDocumentForm((current) => ({ ...current, description: event.target.value }))} fullWidth />
            <TextField label="Document content" value={textDocumentForm.content} onChange={(event) => setTextDocumentForm((current) => ({ ...current, content: event.target.value }))} fullWidth multiline minRows={6} />
            <Button
                variant="outlined"
                disabled={createTextDocumentMutation.isPending || !textDocumentForm.title.trim() || !textDocumentForm.content.trim()}
                onClick={() => createTextDocumentMutation.mutate(textDocumentForm)}
            >
                {createTextDocumentMutation.isPending ? "Ingesting..." : "Create text document"}
            </Button>
            <Button component="label" variant="contained" disabled={uploadDocumentMutation.isPending}>
                {uploadDocumentMutation.isPending ? "Uploading..." : "Upload text/markdown/json file"}
                <input
                    hidden
                    type="file"
                    accept=".txt,.md,.json,.ndjson,text/plain,text/markdown,application/json"
                    onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (file) {
                            uploadDocumentMutation.mutate({ file, description: uploadDescription || undefined });
                        }
                        event.currentTarget.value = "";
                    }}
                />
            </Button>
            <TextField label="Upload description" value={uploadDescription} onChange={(event) => setUploadDescription(event.target.value)} fullWidth />
            {documents.length > 0 ? (
                <Stack spacing={1}>
                    {documents.map((document) => (
                        <Box key={document.id} sx={(theme) => ({ p: 1.5, borderRadius: 3, border: `1px solid ${theme.palette.divider}` })}>
                            <Typography variant="subtitle2">{document.title}</Typography>
                            <Typography variant="caption" color="text.secondary">
                                {document.chunk_count} chunks • {document.content_type} • {formatDateTime(document.updated_at)}
                            </Typography>
                        </Box>
                    ))}
                </Stack>
            ) : (
                <EmptyState icon={<DocumentIcon />} title="No documents indexed" description="Upload source material to power retrieval-augmented runs." />
            )}
        </Stack>
    </SectionCard>
    );
}

