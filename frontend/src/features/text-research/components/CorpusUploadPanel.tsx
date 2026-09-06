import {
    Box,
    Button,
    Chip,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Typography,
} from "@mui/material";
import { UploadFile as UploadIcon } from "@mui/icons-material";
import { useRef, useState } from "react";
import {
    CORPUS_UPLOAD_ACCEPT,
    uploadFilesToCorpus,
    type CorpusUploadRow,
    type UploadStepStatus,
} from "../corpusUpload";

function StepCell({ status }: { status: UploadStepStatus }) {
    if (status === "done") return <Chip size="small" color="success" label="✓" />;
    if (status === "processing") return <Chip size="small" color="warning" label="processing" />;
    if (status === "error") return <Chip size="small" color="error" label="failed" />;
    if (status === "skipped") return <Chip size="small" label="—" />;
    return <Typography variant="caption" color="text.secondary">—</Typography>;
}

type CorpusUploadPanelProps = {
    corpusId: string;
    disabled?: boolean;
    onComplete?: () => void;
};

export function CorpusUploadPanel({ corpusId, disabled, onComplete }: CorpusUploadPanelProps) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [rows, setRows] = useState<CorpusUploadRow[]>([]);
    const [uploading, setUploading] = useState(false);

    async function handleFiles(fileList: FileList | null) {
        if (!fileList?.length || !corpusId) return;
        const files = Array.from(fileList);
        setUploading(true);
        try {
            await uploadFilesToCorpus(files, corpusId, setRows);
            onComplete?.();
        } finally {
            setUploading(false);
            if (inputRef.current) inputRef.current.value = "";
        }
    }

    return (
        <Stack spacing={1.5}>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
                <Button
                    variant="contained"
                    startIcon={<UploadIcon />}
                    disabled={disabled || uploading || !corpusId}
                    onClick={() => inputRef.current?.click()}
                >
                    {uploading ? "Uploading…" : "Upload documents"}
                </Button>
                <Typography variant="caption" color="text.secondary">
                    PDF, DOCX, TXT, Markdown, CSV — each file is ingested via RAG, then linked to this corpus.
                </Typography>
                <Box
                    component="input"
                    ref={inputRef}
                    type="file"
                    multiple
                    accept={CORPUS_UPLOAD_ACCEPT}
                    hidden
                    onChange={(event) => void handleFiles(event.target.files)}
                />
            </Stack>

            {rows.length > 0 ? (
                <Box sx={{ overflowX: "auto" }}>
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Document</TableCell>
                                <TableCell>Parse</TableCell>
                                <TableCell>Ingest</TableCell>
                                <TableCell>Added to corpus</TableCell>
                                <TableCell>Error</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {rows.map((row) => (
                                <TableRow key={row.id}>
                                    <TableCell>{row.fileName}</TableCell>
                                    <TableCell>
                                        <StepCell status={row.parse} />
                                    </TableCell>
                                    <TableCell>
                                        <StepCell status={row.ingest} />
                                    </TableCell>
                                    <TableCell>
                                        <StepCell status={row.added} />
                                    </TableCell>
                                    <TableCell>
                                        <Typography variant="caption" color="error">
                                            {row.error ?? ""}
                                        </Typography>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </Box>
            ) : null}
        </Stack>
    );
}
