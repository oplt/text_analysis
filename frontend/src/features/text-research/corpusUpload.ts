import { getAiDocument, uploadAiDocument } from "../../api/ai";
import { addDocument } from "../../api/textResearch";
import { getQueryErrorMessage } from "../../utils/queryErrors";

export const CORPUS_UPLOAD_ACCEPT =
    ".pdf,.docx,.txt,.md,.markdown,.csv,application/pdf,text/plain,text/markdown,text/csv,application/vnd.openxmlformats-officedocument.wordprocessingml.document";

export type UploadStepStatus = "idle" | "processing" | "done" | "error" | "skipped";

export type CorpusUploadRow = {
    id: string;
    fileName: string;
    parse: UploadStepStatus;
    ingest: UploadStepStatus;
    added: UploadStepStatus;
    error: string | null;
    ragDocumentId: string | null;
};

function sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForIngestion(
    documentId: string,
    onStatus: (status: string) => void
): Promise<void> {
    for (let attempt = 0; attempt < 90; attempt += 1) {
        const document = await getAiDocument(documentId);
        onStatus(document.ingestion_status);
        if (document.ingestion_status === "completed") return;
        if (document.ingestion_status === "failed") {
            throw new Error("RAG ingestion failed for this document.");
        }
        await sleep(1500);
    }
    throw new Error("Timed out waiting for document ingestion.");
}

export async function uploadFileToCorpus(
    file: File,
    corpusId: string,
    onUpdate: (patch: Partial<CorpusUploadRow>) => void
): Promise<void> {
    let stage: "parse" | "ingest" | "added" = "parse";
    onUpdate({ parse: "processing", ingest: "idle", added: "idle", error: null });
    try {
        const uploaded = await uploadAiDocument(file);
        onUpdate({
            parse: "done",
            ingest: "processing",
            ragDocumentId: uploaded.id,
        });
        stage = "ingest";

        if (uploaded.ingestion_status !== "completed") {
            await waitForIngestion(uploaded.id, (status) => {
                if (status === "completed") {
                    onUpdate({ ingest: "done" });
                } else if (status !== "failed") {
                    onUpdate({ ingest: "processing" });
                }
            });
        }
        onUpdate({ ingest: "done", added: "processing" });
        stage = "added";

        await addDocument(corpusId, {
            rag_document_id: uploaded.id,
            title: uploaded.title || file.name,
        });
        onUpdate({ added: "done" });
    } catch (error) {
        const message = getQueryErrorMessage(error, "Upload failed.");
        const patch: Partial<CorpusUploadRow> = {
            error: typeof message === "string" ? message : "Upload failed.",
        };
        if (stage === "parse") patch.parse = "error";
        else if (stage === "ingest") patch.ingest = "error";
        else patch.added = "error";
        onUpdate(patch);
    }
}

export async function uploadFilesToCorpus(
    files: File[],
    corpusId: string,
    onRowsChange: (rows: CorpusUploadRow[]) => void
): Promise<CorpusUploadRow[]> {
    const rows: CorpusUploadRow[] = files.map((file, index) => ({
        id: `${file.name}-${file.size}-${index}-${Date.now()}`,
        fileName: file.name,
        parse: "idle",
        ingest: "idle",
        added: "idle",
        error: null,
        ragDocumentId: null,
    }));
    onRowsChange([...rows]);

    for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        const updateRow = (patch: Partial<CorpusUploadRow>) => {
            rows[index] = { ...rows[index], ...patch };
            onRowsChange([...rows]);
        };
        await uploadFileToCorpus(file, corpusId, updateRow);
    }

    return rows;
}
