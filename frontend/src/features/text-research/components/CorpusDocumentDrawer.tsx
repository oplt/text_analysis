import { Box, Drawer, Typography } from "@mui/material";
import type { CorpusDocument } from "../types";
import { CorpusDocumentInspector } from "./CorpusDocumentInspector";

type CorpusDocumentDrawerProps = {
    open: boolean;
    document: CorpusDocument | null;
    corpusId: string;
    onClose: () => void;
    highlightSnippet?: string | null;
    pageHint?: number | null;
    charStart?: number | null;
    charEnd?: number | null;
    sourceSpanIds?: string[] | null;
    offsetScope?: "parsed_document" | "page" | "canonical_document" | null;
    sourceCoordinate?: Pick<
        import("../../../api/textResearch").AssistantCitation,
        "offset_scope_id" | "offset_coordinate_system" | "source_spans"
    > | null;
};

/**
 * Mobile / citation overlay for document inspection.
 * Desktop corpus workspace uses the inline ContextInspector instead.
 */
export function CorpusDocumentDrawer({
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
    sourceCoordinate,
}: CorpusDocumentDrawerProps) {
    return (
        <Drawer
            anchor="right"
            open={open}
            onClose={onClose}
            PaperProps={{ sx: { width: { xs: "100%", sm: 440 } } }}
        >
            <Box
                sx={{
                    p: 2.5,
                    display: "flex",
                    flexDirection: "column",
                    gap: 2,
                    height: "100%",
                    minHeight: 0,
                }}
            >
                <Typography variant="h6">Document details</Typography>
                <Box sx={{ flex: 1, minHeight: 0, overflow: "hidden", display: "flex" }}>
                    <CorpusDocumentInspector
                        document={document}
                        corpusId={corpusId}
                        active={open}
                        compactActions
                        highlightSnippet={highlightSnippet}
                        pageHint={pageHint}
                        charStart={charStart}
                        charEnd={charEnd}
                        sourceSpanIds={sourceSpanIds}
                        offsetScope={offsetScope}
                        sourceCoordinate={sourceCoordinate}
                        onRemoved={onClose}
                    />
                </Box>
            </Box>
        </Drawer>
    );
}
