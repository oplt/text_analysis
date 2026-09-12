import { Alert, Chip, Stack, Typography } from "@mui/material";
import type { AssistantCitation, AssistantCoverage, AssistantScope } from "../../../../api/textResearch";

type Props = {
    scope: AssistantScope;
    coverage: AssistantCoverage;
    evidenceRevisionHash?: string | null;
    degraded: boolean;
    degradationReason?: string | null;
    noResults: boolean;
    citations: AssistantCitation[];
    synthesisOmissions?: Array<{ rag_document_id: string; reason: string }>;
};

export function EvidenceProvenance({
    scope,
    coverage,
    evidenceRevisionHash,
    degraded,
    degradationReason,
    noResults,
    citations,
    synthesisOmissions = [],
}: Props) {
    const used = citations.filter((citation) => citation.used_in_answer).length;
    const unavailable = citations.filter(
        (citation) => !citation.offset_scope || !citation.source_span_ids?.length
    ).length;

    return (
        <Stack spacing={0.75} aria-label="Evidence provenance">
            <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                <Chip
                    size="small"
                    label={scope.scope_mode === "fixed" ? "Fixed snapshot" : "Live corpus"}
                    color={scope.scope_mode === "fixed" ? "primary" : "default"}
                />
                <Chip
                    size="small"
                    variant="outlined"
                    label={`Revision ${evidenceRevisionHash?.slice(0, 12) ?? "unavailable"}`}
                />
                <Chip
                    size="small"
                    variant="outlined"
                    label={`${coverage.documents_with_retrieved_evidence}/${coverage.documents_in_scope} documents`}
                />
            </Stack>
            <Typography variant="caption" color="text.secondary">
                {used} cited · {citations.length - used} retrieved only · {scope.unavailable_count} unavailable in scope
            </Typography>
            {noResults ? <Alert severity="info">No relevant evidence was found in this snapshot.</Alert> : null}
            {degraded ? (
                <Alert severity="warning">
                    Retrieval was degraded{degradationReason ? `: ${degradationReason}` : "."}
                </Alert>
            ) : null}
            {unavailable ? (
                <Alert severity="warning">
                    {unavailable} citation{unavailable === 1 ? " is" : "s are"} not precisely resolvable; opening one will not substitute a newer revision.
                </Alert>
            ) : null}
            {synthesisOmissions.length ? (
                <Alert severity="info">
                    Synthesis omitted {synthesisOmissions.length} document{synthesisOmissions.length === 1 ? "" : "s"} due to unavailable evidence or map failures.
                </Alert>
            ) : null}
        </Stack>
    );
}
