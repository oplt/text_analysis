import { useState } from "react";
import {
    Alert,
    Button,
    Checkbox,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    FormControlLabel,
    FormGroup,
    Stack,
    ToggleButton,
    ToggleButtonGroup,
    Typography,
} from "@mui/material";
import { useMutation } from "@tanstack/react-query";
import {
    updateAssistantThreadScope,
    type AssistantScope,
} from "../../../../api/textResearch";
import { getQueryErrorMessage } from "../../../../utils/queryErrors";

type ScopeUpdate = {
    document_ids: string[] | null;
    scope_mode: "fixed" | "live";
    reason: string;
};

type Props = {
    threadId: string | null;
    scope: AssistantScope | null;
    liveScope: AssistantScope | null;
    onUpdated: (scope: AssistantScope) => void;
};

function difference(left: string[], right: string[]) {
    const rightIds = new Set(right);
    return left.filter((id) => !rightIds.has(id));
}

export function AssistantScopeControls({ threadId, scope, liveScope, onUpdated }: Props) {
    const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>(
        () => scope?.corpus_document_ids ?? []
    );
    const [pendingUpdate, setPendingUpdate] = useState<ScopeUpdate | null>(null);

    const updateMutation = useMutation({
        mutationFn: (update: ScopeUpdate) => updateAssistantThreadScope(threadId!, update),
        onSuccess: (updatedScope) => {
            setPendingUpdate(null);
            onUpdated(updatedScope);
        },
    });

    const availableDocuments = liveScope?.document_bindings ?? scope?.document_bindings ?? [];
    const previousDocumentIds = scope?.corpus_document_ids ?? [];
    const liveDocumentIds = liveScope?.corpus_document_ids ?? [];
    const addedDocumentIds = difference(liveDocumentIds, previousDocumentIds);
    const removedDocumentIds = difference(previousDocumentIds, liveDocumentIds);

    if (!scope) return null;

    const selectDocument = (documentId: string, selected: boolean) => {
        setSelectedDocumentIds((current) =>
            selected
                ? [...new Set([...current, documentId])]
                : current.filter((id) => id !== documentId)
        );
    };
    const requestUpdate = (update: ScopeUpdate) => {
        if (!threadId) return;
        setPendingUpdate(update);
    };
    const useCurrentFixedScope = () =>
        requestUpdate({
            document_ids: liveDocumentIds.every((id) => selectedDocumentIds.includes(id))
                ? null
                : selectedDocumentIds,
            scope_mode: "fixed",
            reason: "User updated the thread to an explicit corpus evidence snapshot.",
        });

    return (
        <Stack spacing={1}>
            <Typography variant="subtitle2">Evidence scope</Typography>
            <ToggleButtonGroup
                exclusive
                size="small"
                fullWidth
                value={scope.scope_mode}
                onChange={(_, value: "fixed" | "live" | null) => {
                    if (!value || value === scope.scope_mode) return;
                    requestUpdate({
                        document_ids: value === "live" ? null : selectedDocumentIds,
                        scope_mode: value,
                        reason:
                            value === "live"
                                ? "User switched the thread to the live corpus."
                                : "User pinned the thread to a fixed corpus snapshot.",
                    });
                }}
                aria-label="Evidence scope mode"
                disabled={!threadId || updateMutation.isPending}
            >
                <ToggleButton value="fixed">Fixed snapshot</ToggleButton>
                <ToggleButton value="live">Live corpus</ToggleButton>
            </ToggleButtonGroup>
            <Typography variant="caption" color="text.secondary">
                {scope.scope_mode === "live"
                    ? "Future turns resolve the current corpus and can change when documents are added or reindexed."
                    : "This conversation uses an explicit corpus evidence snapshot until you update it."}
            </Typography>
            {!threadId ? (
                <Alert severity="info">
                    Create a thread to choose and preserve an evidence scope for future turns.
                </Alert>
            ) : scope.scope_mode === "fixed" ? (
                <>
                    <Typography variant="caption" color="text.secondary">
                        Optional document subset
                    </Typography>
                    <FormGroup>
                        {availableDocuments.map((binding) => (
                            <FormControlLabel
                                key={binding.corpus_document_id}
                                control={
                                    <Checkbox
                                        size="small"
                                        checked={selectedDocumentIds.includes(binding.corpus_document_id)}
                                        onChange={(event) =>
                                            selectDocument(
                                                binding.corpus_document_id,
                                                event.target.checked
                                            )
                                        }
                                    />
                                }
                                label={
                                    <Typography variant="caption">
                                        {binding.corpus_document_id}
                                        {binding.availability === "unavailable"
                                            ? ` · unavailable${binding.unavailable_reason ? `: ${binding.unavailable_reason}` : ""}`
                                            : ""}
                                    </Typography>
                                }
                            />
                        ))}
                    </FormGroup>
                    <Button
                        size="small"
                        variant="outlined"
                        onClick={useCurrentFixedScope}
                        disabled={updateMutation.isPending}
                    >
                        Update fixed snapshot
                    </Button>
                </>
            ) : null}
            {updateMutation.isError ? (
                <Alert severity="error">
                    {getQueryErrorMessage(updateMutation.error, "Thread scope update failed.")}
                </Alert>
            ) : null}

            <Dialog open={Boolean(pendingUpdate)} onClose={() => setPendingUpdate(null)}>
                <DialogTitle>Change evidence scope?</DialogTitle>
                <DialogContent>
                    <Stack spacing={1} sx={{ pt: 0.5 }}>
                        <Typography variant="body2">
                            Historical messages keep their original evidence snapshots. This only
                            changes future turns in this thread.
                        </Typography>
                        <Typography variant="body2">
                            Previous evidence revision: {scope.evidence_revision_hash?.slice(0, 12) ?? "—"}
                            <br />
                            New evidence revision: {liveScope?.evidence_revision_hash?.slice(0, 12) ?? "—"}
                        </Typography>
                        {addedDocumentIds.length || removedDocumentIds.length || liveScope?.unavailable_count ? (
                            <Alert severity="warning">
                                {addedDocumentIds.length
                                    ? `Added: ${addedDocumentIds.slice(0, 5).join(", ")}${addedDocumentIds.length > 5 ? "…" : ""}. `
                                    : ""}
                                {removedDocumentIds.length
                                    ? `Removed: ${removedDocumentIds.slice(0, 5).join(", ")}${removedDocumentIds.length > 5 ? "…" : ""}. `
                                    : ""}
                                {liveScope?.unavailable_count
                                    ? `Unavailable: ${liveScope.unavailable_corpus_document_ids.slice(0, 5).join(", ")}${liveScope.unavailable_count > 5 ? "…" : ""}.`
                                    : ""}
                            </Alert>
                        ) : null}
                    </Stack>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setPendingUpdate(null)}>Keep historical scope</Button>
                    <Button
                        variant="contained"
                        onClick={() => pendingUpdate && updateMutation.mutate(pendingUpdate)}
                        disabled={updateMutation.isPending}
                    >
                        {pendingUpdate?.scope_mode === "live"
                            ? "Switch to live corpus"
                            : "Update fixed snapshot"}
                    </Button>
                </DialogActions>
            </Dialog>
        </Stack>
    );
}
