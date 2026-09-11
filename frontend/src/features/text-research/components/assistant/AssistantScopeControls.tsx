import { Alert, Button, Stack, Typography } from "@mui/material";
import { useMutation } from "@tanstack/react-query";
import {
    updateAssistantThreadScope,
    type AssistantScope,
} from "../../../../api/textResearch";
import { getQueryErrorMessage } from "../../../../utils/queryErrors";

type Props = {
    threadId: string | null;
    scope: AssistantScope | null;
    liveScope: AssistantScope | null;
    onUpdated: (scope: AssistantScope) => void;
};

export function AssistantScopeControls({ threadId, scope, liveScope, onUpdated }: Props) {
    const updateMutation = useMutation({
        mutationFn: () =>
            updateAssistantThreadScope(threadId!, {
                document_ids: null,
                scope_mode: "fixed",
                reason: "User explicitly updated the thread to the current corpus scope",
            }),
        onSuccess: onUpdated,
    });

    if (!scope) return null;

    const membershipChanged = Boolean(
        threadId && liveScope && liveScope.scope_hash !== scope.scope_hash
    );
    const scopeDescription =
        scope.scope_mode === "live"
            ? "Live scope · follows current corpus membership"
            : `Fixed scope · ${scope.corpus_document_ids.length} document(s)`;

    return (
        <Stack spacing={0.75}>
            <Typography variant="caption" color="text.secondary">
                {scopeDescription}
            </Typography>
            {membershipChanged ? (
                <Alert
                    severity="warning"
                    action={
                        <Button
                            color="inherit"
                            size="small"
                            onClick={() => updateMutation.mutate()}
                            disabled={updateMutation.isPending}
                        >
                            Update scope
                        </Button>
                    }
                >
                    The current corpus differs from this thread&apos;s fixed evidence scope.
                    Historical answers remain tied to their original scope.
                </Alert>
            ) : null}
            {updateMutation.isError ? (
                <Alert severity="error">
                    {getQueryErrorMessage(updateMutation.error, "Thread scope update failed.")}
                </Alert>
            ) : null}
        </Stack>
    );
}
