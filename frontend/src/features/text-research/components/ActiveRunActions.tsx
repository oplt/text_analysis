import { Button } from "@mui/material";
import { Stop as CancelIcon } from "@mui/icons-material";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { cancelRun } from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { isActiveRunStatus } from "../runPolling";
import type { AnalysisRun } from "../types";

type ActiveRunActionsProps = {
    run: Pick<AnalysisRun, "id" | "status">;
    /** Optional project context so list queries invalidate after cancel. */
    projectId?: string | null;
    /** Reserved for callers; list invalidation uses project-scoped runs prefix. */
    corpusId?: string | null;
    size?: "small" | "medium";
};

/**
 * Cancel control for in-progress analysis runs (Analysis / Topics / Classification
 * result panels, not only the Runs view).
 */
export function ActiveRunActions({
    run,
    projectId,
    corpusId,
    size = "small",
}: ActiveRunActionsProps) {
    const { showToast } = useSnackbar();
    const queryClient = useQueryClient();

    const cancelMutation = useMutation({
        mutationFn: () => cancelRun(run.id),
        onSuccess: async (updated) => {
            await queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.run(updated.id),
            });
            if (projectId) {
                await queryClient.invalidateQueries({
                    queryKey: ["text-research", projectId, "runs"],
                });
            } else {
                await queryClient.invalidateQueries({
                    queryKey: ["text-research"],
                    predicate: (query) =>
                        Array.isArray(query.queryKey) && query.queryKey.includes("runs"),
                });
            }
            showToast({ message: "Run cancelled.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Cancel failed."),
                severity: "error",
            }),
    });

    if (!isActiveRunStatus(run.status)) return null;

    return (
        <Button
            size={size}
            color="warning"
            startIcon={<CancelIcon />}
            disabled={cancelMutation.isPending}
            onClick={() => cancelMutation.mutate()}
        >
            Cancel
        </Button>
    );
}
