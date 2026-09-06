import { useCallback } from "react";

import { useSnackbar } from "../app/snackbarContext";
import { getQueryErrorMessage } from "../utils/queryErrors";

export function useMutationErrorToast() {
    const { showToast } = useSnackbar();

    return useCallback(
        (error: unknown, fallback: string) => {
            showToast({
                message: getQueryErrorMessage(error, fallback),
                severity: "error",
            });
        },
        [showToast]
    );
}
