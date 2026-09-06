import { Alert, Box, Button, CircularProgress, Skeleton, type AlertProps } from "@mui/material";

import { getQueryErrorMessage } from "../../utils/queryErrors";

type QueryBoundaryProps = {
    isLoading?: boolean;
    isError?: boolean;
    error?: unknown;
    isEmpty?: boolean;
    errorFallback?: string;
    errorTitle?: string;
    severity?: AlertProps["severity"];
    onRetry?: () => void;
    loadingFallback?: React.ReactNode;
    emptyFallback?: React.ReactNode;
    variant?: "inline" | "page";
    children: React.ReactNode;
};

export function QueryErrorAlert({
    error,
    fallback = "Failed to load data.",
    title,
    onRetry,
}: {
    error: unknown;
    fallback?: string;
    title?: string;
    onRetry?: () => void;
}) {
    if (!error) {
        return null;
    }

    return (
        <Alert
            severity="error"
            action={
                onRetry ? (
                    <Button color="inherit" size="small" onClick={onRetry}>
                        Retry
                    </Button>
                ) : undefined
            }
        >
            {title ? (
                <>
                    <strong>{title}: </strong>
                    {getQueryErrorMessage(error, fallback)}
                </>
            ) : (
                getQueryErrorMessage(error, fallback)
            )}
        </Alert>
    );
}

export function QueryBoundary({
    isLoading = false,
    isError = false,
    error,
    isEmpty = false,
    errorFallback = "Failed to load data.",
    errorTitle,
    severity = "error",
    onRetry,
    loadingFallback,
    emptyFallback,
    variant = "inline",
    children,
}: QueryBoundaryProps) {
    if (isLoading) {
        if (loadingFallback) {
            return <>{loadingFallback}</>;
        }
        if (variant === "page") {
            return (
                <Box sx={{ display: "grid", placeItems: "center", minHeight: "40vh" }}>
                    <CircularProgress />
                </Box>
            );
        }
        return <Skeleton variant="rounded" height={120} sx={{ borderRadius: 4 }} />;
    }

    if (isError) {
        return (
            <Alert
                severity={severity}
                action={
                    onRetry ? (
                        <Button color="inherit" size="small" onClick={onRetry}>
                            Retry
                        </Button>
                    ) : undefined
                }
            >
                {errorTitle ? (
                    <>
                        <strong>{errorTitle}: </strong>
                        {getQueryErrorMessage(error, errorFallback)}
                    </>
                ) : (
                    getQueryErrorMessage(error, errorFallback)
                )}
            </Alert>
        );
    }

    if (isEmpty && emptyFallback) {
        return <>{emptyFallback}</>;
    }

    return <>{children}</>;
}
