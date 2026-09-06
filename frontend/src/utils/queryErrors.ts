export function getQueryErrorMessage(error: unknown, fallback = "Something went wrong.") {
    return error instanceof Error ? error.message : fallback;
}
