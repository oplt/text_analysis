import { Alert, Stack } from "@mui/material";

/** Domain-aware research warnings (Phase 15). Never treat as lifecycle decisions. */
export function ScientificWarnings({
    warnings,
    title,
}: {
    warnings: string[] | null | undefined;
    title?: string;
}) {
    const items = (warnings ?? []).map(String).filter((item) => item.trim().length > 0);
    if (!items.length) return null;
    return (
        <Stack spacing={1}>
            {title ? (
                <Alert severity="info" sx={{ py: 0.5 }}>
                    {title}
                </Alert>
            ) : null}
            {items.map((message) => (
                <Alert key={message} severity="warning" sx={{ py: 0.5 }}>
                    {message}
                </Alert>
            ))}
        </Stack>
    );
}
