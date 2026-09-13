import { Alert, Box, Stack, Typography } from "@mui/material";

export type AnalysisChartFrameProps = {
    title: string;
    subtitle?: string;
    /** Optional legend / series explanation shown above the plot. */
    legend?: React.ReactNode;
    xAxisLabel?: string;
    yAxisLabel?: string;
    methodologicalNote?: string;
    exportAction?: React.ReactNode;
    children: React.ReactNode;
};

/**
 * Standard chrome for Analysis charts: title, subtitle, legend, axis cues,
 * export action, and a methodological note so plots are never bare numbers.
 */
export function AnalysisChartFrame({
    title,
    subtitle,
    legend,
    xAxisLabel,
    yAxisLabel,
    methodologicalNote,
    exportAction,
    children,
}: AnalysisChartFrameProps) {
    return (
        <Stack spacing={1.25} component="figure" sx={{ m: 0 }}>
            <Stack
                direction={{ xs: "column", sm: "row" }}
                spacing={1}
                justifyContent="space-between"
                alignItems={{ sm: "flex-start" }}
            >
                <Box sx={{ minWidth: 0 }}>
                    <Typography variant="subtitle1" component="figcaption">
                        {title}
                    </Typography>
                    {subtitle ? (
                        <Typography variant="body2" color="text.secondary">
                            {subtitle}
                        </Typography>
                    ) : null}
                </Box>
                {exportAction ? (
                    <Box sx={{ flexShrink: 0 }}>{exportAction}</Box>
                ) : null}
            </Stack>

            {legend ? (
                <Typography variant="caption" color="text.secondary" component="div">
                    {legend}
                </Typography>
            ) : null}

            {xAxisLabel || yAxisLabel ? (
                <Typography variant="caption" color="text.secondary">
                    {[
                        yAxisLabel ? `Vertical: ${yAxisLabel}` : null,
                        xAxisLabel ? `Horizontal: ${xAxisLabel}` : null,
                    ]
                        .filter(Boolean)
                        .join(" · ")}
                </Typography>
            ) : null}

            <Box sx={{ width: "100%", minWidth: 0 }}>{children}</Box>

            {methodologicalNote ? (
                <Alert severity="info" icon={false} sx={{ py: 0.75 }}>
                    <Typography variant="body2" color="text.secondary">
                        {methodologicalNote}
                    </Typography>
                </Alert>
            ) : null}
        </Stack>
    );
}
