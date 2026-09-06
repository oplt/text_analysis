import { Box, Stack, Typography } from "@mui/material";
import { fonts } from "../../app/designTokens";

type Highlight = {
    value: string;
    label: string;
};

type AuthMarketingPanelProps = {
    appName: string;
    eyebrow: string;
    title: string;
    description: string;
    highlights?: Highlight[];
    points?: string[];
};

export function AuthMarketingPanel({
    appName,
    eyebrow,
    title,
    description,
    highlights = [],
    points = [],
}: AuthMarketingPanelProps) {
    return (
        <Stack justifyContent="space-between" spacing={4} sx={{ height: "100%" }}>
            <Stack spacing={3}>
                <Box>
                    <Typography
                        variant="overline"
                        sx={(theme) => ({
                            color: theme.palette.text.secondary,
                            display: "block",
                            mb: 1,
                        })}
                    >
                        {eyebrow}
                    </Typography>
                    <Typography
                        variant="h3"
                        sx={{
                            fontFamily: fonts.display,
                            mb: 1.25,
                        }}
                    >
                        {title}
                    </Typography>
                    <Typography color="text.secondary" sx={{ maxWidth: 620 }}>
                        {description}
                    </Typography>
                </Box>

                {highlights.length > 0 && (
                    <Box
                        sx={{
                            display: "grid",
                            gap: 2,
                            gridTemplateColumns: { xs: "1fr", sm: "repeat(3, minmax(0, 1fr))" },
                        }}
                    >
                        {highlights.map((item) => (
                            <Box key={item.label}>
                                <Typography variant="h5">{item.value}</Typography>
                                <Typography color="text.secondary" sx={{ mt: 0.5 }}>
                                    {item.label}
                                </Typography>
                            </Box>
                        ))}
                    </Box>
                )}
            </Stack>

            <Stack spacing={1.5}>
                <Typography variant="subtitle2" color="text.primary">
                    {appName}
                </Typography>
                {points.map((point) => (
                    <Typography key={point} color="text.secondary">
                        {point}
                    </Typography>
                ))}
            </Stack>
        </Stack>
    );
}
