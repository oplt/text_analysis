import { Box, Tab, Tabs, type SxProps, type Theme } from "@mui/material";

export type PageTabItem<T extends string = string> = {
    value: T;
    label: React.ReactNode;
    disabled?: boolean;
};

type PageTabsProps<T extends string = string> = {
    value: T;
    onChange: (value: T) => void;
    tabs: Array<PageTabItem<T>>;
    ariaLabel?: string;
    sx?: SxProps<Theme>;
};

export function PageTabs<T extends string = string>({
    value,
    onChange,
    tabs,
    ariaLabel = "Page sections",
    sx,
}: PageTabsProps<T>) {
    return (
        <Box
            sx={[
                (theme) => ({
                    borderBottom: `1px solid ${theme.palette.divider}`,
                    mb: 2,
                }),
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            <Tabs
                value={value}
                onChange={(_, next: T) => onChange(next)}
                variant="scrollable"
                scrollButtons="auto"
                allowScrollButtonsMobile
                aria-label={ariaLabel}
                sx={{
                    minHeight: 40,
                    "& .MuiTab-root": {
                        minHeight: 40,
                        py: 1,
                        px: 1.5,
                        fontSize: "0.8125rem",
                        fontWeight: 500,
                        textTransform: "none",
                    },
                    "& .MuiTabs-indicator": {
                        height: 2,
                    },
                }}
            >
                {tabs.map((tab) => (
                    <Tab
                        key={tab.value}
                        value={tab.value}
                        label={tab.label}
                        disabled={tab.disabled}
                    />
                ))}
            </Tabs>
        </Box>
    );
}
