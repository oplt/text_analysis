import { Box, Tab, Tabs } from "@mui/material";
import { alpha } from "@mui/material/styles";
import { useLocation, useNavigate } from "react-router-dom";

import { getActiveSettingsTab, useSettingsTabs } from "../../hooks/useSettingsTabs";

export function SettingsTabs() {
    const location = useLocation();
    const navigate = useNavigate();
    const tabs = useSettingsTabs();
    const activePath =
        getActiveSettingsTab(location.pathname, tabs)?.path ?? tabs[0]?.path ?? "/profile";

    if (tabs.length === 0) return null;

    return (
        <Box
            sx={(theme) => ({
                p: 0.75,
                borderRadius: 4,
                border: `1px solid ${theme.palette.divider}`,
                backgroundColor: alpha(theme.palette.background.paper, 0.82),
            })}
        >
            <Tabs
                value={activePath}
                onChange={(_, value: string) => navigate(value)}
                variant="scrollable"
                scrollButtons="auto"
                allowScrollButtonsMobile
            >
                {tabs.map((item) => (
                    <Tab key={item.path} value={item.path} label={item.label} />
                ))}
            </Tabs>
        </Box>
    );
}
