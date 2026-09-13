import {
    Box,
    Button,
    Chip,
    Stack,
    Typography,
    useMediaQuery,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { Menu as MenuIcon } from "@mui/icons-material";
import {
    RESEARCH_NAV_GROUPS,
    pathForNavItem,
    type ResearchNavGroup,
    type ResearchNavGroupId,
    type ResearchNavItem,
    type ResolvedResearchNav,
} from "../researchNavigation";

type ResearchNavBarProps = {
    projectId: string;
    resolved: ResolvedResearchNav | null;
    /** When browsing another group without navigating yet (desktop). */
    browseGroupId: ResearchNavGroupId | null;
    onBrowseGroup: (groupId: ResearchNavGroupId) => void;
    onNavigate: (item: ResearchNavItem) => void;
    onOpenAll: () => void;
};

function GroupButton({
    group,
    selected,
    onClick,
}: {
    group: ResearchNavGroup;
    selected: boolean;
    onClick: () => void;
}) {
    return (
        <Button
            size="small"
            variant={selected ? "contained" : "text"}
            onClick={onClick}
            aria-current={selected ? "true" : undefined}
            sx={{
                flexShrink: 0,
                textTransform: "none",
                fontWeight: selected ? 700 : 600,
                px: 1.25,
                minHeight: 34,
            }}
        >
            {group.label}
        </Button>
    );
}

function ItemChip({
    item,
    selected,
    onClick,
}: {
    item: ResearchNavItem;
    selected: boolean;
    onClick: () => void;
}) {
    return (
        <Chip
            size="small"
            label={item.label}
            color={selected ? "primary" : "default"}
            variant={selected ? "filled" : "outlined"}
            onClick={onClick}
            aria-current={selected ? "page" : undefined}
            sx={{ fontWeight: selected ? 700 : 500 }}
        />
    );
}

export function ResearchNavBar({
    projectId,
    resolved,
    browseGroupId,
    onBrowseGroup,
    onNavigate,
    onOpenAll,
}: ResearchNavBarProps) {
    const theme = useTheme();
    const isMedium = useMediaQuery(theme.breakpoints.up("md"));
    const activeGroupId = browseGroupId ?? resolved?.group.id ?? "overview";
    const activeGroup =
        RESEARCH_NAV_GROUPS.find((group) => group.id === activeGroupId) ?? RESEARCH_NAV_GROUPS[0];

    if (!isMedium) {
        return (
            <Box
                component="nav"
                aria-label="Research navigation"
                sx={{
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 2,
                    bgcolor: "background.paper",
                    px: 1.5,
                    py: 1.25,
                }}
            >
                <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
                    <Box sx={{ minWidth: 0 }}>
                        <Typography variant="caption" color="text.secondary">
                            {resolved?.group.label ?? "Research"}
                        </Typography>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700 }} noWrap>
                            {resolved?.item.label ?? "Choose a stage"}
                        </Typography>
                    </Box>
                    <Button
                        size="small"
                        variant="outlined"
                        startIcon={<MenuIcon />}
                        onClick={onOpenAll}
                        aria-label="Open research navigation"
                    >
                        Navigate
                    </Button>
                </Stack>
            </Box>
        );
    }

    return (
        <Box
            component="nav"
            aria-label="Research navigation"
            sx={{
                position: "sticky",
                top: 0,
                zIndex: 2,
                border: 1,
                borderColor: "divider",
                borderRadius: 2,
                bgcolor: (t) =>
                    t.palette.mode === "dark" ? "background.paper" : "background.default",
                px: 1,
                py: 0.75,
            }}
        >
            <Stack
                direction="row"
                alignItems="center"
                spacing={0.5}
                sx={{
                    overflowX: "auto",
                    WebkitOverflowScrolling: "touch",
                    scrollbarWidth: "thin",
                    pb: 0.5,
                    maxWidth: "100%",
                }}
            >
                {RESEARCH_NAV_GROUPS.map((group) => (
                    <GroupButton
                        key={group.id}
                        group={group}
                        selected={activeGroupId === group.id}
                        onClick={() => {
                            onBrowseGroup(group.id);
                            // Single-item groups navigate immediately.
                            if (group.items.length === 1) {
                                onNavigate(group.items[0]);
                            }
                        }}
                    />
                ))}
                <Box sx={{ flex: 1, minWidth: 8 }} />
                <Button size="small" variant="text" onClick={onOpenAll} sx={{ flexShrink: 0 }}>
                    Progress
                </Button>
            </Stack>

            <Stack
                direction="row"
                spacing={0.75}
                flexWrap="wrap"
                useFlexGap
                sx={{ pt: 0.75, borderTop: 1, borderColor: "divider" }}
                aria-label={`${activeGroup.label} destinations`}
            >
                {activeGroup.items.map((item) => {
                    const selected =
                        resolved?.item.id === item.id &&
                        (browseGroupId == null || browseGroupId === resolved.group.id);
                    return (
                        <ItemChip
                            key={item.id}
                            item={item}
                            selected={Boolean(selected)}
                            onClick={() => onNavigate(item)}
                        />
                    );
                })}
            </Stack>

            {/* Hidden parity helper for tests / screen readers listing full hrefs */}
            <Box component="ul" sx={{ display: "none" }} aria-hidden>
                {RESEARCH_NAV_GROUPS.flatMap((group) =>
                    group.items.map((item) => (
                        <li key={`${group.id}-${item.id}`}>{pathForNavItem(projectId, item)}</li>
                    ))
                )}
            </Box>
        </Box>
    );
}
