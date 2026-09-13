import { useId, useState } from "react";
import {
    Box,
    IconButton,
    Menu,
    MenuItem,
    Stack,
    Typography,
    type SxProps,
    type Theme,
} from "@mui/material";
import { MoreVert as MoreIcon } from "@mui/icons-material";
import { headingHierarchy, layoutSpacing } from "./layoutTokens";

export type PageHeaderOverflowItem = {
    id: string;
    label: React.ReactNode;
    onClick: () => void;
    disabled?: boolean;
};

type PageHeaderProps = {
    title: React.ReactNode;
    description?: React.ReactNode;
    /** Optional status chip / badge rendered under the title row. */
    status?: React.ReactNode;
    /** Preferred primary CTA (right-aligned). */
    primaryAction?: React.ReactNode;
    /** Secondary actions before the primary. */
    secondaryActions?: React.ReactNode;
    /** Overflow menu items (⋯). */
    overflowItems?: PageHeaderOverflowItem[];
    /**
     * Legacy single actions slot. Prefer primaryAction / secondaryActions.
     * Rendered after secondaryActions and before primaryAction when provided.
     */
    actions?: React.ReactNode;
    icon?: React.ReactNode;
    dense?: boolean;
    sx?: SxProps<Theme>;
};

export function PageHeader({
    title,
    description,
    status,
    primaryAction,
    secondaryActions,
    overflowItems,
    actions,
    icon,
    dense = false,
    sx,
}: PageHeaderProps) {
    const menuId = useId();
    const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
    const hasActions =
        Boolean(primaryAction) ||
        Boolean(secondaryActions) ||
        Boolean(actions) ||
        Boolean(overflowItems?.length);

    return (
        <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={dense ? 1.5 : 2}
            justifyContent="space-between"
            alignItems={{ xs: "stretch", sm: "flex-start" }}
            sx={[
                {
                    mb: dense
                        ? layoutSpacing.headerMarginBottom.dense
                        : layoutSpacing.headerMarginBottom.default,
                },
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            <Stack direction="row" spacing={1.5} alignItems="flex-start" sx={{ minWidth: 0 }}>
                {icon ? (
                    <Box sx={{ color: "primary.main", display: "flex", mt: 0.25 }}>{icon}</Box>
                ) : null}
                <Box sx={{ minWidth: 0 }}>
                    <Typography
                        variant={dense ? headingHierarchy.pageDense : headingHierarchy.page}
                        component="h1"
                    >
                        {title}
                    </Typography>
                    {description ? (
                        <Typography
                            variant="body2"
                            color="text.secondary"
                            sx={{ mt: 0.5, maxWidth: 720 }}
                        >
                            {description}
                        </Typography>
                    ) : null}
                    {status ? <Box sx={{ mt: 1 }}>{status}</Box> : null}
                </Box>
            </Stack>
            {hasActions ? (
                <Stack
                    direction="row"
                    spacing={1}
                    flexWrap="wrap"
                    useFlexGap
                    justifyContent={{ xs: "flex-start", sm: "flex-end" }}
                    alignItems="center"
                    sx={{ flexShrink: 0 }}
                >
                    {secondaryActions}
                    {actions}
                    {primaryAction}
                    {overflowItems && overflowItems.length > 0 ? (
                        <>
                            <IconButton
                                size="small"
                                aria-label="More actions"
                                aria-controls={menuAnchor ? menuId : undefined}
                                aria-haspopup="true"
                                aria-expanded={menuAnchor ? "true" : undefined}
                                onClick={(event) => setMenuAnchor(event.currentTarget)}
                            >
                                <MoreIcon fontSize="small" />
                            </IconButton>
                            <Menu
                                id={menuId}
                                anchorEl={menuAnchor}
                                open={Boolean(menuAnchor)}
                                onClose={() => setMenuAnchor(null)}
                                anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
                                transformOrigin={{ vertical: "top", horizontal: "right" }}
                            >
                                {overflowItems.map((item) => (
                                    <MenuItem
                                        key={item.id}
                                        disabled={item.disabled}
                                        onClick={() => {
                                            setMenuAnchor(null);
                                            item.onClick();
                                        }}
                                    >
                                        {item.label}
                                    </MenuItem>
                                ))}
                            </Menu>
                        </>
                    ) : null}
                </Stack>
            ) : null}
        </Stack>
    );
}
