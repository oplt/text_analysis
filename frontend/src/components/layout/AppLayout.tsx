import { useMemo, useState, type ReactNode } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import {
    AppBar,
    Avatar,
    Box,
    Button,
    Chip,
    Drawer,
    IconButton,
    List,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Stack,
    Toolbar,
    Tooltip,
    Typography,
    useMediaQuery,
} from "@mui/material";
import {
    CalendarMonth as CalendarIcon,
    ChevronLeft as ChevronLeftIcon,
    ChevronRight as ChevronRightIcon,
    Dashboard as DashboardIcon,
    DarkMode as DarkModeIcon,
    FolderOpen as ProjectsIcon,
    LightMode as LightModeIcon,
    Logout as LogoutIcon,
    Menu as MenuIcon,
    Notifications as NotificationsIcon,
    Settings as SettingsIcon,
    SettingsBrightness as SystemModeIcon,
} from "@mui/icons-material";
import { useTheme } from "@mui/material/styles";
import { colors, fonts } from "../../app/designTokens";
import { useColorMode } from "../../app/colorModeContext";
import { useAuth } from "../../hooks/useAuth";
import { usePlatformMetadata } from "../../hooks/usePlatformMetadata";
import { useUserProfile } from "../../hooks/useUserProfile";
import {
    getSettingsHubLabel,
    isSettingsHubPath,
    useSettingsTabs,
} from "../../hooks/useSettingsTabs";
import { NotificationNavBadge } from "./NotificationNavBadge";
import { getInitials } from "../../utils/formatters";

const DRAWER_WIDTH = 288;
const COLLAPSED_DRAWER_WIDTH = 96;

type NavItem = {
    label: string;
    icon: ReactNode;
    path: string;
    group: "workspace";
    isSelected?: (pathname: string) => boolean;
};

function ThemeToggle() {
    const { colorMode, setColorMode } = useColorMode();
    const cycle = () => {
        const next: Record<string, typeof colorMode> = { light: "dark", dark: "system", system: "light" };
        setColorMode(next[colorMode]);
    };
    const icon =
        colorMode === "light" ? <LightModeIcon fontSize="small" /> :
        colorMode === "dark" ? <DarkModeIcon fontSize="small" /> :
        <SystemModeIcon fontSize="small" />;

    return (
        <Tooltip title={`Theme: ${colorMode}`}>
            <IconButton
                onClick={cycle}
                size="small"
                aria-label="Cycle color theme"
            >
                {icon}
            </IconButton>
        </Tooltip>
    );
}

function NavBlock({
    title,
    items,
    currentPath,
    onNavigate,
    collapsed,
}: {
    title: string;
    items: NavItem[];
    currentPath: string;
    onNavigate: (path: string) => void;
    collapsed: boolean;
}) {
    if (items.length === 0) {
        return null;
    }

    return (
        <Stack spacing={1}>
            {!collapsed && (
                <Typography variant="subtitle2" color="text.secondary" sx={{ px: 2 }}>
                    {title}
                </Typography>
            )}
            <List disablePadding sx={{ display: "grid", gap: 0.75 }}>
                {items.map((item) => {
                    const selected = item.isSelected
                        ? item.isSelected(currentPath)
                        : item.path === "/dashboard"
                          ? currentPath === item.path
                          : currentPath.startsWith(item.path);
                    const itemButton = (
                        <ListItemButton
                            key={item.path}
                            selected={selected}
                            aria-current={selected ? "page" : undefined}
                            onClick={() => onNavigate(item.path)}
                            sx={
                                collapsed
                                    ? {
                                          minHeight: 48,
                                          px: 1,
                                          justifyContent: "center",
                                      }
                                    : undefined
                            }
                        >
                            <ListItemIcon
                                sx={{
                                    minWidth: collapsed ? "auto" : 40,
                                    justifyContent: "center",
                                }}
                            >
                                {item.icon}
                            </ListItemIcon>
                            {!collapsed && (
                                <ListItemText
                                    primary={item.label}
                                    secondary={selected ? "Current section" : undefined}
                                    secondaryTypographyProps={{ sx: { fontSize: "0.74rem" } }}
                                />
                            )}
                        </ListItemButton>
                    );

                    if (!collapsed) {
                        return itemButton;
                    }

                    return (
                        <Tooltip key={item.path} title={item.label} placement="right">
                            {itemButton}
                        </Tooltip>
                    );
                })}
            </List>
        </Stack>
    );
}

export function AppLayout() {
    const [drawerOpen, setDrawerOpen] = useState(false);
    const [desktopNavCollapsed, setDesktopNavCollapsed] = useState(false);
    const { logout, currentUser } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down("md"));
    const { data: platformMetadata } = usePlatformMetadata();
    const { data: profile } = useUserProfile();
    const settingsTabs = useSettingsTabs();
    const appName = platformMetadata?.app_name ?? "Your App";
    const coreDomainPlural = platformMetadata?.core_domain_plural ?? "Projects";
    const drawerCollapsed = !isMobile && desktopNavCollapsed;
    const desktopDrawerWidth = drawerCollapsed ? COLLAPSED_DRAWER_WIDTH : DRAWER_WIDTH;

    const settingsNavItem = useMemo<NavItem>(
        () => ({
            label: "Settings",
            icon: <SettingsIcon />,
            path: "/profile",
            group: "workspace",
            isSelected: (pathname) => isSettingsHubPath(pathname, settingsTabs),
        }),
        [settingsTabs]
    );

    const navItems = useMemo<NavItem[]>(
        () => [
            { label: "Dashboard", icon: <DashboardIcon />, path: "/dashboard", group: "workspace" },
            { label: coreDomainPlural, icon: <ProjectsIcon />, path: "/projects", group: "workspace" },
        ],
        [coreDomainPlural]
    );

    const visibleNavItems = navItems;
    const settingsSelected = settingsNavItem.isSelected?.(location.pathname) ?? false;
    const currentItem =
        settingsSelected
            ? settingsNavItem
            : visibleNavItems.find((item) =>
                  item.path === "/dashboard"
                      ? location.pathname === item.path
                      : location.pathname.startsWith(item.path)
              );
    const pageTitle =
        getSettingsHubLabel(location.pathname, settingsTabs) ??
        currentItem?.label ??
        (location.pathname.startsWith("/calendar") ? "Calendar" : undefined) ??
        (location.pathname.startsWith("/notifications") ? "Notifications" : undefined) ??
        "Workspace";
    const avatarLabel = getInitials(currentUser?.full_name, currentUser?.email);

    function handleNavigate(path: string) {
        navigate(path);
        setDrawerOpen(false);
    }

    async function handleSignOut() {
        await logout();
        setDrawerOpen(false);
        navigate("/", { replace: true });
    }

    const drawerContent = (
        <Stack sx={{ height: "100%", p: drawerCollapsed ? 1.25 : 2 }}>
            <Tooltip
                title={appName}
                placement="right"
                disableHoverListener={!drawerCollapsed}
            >
                <Box
                    sx={{
                        px: drawerCollapsed ? 1 : 2,
                        py: drawerCollapsed ? 1.75 : 2.25,
                        mb: 2,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: drawerCollapsed ? "center" : "flex-start",
                        textAlign: drawerCollapsed ? "center" : "left",
                    }}
                >
                    {drawerCollapsed ? (
                        <Typography
                            variant="h6"
                            sx={{
                                lineHeight: 1,
                                fontFamily: fonts.display,
                                letterSpacing: "0.12em",
                            }}
                        >
                            {appName.trim().charAt(0).toUpperCase() || "W"}
                        </Typography>
                    ) : (
                        <Box>
                            <Typography
                                variant="h6"
                                sx={{
                                    fontFamily: fonts.display,
                                    letterSpacing: "0.12em",
                                    textTransform: "uppercase",
                                }}
                            >
                                {appName}
                            </Typography>
                            {platformMetadata?.module_pack && (
                                <Chip
                                    label={`Pack: ${platformMetadata.module_pack}`}
                                    size="small"
                                    variant="outlined"
                                    sx={{ mt: 1.5 }}
                                />
                            )}
                        </Box>
                    )}
                </Box>
            </Tooltip>

            <Stack spacing={drawerCollapsed ? 1 : 2}>
                <NavBlock
                    title="Product"
                    items={visibleNavItems}
                    currentPath={location.pathname}
                    onNavigate={handleNavigate}
                    collapsed={drawerCollapsed}
                />
            </Stack>

            <Box sx={{ flexGrow: 1 }} />

            <Box sx={{ p: drawerCollapsed ? 1.25 : 2 }}>
                <Stack spacing={1.5} alignItems={drawerCollapsed ? "center" : "stretch"}>
                    {drawerCollapsed ? (
                        <Tooltip title="Settings" placement="right">
                            <IconButton
                                aria-label="Settings"
                                aria-current={settingsSelected ? "page" : undefined}
                                onClick={() => handleNavigate(settingsNavItem.path)}
                                color={settingsSelected ? "primary" : "default"}
                            >
                                <SettingsIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                    ) : (
                        <ListItemButton
                            selected={settingsSelected}
                            aria-current={settingsSelected ? "page" : undefined}
                            onClick={() => handleNavigate(settingsNavItem.path)}
                            sx={{ borderRadius: 2 }}
                        >
                            <ListItemIcon sx={{ minWidth: 40 }}>{settingsNavItem.icon}</ListItemIcon>
                            <ListItemText
                                primary={settingsNavItem.label}
                                secondary={settingsSelected ? "Current section" : undefined}
                                secondaryTypographyProps={{ sx: { fontSize: "0.74rem" } }}
                            />
                        </ListItemButton>
                    )}
                    <Tooltip title="Profile" placement="right" disableHoverListener={!drawerCollapsed}>
                        <ListItemButton
                            selected={location.pathname.startsWith("/profile")}
                            aria-label="Open profile settings"
                            onClick={() => handleNavigate("/profile")}
                            sx={{
                                borderRadius: 2,
                                px: drawerCollapsed ? 1 : 2,
                                py: drawerCollapsed ? 1 : 1.25,
                                justifyContent: drawerCollapsed ? "center" : "flex-start",
                            }}
                        >
                            <ListItemIcon
                                sx={{
                                    minWidth: drawerCollapsed ? "auto" : 52,
                                    justifyContent: "center",
                                }}
                            >
                                <Avatar
                                    src={profile?.avatar_url ?? undefined}
                                    sx={{ width: drawerCollapsed ? 40 : 44, height: drawerCollapsed ? 40 : 44 }}
                                >
                                    {avatarLabel}
                                </Avatar>
                            </ListItemIcon>
                            {!drawerCollapsed && (
                                <ListItemText
                                    primary={currentUser?.full_name ?? "Your profile"}
                                    secondary={currentUser?.email ?? "Signed in"}
                                    primaryTypographyProps={{ noWrap: true }}
                                    secondaryTypographyProps={{ noWrap: true }}
                                />
                            )}
                        </ListItemButton>
                    </Tooltip>
                    {drawerCollapsed ? (
                        <Tooltip title="Sign out" placement="right">
                            <IconButton aria-label="Sign out" onClick={() => void handleSignOut()}>
                                <LogoutIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                    ) : (
                        <Button
                            variant="text"
                            color="inherit"
                            fullWidth
                            startIcon={<LogoutIcon />}
                            onClick={() => void handleSignOut()}
                        >
                            Sign out
                        </Button>
                    )}
                </Stack>
            </Box>
        </Stack>
    );

    return (
        <Box sx={{ minHeight: "100vh" }}>
            <AppBar
                position="fixed"
                elevation={0}
                color="inherit"
                sx={{
                    left: { md: `${desktopDrawerWidth}px` },
                    width: { md: `calc(100% - ${desktopDrawerWidth}px)` },
                    backgroundColor:
                        theme.palette.mode === "dark"
                            ? theme.palette.background.paper
                            : colors.frostedGlass,
                    backdropFilter: "blur(12px)",
                    color: "text.primary",
                    transition: theme.transitions.create(["left", "width", "background-color"], {
                        duration: theme.transitions.duration.shorter,
                    }),
                }}
            >
                <Toolbar sx={{ minHeight: { xs: 56, md: 64 }, px: { xs: 2, md: 3 } }}>
                    {isMobile ? (
                        <IconButton
                            edge="start"
                            onClick={() => setDrawerOpen(true)}
                            sx={{ mr: 1.25 }}
                            aria-label="Open navigation menu"
                        >
                            <MenuIcon />
                        </IconButton>
                    ) : (
                        <Tooltip title={drawerCollapsed ? "Expand menu" : "Collapse menu"}>
                            <IconButton
                                edge="start"
                                onClick={() => setDesktopNavCollapsed((current) => !current)}
                                aria-label={drawerCollapsed ? "Expand menu" : "Collapse menu"}
                                sx={{ mr: 1.25 }}
                            >
                                {drawerCollapsed ? <ChevronRightIcon /> : <ChevronLeftIcon />}
                            </IconButton>
                        </Tooltip>
                    )}
                    <Box sx={{ minWidth: 0, flexGrow: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                            {appName}
                        </Typography>
                        <Typography variant="h6" noWrap>
                            {pageTitle}
                        </Typography>
                    </Box>
                    <Stack direction="row" spacing={0.25} alignItems="center">
                        <Tooltip title="Calendar">
                            <IconButton
                                aria-label="Open calendar"
                                size="small"
                                onClick={() => navigate("/calendar")}
                                color={location.pathname.startsWith("/calendar") ? "primary" : "default"}
                            >
                                <CalendarIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Notifications">
                            <IconButton
                                aria-label="Open notifications"
                                size="small"
                                onClick={() => navigate("/notifications")}
                                color={location.pathname.startsWith("/notifications") ? "primary" : "default"}
                            >
                                <NotificationNavBadge>
                                    <NotificationsIcon fontSize="small" />
                                </NotificationNavBadge>
                            </IconButton>
                        </Tooltip>
                        <ThemeToggle />
                    </Stack>
                </Toolbar>
            </AppBar>

            {isMobile ? (
                <Drawer
                    open={drawerOpen}
                    onClose={() => setDrawerOpen(false)}
                    ModalProps={{ keepMounted: true }}
                    sx={{ "& .MuiDrawer-paper": { width: DRAWER_WIDTH } }}
                >
                    {drawerContent}
                </Drawer>
            ) : (
                <Drawer
                    variant="permanent"
                    open
                    sx={{
                        width: desktopDrawerWidth,
                        flexShrink: 0,
                        "& .MuiDrawer-paper": {
                            width: desktopDrawerWidth,
                            boxSizing: "border-box",
                            overflowX: "hidden",
                            transition: theme.transitions.create("width", {
                                duration: theme.transitions.duration.shorter,
                            }),
                        },
                    }}
                >
                    {drawerContent}
                </Drawer>
            )}

            <Box
                component="main"
                sx={{
                    minHeight: "100vh",
                    ml: { md: `${desktopDrawerWidth}px` },
                    pt: { xs: "56px", md: "64px" },
                    transition: theme.transitions.create("margin-left", {
                        duration: theme.transitions.duration.shorter,
                    }),
                }}
            >
                <Outlet />
            </Box>
        </Box>
    );
}
