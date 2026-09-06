import { useQuery } from "@tanstack/react-query";
import {
    Box,
    Button,
    Stack,
    Typography,
} from "@mui/material";
import {
    ArrowForward as ArrowForwardIcon,
    FolderOpen as ProjectsIcon,
    Notifications as NotificationsIcon,
    Security as SecurityIcon,
    VerifiedUser as VerifiedUserIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { colors } from "../app/designTokens";
import { listProjects } from "../api/projects";
import { queryKeys } from "../config/queryKeys";
import { DashboardCalendar } from "../components/dashboard/DashboardCalendar";
import { NotificationListItem } from "../components/notifications/NotificationListItem";
import { PageShell } from "../components/ui/PageShell";
import { QueryBoundary } from "../components/ui/QueryBoundary";
import { SectionCard } from "../components/ui/SectionCard";
import { StatCard } from "../components/ui/StatCard";
import { EmptyState } from "../components/ui/EmptyState";
import { useAuth } from "../hooks/useAuth";
import { useNotifications } from "../hooks/useNotifications";
import { usePlatformMetadata } from "../hooks/usePlatformMetadata";

export default function DashboardPage() {
    const navigate = useNavigate();
    const { currentUser: user } = useAuth();
    const { data: platformMetadata } = usePlatformMetadata();
    const projectsQuery = useQuery({
        queryKey: queryKeys.projects.all,
        queryFn: listProjects,
    });
    const notificationsQuery = useNotifications();
    const {
        data: projects,
        isLoading: projectsLoading,
        isError: projectsIsError,
        error: projectsError,
        refetch: refetchProjects,
    } = projectsQuery;
    const {
        data: notifications,
        isLoading: notificationsLoading,
        isError: notificationsIsError,
        error: notificationsError,
        refetch: refetchNotifications,
    } = notificationsQuery;

    const coreDomainPlural = platformMetadata?.core_domain_plural ?? "Projects";
    const unreadCount = notifications?.filter((item) => !item.is_read).length ?? 0;
    const recentNotifications = notifications?.slice(0, 5) ?? [];
    const accountChecks = [
        {
            label: "Email verification",
            value: user?.is_verified ? "Verified" : "Action needed",
            color: user?.is_verified ? "success.main" : "warning.main",
            description: user?.is_verified
                ? "Your sign-in identity is confirmed."
                : "Verify your email to improve recovery and trust.",
        },
        {
            label: "Multi-factor authentication",
            value: user?.mfa_enabled ? "Enabled" : "Recommended",
            color: user?.mfa_enabled ? "success.main" : "warning.main",
            description: user?.mfa_enabled
                ? "An extra layer of account protection is active."
                : "Turn on MFA to reduce account risk.",
        },
    ];

    return (
        <PageShell maxWidth="xl">
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1} sx={{ mb: 2 }}>
                <Button variant="contained" endIcon={<ArrowForwardIcon />} onClick={() => navigate("/projects")}>
                    Open {coreDomainPlural}
                </Button>
                <Button variant="outlined" onClick={() => navigate("/notifications")}>
                    View inbox
                </Button>
            </Stack>

            <Box
                sx={{
                    display: "grid",
                    gap: 2,
                    gridTemplateColumns: {
                        xs: "1fr",
                        sm: "repeat(2, minmax(0, 1fr))",
                        xl: "repeat(4, minmax(0, 1fr))",
                    },
                }}
            >
                <StatCard
                    label={coreDomainPlural}
                    value={projects?.length ?? 0}
                    description={`Total ${coreDomainPlural.toLowerCase()} in your workspace`}
                    icon={<ProjectsIcon />}
                    loading={projectsLoading}
                />
                <StatCard
                    label="Unread notifications"
                    value={unreadCount}
                    description="New updates waiting for a response"
                    icon={<NotificationsIcon />}
                    loading={notificationsLoading}
                    color="warning"
                />
                <StatCard
                    label="Email status"
                    value={user?.is_verified ? "Verified" : "Pending"}
                    description="Identity confirmation for your account"
                    icon={<VerifiedUserIcon />}
                    color={user?.is_verified ? "success" : "warning"}
                />
                <StatCard
                    label="Multi-factor auth"
                    value={user?.mfa_enabled ? "Enabled" : "Not enabled"}
                    description="Additional sign-in protection"
                    icon={<SecurityIcon />}
                    color={user?.mfa_enabled ? "success" : "secondary"}
                />
            </Box>

            <Box
                sx={{
                    display: "grid",
                    gap: 2,
                    gridTemplateColumns: { xs: "1fr", lg: "minmax(0, 1.25fr) minmax(320px, 0.9fr)" },
                }}
            >
                <SectionCard
                    title="Recent activity"
                    description="The latest notifications and alerts across your account."
                    action={
                        <Button variant="text" onClick={() => navigate("/notifications")}>
                            Open all
                        </Button>
                    }
                >
                    <QueryBoundary
                        isLoading={notificationsLoading}
                        isError={notificationsIsError}
                        error={notificationsError}
                        errorFallback="Failed to load notifications."
                        onRetry={() => void refetchNotifications()}
                        isEmpty={recentNotifications.length === 0}
                        emptyFallback={
                            <EmptyState
                                icon={<NotificationsIcon />}
                                title="No notifications yet"
                                description="Updates, reminders, and account events will appear here as soon as the workspace becomes active."
                                action={
                                    <Button variant="outlined" onClick={() => navigate("/projects")}>
                                        Explore workspace
                                    </Button>
                                }
                            />
                        }
                    >
                        <Stack spacing={1.5}>
                            {recentNotifications.map((notification) => (
                                <NotificationListItem
                                    key={notification.id}
                                    notification={notification}
                                    variant="compact"
                                />
                            ))}
                        </Stack>
                    </QueryBoundary>
                </SectionCard>

                <Stack spacing={2}>
                    <SectionCard title="Account health" description="A quick view of the settings that affect trust and security.">
                        <Stack spacing={1.5}>
                            {accountChecks.map((item) => (
                                <Box
                                    key={item.label}
                                    sx={(theme) => ({
                                        p: 2,
                                        borderRadius: 1,
                                        backgroundColor:
                                            theme.palette.mode === "dark"
                                                ? theme.palette.background.paper
                                                : colors.white,
                                    })}
                                >
                                    <Stack direction="row" justifyContent="space-between" spacing={1} sx={{ mb: 0.5 }}>
                                        <Typography variant="subtitle2">{item.label}</Typography>
                                        <Typography variant="body2" sx={{ color: item.color, fontWeight: 500 }}>
                                            {item.value}
                                        </Typography>
                                    </Stack>
                                    <Typography variant="body2" color="text.secondary">
                                        {item.description}
                                    </Typography>
                                </Box>
                            ))}
                        </Stack>
                    </SectionCard>

                    <DashboardCalendar
                        projects={projects ?? []}
                        projectsLoading={projectsLoading}
                        onOpenProjects={() => navigate("/projects")}
                        allowedViews={["month"]}
                        initialView="month"
                    />

                    <SectionCard
                        title={`${coreDomainPlural} snapshot`}
                        description={`A quick look at the current ${coreDomainPlural.toLowerCase()} in your workspace.`}
                    >
                        <QueryBoundary
                            isLoading={projectsLoading}
                            isError={projectsIsError}
                            error={projectsError}
                            errorFallback={`Failed to load ${coreDomainPlural.toLowerCase()}.`}
                            onRetry={() => void refetchProjects()}
                            isEmpty={!projects || projects.length === 0}
                            emptyFallback={
                                <EmptyState
                                    icon={<ProjectsIcon />}
                                    title={`No ${coreDomainPlural.toLowerCase()} yet`}
                                    description={`Create your first ${platformMetadata?.core_domain_singular?.toLowerCase() ?? "project"} to start building out the workspace.`}
                                    action={
                                        <Button variant="contained" onClick={() => navigate("/projects")}>
                                            Create first item
                                        </Button>
                                    }
                                />
                            }
                        >
                            <Stack spacing={1.25}>
                                {projects?.slice(0, 3).map((project) => (
                                    <Box
                                        key={project.id}
                                        sx={(theme) => ({
                                            p: 2,
                                            borderRadius: 1,
                                            backgroundColor:
                                                theme.palette.mode === "dark"
                                                    ? theme.palette.background.paper
                                                    : colors.white,
                                        })}
                                    >
                                        <Typography variant="subtitle2">{project.name}</Typography>
                                        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                                            {project.description || `No description added for this ${platformMetadata?.core_domain_singular?.toLowerCase() ?? "project"} yet.`}
                                        </Typography>
                                    </Box>
                                ))}
                            </Stack>
                        </QueryBoundary>
                    </SectionCard>
                </Stack>
            </Box>
        </PageShell>
    );
}
