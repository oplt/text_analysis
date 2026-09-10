import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
    Box,
    Button,
    FormControlLabel,
    Skeleton,
    Stack,
    Switch,
    Typography,
} from "@mui/material";
import {
    DoneAll as DoneAllIcon,
    MailOutline as MailOutlineIcon,
    NotificationsActive as NotificationsActiveIcon,
    Campaign as CampaignIcon,
} from "@mui/icons-material";
import {
    markAllRead,
    markRead,
    updatePreferences,
    type Notification,
    type NotificationPreferences,
} from "../api/notifications";
import { queryKeys } from "../config/queryKeys";
import { NotificationListItem } from "../components/notifications/NotificationListItem";
import { EmptyState } from "../components/ui/EmptyState";
import { useNotificationPreferences } from "../hooks/useNotificationPreferences";
import { useNotifications } from "../hooks/useNotifications";
import { PageShell } from "../components/ui/PageShell";
import { QueryBoundary } from "../components/ui/QueryBoundary";
import { SectionCard } from "../components/ui/SectionCard";
import { StatCard } from "../components/ui/StatCard";
import { useMutationErrorToast } from "../hooks/useMutationErrorToast";

function PreferenceItem({
    id,
    label,
    description,
    checked,
    disabled,
    onChange,
}: {
    id: string;
    label: string;
    description: string;
    checked: boolean;
    disabled: boolean;
    onChange: (nextValue: boolean) => void;
}) {
    const descriptionId = `${id}-description`;

    return (
        <Box
            sx={(theme) => ({
                p: 2,
                borderRadius: 4,
                border: `1px solid ${theme.palette.divider}`,
                backgroundColor: theme.palette.background.paper,
            })}
        >
            <FormControlLabel
                sx={{ alignItems: "flex-start", m: 0, width: "100%" }}
                control={
                    <Switch
                        checked={checked}
                        onChange={(event) => onChange(event.target.checked)}
                        disabled={disabled}
                        inputProps={{ "aria-describedby": descriptionId }}
                    />
                }
                label={
                    <Box sx={{ ml: 1 }}>
                        <Typography variant="subtitle2">{label}</Typography>
                        <Typography id={descriptionId} variant="body2" color="text.secondary">
                            {description}
                        </Typography>
                    </Box>
                }
            />
        </Box>
    );
}

export default function NotificationsPage() {
    const queryClient = useQueryClient();
    const toastMutationError = useMutationErrorToast();
    const {
        data: notifications,
        isLoading,
        isError,
        error,
        refetch,
    } = useNotifications({ refetchInterval: false });
    const {
        data: prefs,
        isLoading: prefsLoading,
        isError: prefsIsError,
        error: prefsError,
        refetch: refetchPrefs,
    } = useNotificationPreferences();

    const markOneMutation = useMutation({
        mutationFn: markRead,
        onMutate: async (id) => {
            await queryClient.cancelQueries({ queryKey: queryKeys.notifications.all });
            const previous = queryClient.getQueryData<Notification[]>(queryKeys.notifications.all);
            queryClient.setQueryData<Notification[]>(
                queryKeys.notifications.all,
                (old) => old?.map((item) => (item.id === id ? { ...item, is_read: true } : item)) ?? []
            );
            return { previous };
        },
        onError: (mutationError, _id, context) => {
            if (context?.previous) {
                queryClient.setQueryData(queryKeys.notifications.all, context.previous);
            }
            toastMutationError(mutationError, "Failed to mark notification as read.");
        },
        onSettled: () => void queryClient.invalidateQueries({ queryKey: queryKeys.notifications.all }),
    });
    const markAllMutation = useMutation({
        mutationFn: markAllRead,
        onMutate: async () => {
            await queryClient.cancelQueries({ queryKey: queryKeys.notifications.all });
            const previous = queryClient.getQueryData<Notification[]>(queryKeys.notifications.all);
            queryClient.setQueryData<Notification[]>(
                queryKeys.notifications.all,
                (old) => old?.map((item) => ({ ...item, is_read: true })) ?? []
            );
            return { previous };
        },
        onError: (mutationError, _vars, context) => {
            if (context?.previous) {
                queryClient.setQueryData(queryKeys.notifications.all, context.previous);
            }
            toastMutationError(mutationError, "Failed to mark all notifications as read.");
        },
        onSettled: () => void queryClient.invalidateQueries({ queryKey: queryKeys.notifications.all }),
    });
    const prefsMutation = useMutation({
        mutationFn: updatePreferences,
        onMutate: async (partial) => {
            await queryClient.cancelQueries({ queryKey: queryKeys.notifications.preferences });
            const previous = queryClient.getQueryData<NotificationPreferences>(
                queryKeys.notifications.preferences
            );
            queryClient.setQueryData<NotificationPreferences>(
                queryKeys.notifications.preferences,
                (old) => ({
                    email_enabled: old?.email_enabled ?? true,
                    push_enabled: old?.push_enabled ?? true,
                    marketing_enabled: old?.marketing_enabled ?? false,
                    ...partial,
                })
            );
            return { previous };
        },
        onError: (mutationError, _partial, context) => {
            if (context?.previous) {
                queryClient.setQueryData(queryKeys.notifications.preferences, context.previous);
            }
            toastMutationError(mutationError, "Failed to update notification preferences.");
        },
        onSettled: () =>
            void queryClient.invalidateQueries({ queryKey: queryKeys.notifications.preferences }),
    });

    const unreadCount = notifications?.filter((item) => !item.is_read).length ?? 0;
    const totalCount = notifications?.length ?? 0;
    const enabledChannels = prefs
        ? [prefs.email_enabled, prefs.push_enabled, prefs.marketing_enabled].filter(Boolean).length
        : 0;

    return (
        <PageShell width="wide">
            {unreadCount > 0 ? (
                <Stack direction="row" justifyContent="flex-end" sx={{ mb: 2 }}>
                    <Button
                        variant="contained"
                        startIcon={<DoneAllIcon />}
                        disabled={markAllMutation.isPending}
                        onClick={() => markAllMutation.mutate()}
                    >
                        {markAllMutation.isPending ? "Updating..." : "Mark all read"}
                    </Button>
                </Stack>
            ) : null}

            <Box
                sx={{
                    display: "grid",
                    gap: 2,
                    gridTemplateColumns: { xs: "1fr", md: "repeat(3, minmax(0, 1fr))" },
                }}
            >
                <StatCard
                    label="Unread"
                    value={unreadCount}
                    description="Fresh activity that still needs attention"
                    icon={<NotificationsActiveIcon />}
                    loading={isLoading}
                />
                <StatCard
                    label="All notifications"
                    value={totalCount}
                    description="Historical inbox items available for review"
                    icon={<MailOutlineIcon />}
                    loading={isLoading}
                    color="secondary"
                />
                <StatCard
                    label="Channels enabled"
                    value={prefsLoading ? "—" : `${enabledChannels}/3`}
                    description="Delivery routes currently turned on"
                    icon={<CampaignIcon />}
                    loading={prefsLoading}
                    color="success"
                />
            </Box>

            <Box
                sx={{
                    display: "grid",
                    gap: 2,
                    gridTemplateColumns: { xs: "1fr", lg: "minmax(0, 1.25fr) minmax(320px, 0.85fr)" },
                }}
            >
                <SectionCard title="Inbox" description="Messages are sorted for fast scanning and clear read state.">
                    <QueryBoundary
                        isLoading={isLoading}
                        isError={isError}
                        error={error}
                        errorFallback="Failed to load notifications."
                        onRetry={() => void refetch()}
                        isEmpty={!notifications || notifications.length === 0}
                        emptyFallback={
                            <EmptyState
                                icon={<NotificationsActiveIcon />}
                                title="Inbox is clear"
                                description="You have no notifications yet. New product updates and account events will appear here."
                            />
                        }
                    >
                        <Stack spacing={1.5}>
                            {notifications?.map((notification) => {
                                const isUpdatingThisItem =
                                    markOneMutation.isPending &&
                                    markOneMutation.variables === notification.id;
                                return (
                                    <NotificationListItem
                                        key={notification.id}
                                        notification={notification}
                                        onMarkRead={(id) => markOneMutation.mutate(id)}
                                        isMarkingRead={isUpdatingThisItem}
                                    />
                                );
                            })}
                        </Stack>
                    </QueryBoundary>
                </SectionCard>

                <SectionCard title="Delivery preferences" description="Choose how you want this workspace to reach you.">
                    <QueryBoundary
                        isLoading={prefsLoading}
                        isError={prefsIsError}
                        error={prefsError}
                        errorFallback="Failed to load notification preferences."
                        onRetry={() => void refetchPrefs()}
                        loadingFallback={
                            <Stack spacing={1.5}>
                                {Array.from({ length: 3 }).map((_, index) => (
                                    <Skeleton key={index} variant="rounded" height={88} sx={{ borderRadius: 4 }} />
                                ))}
                            </Stack>
                        }
                    >
                        <Stack spacing={1.5}>
                            <PreferenceItem
                                id="email-notifications"
                                label="Email notifications"
                                description="Receive operational updates and account messages in your inbox."
                                checked={prefs?.email_enabled ?? false}
                                disabled={prefsMutation.isPending}
                                onChange={(nextValue) => prefsMutation.mutate({ email_enabled: nextValue })}
                            />
                            <PreferenceItem
                                id="push-notifications"
                                label="Push notifications"
                                description="Surface urgent activity directly inside the app experience."
                                checked={prefs?.push_enabled ?? false}
                                disabled={prefsMutation.isPending}
                                onChange={(nextValue) => prefsMutation.mutate({ push_enabled: nextValue })}
                            />
                            <PreferenceItem
                                id="marketing-emails"
                                label="Marketing emails"
                                description="Get launch announcements, feature roundups, and educational updates."
                                checked={prefs?.marketing_enabled ?? false}
                                disabled={prefsMutation.isPending}
                                onChange={(nextValue) => prefsMutation.mutate({ marketing_enabled: nextValue })}
                            />
                        </Stack>
                    </QueryBoundary>
                </SectionCard>
            </Box>
        </PageShell>
    );
}
