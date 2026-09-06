import {
    Box,
    Button,
    Chip,
    Stack,
    Typography,
} from "@mui/material";
import { alpha } from "@mui/material/styles";
import type { Notification } from "../../api/notifications";
import { colors } from "../../app/designTokens";
import { formatDateTime, humanizeKey } from "../../utils/formatters";

type NotificationListItemProps = {
    notification: Notification;
    variant?: "compact" | "detailed";
    onMarkRead?: (id: string) => void;
    isMarkingRead?: boolean;
};

export function NotificationListItem({
    notification,
    variant = "detailed",
    onMarkRead,
    isMarkingRead = false,
}: NotificationListItemProps) {
    const isCompact = variant === "compact";
    const readStateLabel = notification.is_read ? "Read" : "Unread";
    const ariaLabel = `${readStateLabel} notification: ${notification.title}. ${notification.body ?? ""} Sent ${formatDateTime(notification.created_at)}.`;

    return (
        <Box
            role="article"
            aria-label={ariaLabel.trim()}
            sx={(theme) => ({
                borderRadius: isCompact ? 1 : 4,
                px: isCompact ? 2 : 2.25,
                py: isCompact ? 1.75 : 2.25,
                border: isCompact ? "none" : `1px solid ${theme.palette.divider}`,
                backgroundColor: notification.is_read
                    ? isCompact
                        ? "transparent"
                        : alpha(theme.palette.background.paper, 0.68)
                    : isCompact
                        ? theme.palette.mode === "dark"
                            ? "action.selected"
                            : colors.lightAsh
                        : alpha(
                              theme.palette.primary.main,
                              theme.palette.mode === "dark" ? 0.16 : 0.06
                          ),
            })}
        >
            <Stack spacing={isCompact ? 0 : 1.25}>
                <Stack
                    direction={{ xs: "column", sm: "row" }}
                    justifyContent="space-between"
                    spacing={1.5}
                >
                    <Box>
                        <Stack
                            direction="row"
                            spacing={1}
                            alignItems="center"
                            flexWrap="wrap"
                            useFlexGap
                            sx={{ mb: notification.body ? 0.5 : 0 }}
                        >
                            <Typography variant="subtitle2">{notification.title}</Typography>
                            {!isCompact && (
                                <Chip
                                    label={humanizeKey(notification.type)}
                                    size="small"
                                    variant="outlined"
                                />
                            )}
                            {!notification.is_read && (
                                <Chip label="New" size="small" color="primary" />
                            )}
                        </Stack>
                        {notification.body && (
                            <Typography variant="body2" color="text.secondary">
                                {notification.body}
                            </Typography>
                        )}
                    </Box>
                    <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ whiteSpace: "nowrap" }}
                    >
                        {formatDateTime(notification.created_at)}
                    </Typography>
                </Stack>
                {!notification.is_read && onMarkRead && (
                    <Box>
                        <Button
                            size="small"
                            variant="outlined"
                            disabled={isMarkingRead}
                            onClick={() => onMarkRead(notification.id)}
                        >
                            {isMarkingRead ? "Saving..." : "Mark as read"}
                        </Button>
                    </Box>
                )}
            </Stack>
        </Box>
    );
}
