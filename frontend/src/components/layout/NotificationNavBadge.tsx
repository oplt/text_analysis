import { memo } from "react";
import { Badge } from "@mui/material";
import { useNotifications } from "../../hooks/useNotifications";

type NotificationNavBadgeProps = {
    children: React.ReactNode;
};

function NotificationNavBadgeInner({ children }: NotificationNavBadgeProps) {
    const { data: notifications } = useNotifications();
    const unreadCount = notifications?.filter((notification) => !notification.is_read).length ?? 0;

    if (unreadCount <= 0) {
        return <>{children}</>;
    }

    return (
        <Badge badgeContent={unreadCount} color="error">
            {children}
        </Badge>
    );
}

export const NotificationNavBadge = memo(NotificationNavBadgeInner);
