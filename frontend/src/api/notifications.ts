import { apiFetch, apiFetchItems } from "./client";

export type Notification = {
    id: string;
    type: string;
    title: string;
    body: string | null;
    is_read: boolean;
    created_at: string;
};

export type NotificationPreferences = {
    email_enabled: boolean;
    push_enabled: boolean;
    marketing_enabled: boolean;
};

export async function getNotifications(): Promise<Notification[]> {
    return apiFetchItems<Notification>("/notifications");
}

export async function markRead(id: string): Promise<void> {
    return apiFetch(`/notifications/${id}/read`, { method: "PATCH" });
}

export async function markAllRead(): Promise<void> {
    return apiFetch("/notifications/read-all", { method: "PATCH" });
}

export async function getPreferences(): Promise<NotificationPreferences> {
    return apiFetch("/notifications/preferences");
}

export async function updatePreferences(
    payload: Partial<NotificationPreferences>
): Promise<NotificationPreferences> {
    return apiFetch("/notifications/preferences", {
        method: "PUT",
        body: JSON.stringify(payload),
    });
}
