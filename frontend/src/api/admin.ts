import { apiFetch } from "./client";

export type AdminUser = {
    id: string;
    email: string;
    full_name: string | null;
    is_verified: boolean;
    is_active: boolean;
    created_at: string;
    roles: string[];
};

export type AdminUserListResponse = {
    items: AdminUser[];
    total: number;
    page: number;
    page_size: number;
};

export async function listAdminUsers(params?: {
    page?: number;
    page_size?: number;
    search?: string;
}): Promise<AdminUserListResponse> {
    const qs = new URLSearchParams();
    if (params?.page) qs.set("page", String(params.page));
    if (params?.page_size) qs.set("page_size", String(params.page_size));
    if (params?.search) qs.set("search", params.search);
    return apiFetch(`/admin/users?${qs.toString()}`);
}

export async function updateUserStatus(
    userId: string,
    payload: { is_active: boolean }
): Promise<AdminUser> {
    return apiFetch(`/admin/users/${userId}/status`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}
