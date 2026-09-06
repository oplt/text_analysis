import { apiFetch } from "./client";

export type ConfigEntry = {
    key: string;
    value: string;
    value_type: string;
    description: string | null;
    requires_restart: boolean;
    is_custom: boolean;
    is_secret: boolean;
};

export type ConfigSettingsResponse = {
    items: ConfigEntry[];
    notice: string;
};

export type ConfigSettingsUpdateRequest = {
    items: Array<{
        key: string;
        value: string;
    }>;
};

export type DatabaseSetting = {
    id: string;
    key: string;
    value: string;
    description: string | null;
    updated_at: string;
};

export async function getConfigSettings(): Promise<ConfigSettingsResponse> {
    return apiFetch("/settings/config");
}

export async function updateConfigSettings(
    payload: ConfigSettingsUpdateRequest
): Promise<ConfigSettingsResponse> {
    return apiFetch("/settings/config", {
        method: "PUT",
        body: JSON.stringify(payload),
    });
}

export async function listDatabaseSettings(): Promise<DatabaseSetting[]> {
    return apiFetch("/settings/database");
}

export async function createDatabaseSetting(payload: {
    key: string;
    value: string;
    description?: string;
}): Promise<DatabaseSetting> {
    return apiFetch("/settings/database", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updateDatabaseSetting(
    settingId: string,
    payload: {
        value?: string;
        description?: string | null;
    }
): Promise<DatabaseSetting> {
    return apiFetch(`/settings/database/${settingId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function deleteDatabaseSetting(settingId: string): Promise<void> {
    return apiFetch(`/settings/database/${settingId}`, {
        method: "DELETE",
    });
}
