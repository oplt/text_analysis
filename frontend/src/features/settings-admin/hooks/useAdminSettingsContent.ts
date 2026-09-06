import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
    createDatabaseSetting, deleteDatabaseSetting, updateConfigSettings, updateDatabaseSetting,
    type ConfigSettingsResponse, type DatabaseSetting,
} from "../../../api/settings";
import { queryKeys } from "../../../config/queryKeys";
import { useMutationErrorToast } from "../../../hooks/useMutationErrorToast";
import { buildConfigGroups, type DatabaseSettingDrafts, type SettingsTabValue } from "../settingsModel";

export function useAdminSettingsContent(configData: ConfigSettingsResponse, databaseSettings: DatabaseSetting[], activeTab: SettingsTabValue) {
    const queryClient = useQueryClient();
    const toastError = useMutationErrorToast();
    const configGroups = buildConfigGroups(configData.items);
    const [configDrafts, setConfigDrafts] = useState<Record<string, string>>(() => Object.fromEntries(configData.items.map((item) => [item.key, item.value])));
    const [databaseDrafts, setDatabaseDrafts] = useState<DatabaseSettingDrafts>(() => Object.fromEntries(databaseSettings.map((item) => [item.id, { value: item.value, description: item.description ?? "" }])));
    const [newSetting, setNewSetting] = useState({ key: "", value: "", description: "" });
    const configMutation = useMutation({
        mutationFn: updateConfigSettings,
        onSuccess: (data) => {
            queryClient.setQueryData(queryKeys.settings.config, data);
            setConfigDrafts(Object.fromEntries(data.items.map((item) => [item.key, item.value])));
        },
        onError: (error) => toastError(error, "Failed to save config."),
    });
    const refreshDatabase = () => queryClient.invalidateQueries({ queryKey: queryKeys.settings.database });
    const createDatabaseMutation = useMutation({
        mutationFn: createDatabaseSetting,
        onSuccess: async () => { setNewSetting({ key: "", value: "", description: "" }); await refreshDatabase(); },
        onError: (error) => toastError(error, "Failed to create database setting."),
    });
    const updateDatabaseMutation = useMutation({
        mutationFn: ({ id, value, description }: { id: string; value: string; description: string }) => updateDatabaseSetting(id, { value, description }),
        onSuccess: refreshDatabase,
        onError: (error) => toastError(error, "Failed to update database setting."),
    });
    const deleteDatabaseMutation = useMutation({
        mutationFn: deleteDatabaseSetting,
        onSuccess: refreshDatabase,
        onError: (error) => toastError(error, "Failed to delete database setting."),
    });
    const activeConfigGroup = activeTab === "database" ? null : configGroups.find((group) => group.id === activeTab) ?? configGroups[0] ?? null;
    const changedConfigCount = configData.items.filter((item) => (configDrafts[item.key] ?? item.value) !== item.value).length;
    return { configGroups, activeConfigGroup, changedConfigCount, configDrafts, setConfigDrafts,
        databaseDrafts, setDatabaseDrafts, newSetting, setNewSetting, configMutation,
        createDatabaseMutation, updateDatabaseMutation, deleteDatabaseMutation };
}

export type AdminSettingsModel = ReturnType<typeof useAdminSettingsContent>;
