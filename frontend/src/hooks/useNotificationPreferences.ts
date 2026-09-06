import { useQuery, type UseQueryOptions } from "@tanstack/react-query";

import { getPreferences, type NotificationPreferences } from "../api/notifications";
import { queryKeys } from "../config/queryKeys";
import { QUERY_STALE_TIMES } from "../config/queryTiming";

type UseNotificationPreferencesOptions = Pick<UseQueryOptions<NotificationPreferences>, "enabled">;

export function useNotificationPreferences(options?: UseNotificationPreferencesOptions) {
    return useQuery({
        queryKey: queryKeys.notifications.preferences,
        queryFn: getPreferences,
        staleTime: QUERY_STALE_TIMES.notificationPreferences,
        ...options,
    });
}
