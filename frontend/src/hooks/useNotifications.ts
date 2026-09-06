import { useQuery, type UseQueryOptions } from "@tanstack/react-query";

import { getNotifications, type Notification } from "../api/notifications";
import { queryKeys } from "../config/queryKeys";
import {
    NOTIFICATIONS_REFETCH_INTERVAL_MS,
    QUERY_STALE_TIMES,
} from "../config/queryTiming";

type UseNotificationsOptions = Pick<
    UseQueryOptions<Notification[]>,
    "enabled" | "refetchInterval" | "refetchOnWindowFocus"
>;

export function useNotifications(options?: UseNotificationsOptions) {
    return useQuery({
        queryKey: queryKeys.notifications.all,
        queryFn: getNotifications,
        staleTime: QUERY_STALE_TIMES.notifications,
        refetchInterval: NOTIFICATIONS_REFETCH_INTERVAL_MS,
        ...options,
    });
}
