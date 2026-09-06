import { useQuery, type UseQueryOptions } from "@tanstack/react-query";

import { getSessions, type Session } from "../api/users";
import { queryKeys } from "../config/queryKeys";
import { QUERY_STALE_TIMES } from "../config/queryTiming";

type UseUserSessionsOptions = Pick<UseQueryOptions<Session[]>, "enabled">;

export function useUserSessions(options?: UseUserSessionsOptions) {
    return useQuery({
        queryKey: queryKeys.users.sessions,
        queryFn: getSessions,
        staleTime: QUERY_STALE_TIMES.userSessions,
        ...options,
    });
}
