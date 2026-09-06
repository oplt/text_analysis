import { useQuery, type UseQueryOptions } from "@tanstack/react-query";

import { getMe, type UserProfile } from "../api/users";
import { queryKeys } from "../config/queryKeys";
import { QUERY_STALE_TIMES } from "../config/queryTiming";

type UseCurrentUserOptions = Pick<UseQueryOptions<UserProfile>, "enabled">;

export function useCurrentUser(options?: UseCurrentUserOptions) {
    return useQuery({
        queryKey: queryKeys.users.me,
        queryFn: getMe,
        staleTime: QUERY_STALE_TIMES.userProfile,
        ...options,
    });
}
