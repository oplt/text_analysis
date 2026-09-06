import { useQuery, type UseQueryOptions } from "@tanstack/react-query";

import { getProfile, type Profile } from "../api/profile";
import { queryKeys } from "../config/queryKeys";
import { QUERY_STALE_TIMES } from "../config/queryTiming";

type UseUserProfileOptions = Pick<UseQueryOptions<Profile>, "enabled">;

export function useUserProfile(options?: UseUserProfileOptions) {
    return useQuery({
        queryKey: queryKeys.users.profile,
        queryFn: getProfile,
        staleTime: QUERY_STALE_TIMES.userProfile,
        ...options,
    });
}
