import { useQuery } from "@tanstack/react-query";

import { getPlatformMetadata } from "../api/platform";
import { queryKeys } from "../config/queryKeys";
import { QUERY_STALE_TIMES } from "../config/queryTiming";

export function usePlatformMetadata() {
    return useQuery({
        queryKey: queryKeys.platform.metadata,
        queryFn: getPlatformMetadata,
        staleTime: QUERY_STALE_TIMES.platformMetadata,
    });
}
