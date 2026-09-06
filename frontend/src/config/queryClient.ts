import { QueryClient } from '@tanstack/react-query'

import { QUERY_GC_TIME_MS, QUERY_STALE_TIMES } from './queryTiming'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: QUERY_STALE_TIMES.default,
      gcTime: QUERY_GC_TIME_MS,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})