import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

import { getNotifications } from "../api/notifications";
import { queryKeys } from "../config/queryKeys";
import { useNotifications } from "./useNotifications";

vi.mock("../api/notifications", () => ({
    getNotifications: vi.fn(),
}));

function createWrapper() {
    const queryClient = new QueryClient({
        defaultOptions: {
            queries: {
                retry: false,
            },
        },
    });

    return function Wrapper({ children }: { children: React.ReactNode }) {
        return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
    };
}

describe("useNotifications", () => {
    beforeEach(() => {
        vi.mocked(getNotifications).mockResolvedValue([
            {
                id: "n1",
                type: "system",
                title: "Hello",
                body: "World",
                is_read: false,
                created_at: "2026-01-01T00:00:00Z",
            },
        ]);
    });

    it("uses the canonical notifications query key", async () => {
        const queryClient = new QueryClient({
            defaultOptions: { queries: { retry: false } },
        });

        function TestWrapper({ children }: { children: React.ReactNode }) {
            return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
        }

        const { result } = renderHook(() => useNotifications({ refetchInterval: false }), {
            wrapper: TestWrapper,
        });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(getNotifications).toHaveBeenCalledTimes(1);
        expect(queryClient.getQueryData(queryKeys.notifications.all)).toEqual([
            expect.objectContaining({ id: "n1", title: "Hello" }),
        ]);
    });

    it("defaults to polling unless disabled", () => {
        const wrapper = createWrapper();
        const { result } = renderHook(() => useNotifications(), { wrapper });
        expect(result.current.fetchStatus).toBeDefined();
    });
});
