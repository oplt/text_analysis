import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ObservabilityPage from "./ObservabilityPage";
import { getObservabilityLinks, getObservabilityStatus } from "../features/observability/api";
import type {
    ObservabilityLinks,
    ObservabilityStatus,
} from "../features/observability/types";
import { useAuth } from "../hooks/useAuth";

vi.mock("../hooks/useAuth", () => ({
    useAuth: vi.fn(() => ({ isAdmin: true })),
}));

vi.mock("../hooks/usePlatformMetadata", () => ({
    usePlatformMetadata: vi.fn(() => ({ data: { module_catalog: [] } })),
}));

vi.mock("../features/observability/api", () => ({
    getObservabilityLinks: vi.fn(),
    getObservabilityStatus: vi.fn(),
}));

const links: ObservabilityLinks = {
    grafana_base_url: { url: "http://localhost:3001", configured: true, allowed: true },
    prometheus_url: { url: "http://localhost:9090/graph", configured: true, allowed: true },
    tempo_explore_url: { url: "http://localhost:3001/explore", configured: true, allowed: true },
    dashboards: {
        application_overview: {
            url: "http://localhost:3001/d/fastapi-overview/fastapi-overview",
            configured: true,
            allowed: true,
        },
        api: { url: "http://localhost:3001/d/api/backend-api", configured: true, allowed: true },
        frontend: { url: null, configured: false, allowed: true },
        database: { url: null, configured: false, allowed: true },
        cache: { url: null, configured: false, allowed: true },
        workers: { url: null, configured: false, allowed: true },
        scheduled_tasks: { url: null, configured: false, allowed: true },
        errors: { url: "http://localhost:3001/d/errors/errors", configured: true, allowed: true },
    },
};

const status: ObservabilityStatus = {
    api: { status: "healthy", detail: "API is responding", last_checked_at: "2026-06-13T12:00:00Z" },
    frontend: { status: "unknown", detail: "Frontend status check is not configured" },
    database: { status: "healthy", detail: "Database connection OK" },
    cache: { status: "unknown", detail: "Cache check is unavailable" },
    workers: { status: "unknown", detail: "Worker queue depth check is not configured", queue_depth: null },
    background_jobs: { status: "unknown", detail: "Background job health check is not configured" },
    error_rate: { status: "unknown", detail: "Error-rate summary is available in Grafana" },
    request_latency: { status: "unknown", detail: "Request latency summary is available in Grafana" },
    prometheus: { status: "healthy", detail: "Prometheus is reachable", url: "http://localhost:9090" },
    grafana: { status: "healthy", detail: "Grafana is reachable", url: "http://localhost:3001" },
    tempo: { status: "unknown", detail: "Tempo is not reachable", url: "http://localhost:3200" },
};

function renderPage() {
    const queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
    });

    return render(
        <QueryClientProvider client={queryClient}>
            <MemoryRouter initialEntries={["/observability"]}>
                <ObservabilityPage />
            </MemoryRouter>
        </QueryClientProvider>
    );
}

describe("ObservabilityPage", () => {
    beforeEach(() => {
        vi.mocked(useAuth).mockReturnValue({ isAdmin: true } as ReturnType<typeof useAuth>);
        vi.mocked(getObservabilityLinks).mockResolvedValue(links);
        vi.mocked(getObservabilityStatus).mockResolvedValue(status);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("renders a loading state", () => {
        vi.mocked(getObservabilityLinks).mockReturnValue(new Promise(() => undefined));
        vi.mocked(getObservabilityStatus).mockReturnValue(new Promise(() => undefined));

        renderPage();

        expect(screen.getByText("System health")).toBeInTheDocument();
    });

    it("renders health and shortcut cards", async () => {
        renderPage();

        expect(await screen.findByText("API Status")).toBeInTheDocument();
        expect(screen.getByText("Application Overview")).toBeInTheDocument();
        expect(screen.getByText("Backend API")).toBeInTheDocument();
    });

    it("shows not configured for missing URLs and avoids enabled broken buttons", async () => {
        renderPage();

        expect(await screen.findByText("Frontend Performance")).toBeInTheDocument();
        expect(screen.getAllByText("Not configured").length).toBeGreaterThan(0);
        expect(screen.getByRole("button", { name: /open frontend view/i })).toBeDisabled();
    });

    it("hides Prometheus Debug for non-admin users", async () => {
        vi.mocked(useAuth).mockReturnValue({ isAdmin: false } as ReturnType<typeof useAuth>);

        renderPage();

        await waitFor(() => expect(screen.queryByText("Prometheus Debug")).not.toBeInTheDocument());
    });

    it("opens Grafana links in a new tab", async () => {
        const open = vi.spyOn(window, "open").mockImplementation(() => null);
        renderPage();

        await userEvent.click(await screen.findByRole("button", { name: /open api view/i }));

        expect(open).toHaveBeenCalledWith(
            expect.stringContaining("http://localhost:3001/d/api/backend-api"),
            "_blank",
            "noopener,noreferrer"
        );
    });
});
