import { describe, expect, it, vi } from "vitest";
import { buildGrafanaUrl, buildTempoExploreUrl, openExternalUrl } from "./urlBuilders";

describe("observability URL builders", () => {
    it("adds default Grafana time and org parameters", () => {
        const url = buildGrafanaUrl("http://localhost:3001/d/api/backend-api");

        expect(url).toBe("http://localhost:3001/d/api/backend-api?orgId=1&from=now-1h&to=now");
    });

    it("adds safe dashboard variables and encodes route values", () => {
        const url = buildGrafanaUrl("http://localhost:3001/d/api/backend-api", {
            service: "backend",
            environment: "local",
            route: "/api/v1/users",
            jobName: "email",
        });

        expect(url).toContain("var-service=backend");
        expect(url).toContain("var-environment=local");
        expect(url).toContain("var-route=%2Fapi%2Fv1%2Fusers");
        expect(url).toContain("var-job=email");
    });

    it("does not add empty optional values", () => {
        const url = buildGrafanaUrl("http://localhost:3001/d/api/backend-api", {
            service: "",
            route: "   ",
        });

        expect(url).not.toContain("var-service");
        expect(url).not.toContain("var-route");
    });

    it("keeps request and trace IDs for trace exploration", () => {
        const url = buildTempoExploreUrl("http://localhost:3001/explore", {
            traceId: "abc123",
            requestId: "req-1",
        });

        expect(url).toContain("traceId=abc123");
        expect(url).toContain("requestId=req-1");
    });

    it("opens external URLs without opener access", () => {
        const open = vi.spyOn(window, "open").mockImplementation(() => null);

        openExternalUrl("http://localhost:3001");

        expect(open).toHaveBeenCalledWith("http://localhost:3001", "_blank", "noopener,noreferrer");
        open.mockRestore();
    });
});
