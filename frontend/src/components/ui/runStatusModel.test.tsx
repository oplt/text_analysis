import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { RunStatusChip } from "./RunStatusChip";
import { RunStatusPanel } from "./RunStatusPanel";
import {
    formatRunDurationMs,
    isActiveCanonicalStatus,
    normalizeRunStatus,
    runStatusLabel,
    runStatusTone,
} from "./runStatusModel";

describe("runStatusModel", () => {
    it("normalizes backend aliases to canonical statuses", () => {
        expect(normalizeRunStatus("pending")).toBe("queued");
        expect(normalizeRunStatus("processing")).toBe("running");
        expect(normalizeRunStatus("indexed")).toBe("completed");
        expect(normalizeRunStatus("canceled")).toBe("cancelled");
        expect(normalizeRunStatus("failed")).toBe("failed");
    });

    it("exposes consistent labels and tones", () => {
        expect(runStatusLabel("pending")).toBe("Queued");
        expect(runStatusTone("running")).toBe("warning");
        expect(runStatusTone("completed")).toBe("success");
        expect(isActiveCanonicalStatus("queued")).toBe(true);
        expect(isActiveCanonicalStatus("completed")).toBe(false);
    });

    it("formats durations", () => {
        expect(formatRunDurationMs(250)).toBe("250 ms");
        expect(formatRunDurationMs(1500)).toBe("1.5 s");
    });
});

describe("RunStatusChip", () => {
    it("renders canonical label", () => {
        render(<RunStatusChip status="pending" />);
        expect(screen.getByText("Queued")).toBeInTheDocument();
    });
});

describe("RunStatusPanel", () => {
    it("shows status, stage, failure, and retry control", () => {
        const onRetry = vi.fn();
        render(
            <RunStatusPanel
                status="failed"
                runId="run-1"
                stage="embedding"
                startedAt="2026-09-10T00:00:00Z"
                completedAt="2026-09-10T00:00:02Z"
                errorMessage="Worker crashed"
                onRetry={onRetry}
            />
        );
        expect(screen.getByText("Failed")).toBeInTheDocument();
        expect(screen.getByText("embedding")).toBeInTheDocument();
        expect(screen.getByText("Worker crashed")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    });
});
