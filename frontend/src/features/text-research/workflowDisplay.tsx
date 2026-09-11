import type { WorkflowStageState, WorkflowStatus } from "./workflow";

export const STATUS_LABEL: Record<WorkflowStatus, string> = {
    complete: "Complete",
    current: "Current",
    incomplete: "Ready",
    blocked: "Blocked",
    warning: "Attention",
};

export function statusAccent(status: WorkflowStatus): string {
    switch (status) {
        case "complete":
            return "success.main";
        case "current":
            return "primary.main";
        case "warning":
            return "warning.main";
        case "blocked":
            return "text.disabled";
        default:
            return "text.secondary";
    }
}

/** Short secondary line for the compact strip (e.g. "3/20", "κ ready"). */
export function compactStageDetail(stage: WorkflowStageState): string | null {
    if (stage.status === "blocked") return null;
    if (!stage.detail) return null;

    if (stage.id === "annotate") {
        const match = stage.detail.match(/^(\d+\/\d+)/);
        return match?.[1] ?? stage.detail;
    }
    if (stage.id === "reliability") {
        if (stage.detail.startsWith("κ")) return "κ ready";
        return stage.detail.length > 18 ? stage.detail.slice(0, 16) + "…" : stage.detail;
    }
    if (stage.id === "codebook") {
        const labels = stage.detail.match(/(\d+)\s+labels?/);
        if (labels) return `${labels[1]} labels`;
        return null;
    }
    if (stage.detail.length > 18) {
        return stage.detail.slice(0, 16) + "…";
    }
    return stage.detail;
}

export function workflowProgress(stages: WorkflowStageState[]): {
    completed: number;
    total: number;
    percent: number;
    currentIndex: number;
    current: WorkflowStageState | null;
} {
    const completed = stages.filter(
        (stage) => stage.status === "complete" || stage.status === "warning"
    ).length;
    const total = stages.length;
    const currentIndex = Math.max(
        0,
        stages.findIndex((stage) => stage.status === "current")
    );
    const current =
        stages.find((stage) => stage.status === "current") ??
        stages[currentIndex] ??
        null;
    return {
        completed,
        total,
        percent: total === 0 ? 0 : Math.round((completed / total) * 100),
        currentIndex: current ? stages.indexOf(current) : 0,
        current,
    };
}

export function stageActionLabel(stage: WorkflowStageState): string {
    switch (stage.id) {
        case "annotate":
            return stage.status === "incomplete" ? "Start annotation" : "Continue annotation";
        case "reliability":
            return "View reliability";
        case "analyze":
            return stage.status === "incomplete" ? "Start analysis" : "Open analysis";
        case "export":
            return "Open exports";
        default:
            return `Open ${stage.label.toLowerCase()}`;
    }
}
