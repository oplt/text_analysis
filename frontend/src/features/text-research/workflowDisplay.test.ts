import { describe, expect, it } from "vitest";
import { compactStageDetail, workflowProgress } from "./workflowDisplay";
import type { WorkflowStageState } from "./workflow";

function stage(
    partial: Partial<WorkflowStageState> & Pick<WorkflowStageState, "id" | "status">
): WorkflowStageState {
    return {
        label: partial.id,
        route: "corpus",
        description: "",
        detail: null,
        blockedReason: null,
        ...partial,
    };
}

describe("workflowDisplay", () => {
    it("compacts annotation progress", () => {
        expect(
            compactStageDetail(
                stage({
                    id: "annotate",
                    status: "warning",
                    detail: "3/20 completed",
                })
            )
        ).toBe("3/20");
    });

    it("compacts reliability kappa detail", () => {
        expect(
            compactStageDetail(
                stage({
                    id: "reliability",
                    status: "complete",
                    detail: "κ 0.82",
                })
            )
        ).toBe("κ ready");
    });

    it("computes workflow progress from complete and warning stages", () => {
        const progress = workflowProgress([
            stage({ id: "corpus", status: "complete" }),
            stage({ id: "prepare", status: "complete" }),
            stage({ id: "annotate", status: "warning" }),
            stage({ id: "analyze", status: "current" }),
            stage({ id: "export", status: "incomplete" }),
        ]);
        expect(progress.completed).toBe(3);
        expect(progress.total).toBe(5);
        expect(progress.percent).toBe(60);
        expect(progress.current?.id).toBe("analyze");
    });
});
