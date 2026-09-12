import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { getDashboardSummary } from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";
import { useResearchContext } from "../hooks/useResearchContext";
import { deriveWorkflowStages, type WorkflowStageState } from "../workflow";

export function useResearchWorkflow(activeRoute: string): {
    stages: WorkflowStageState[];
    isLoading: boolean;
} {
    const ctx = useResearchContext();

    const dashboardQuery = useQuery({
        queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId),
        queryFn: ({ signal }) => getDashboardSummary(ctx.selectedCorpusId, signal),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const stages = useMemo(
        () =>
            deriveWorkflowStages({
                activeRoute,
                hasCorpus: Boolean(ctx.selectedCorpusId) || ctx.corpora.length > 0,
                hasCodebook: Boolean(ctx.selectedCodebookId) || ctx.codebooks.length > 0,
                labelCount: ctx.labels.length,
                codebook: ctx.selectedCodebook
                    ? {
                          name: ctx.selectedCodebook.name,
                          version: ctx.selectedCodebook.version,
                          is_frozen: ctx.selectedCodebook.is_frozen,
                      }
                    : null,
                unitType: ctx.unitType,
                summary: dashboardQuery.data,
            }),
        [
            activeRoute,
            ctx.selectedCorpusId,
            ctx.corpora.length,
            ctx.selectedCodebookId,
            ctx.codebooks.length,
            ctx.labels.length,
            ctx.selectedCodebook,
            ctx.unitType,
            dashboardQuery.data,
        ]
    );

    return {
        stages,
        isLoading: Boolean(ctx.selectedCorpusId) && dashboardQuery.isLoading,
    };
}
