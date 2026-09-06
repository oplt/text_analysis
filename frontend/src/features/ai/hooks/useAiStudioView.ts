import { useQuery } from "@tanstack/react-query";
import { getAiOverview, listAiDatasetCases, listAiEvaluationRuns, listAiReviews, listPromptVersions } from "../../../api/ai";
import { useSnackbar } from "../../../app/snackbarContext";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES } from "../../../config/queryTiming";
import { useAiStudioForms } from "./useAiStudioForms";
import { useAiPromptAssetMutations } from "./useAiPromptAssetMutations";
import { useAiRunEvaluationMutations } from "./useAiRunEvaluationMutations";
export function useAiStudioView() {
    const { showToast } = useSnackbar();
    const forms = useAiStudioForms();
    const { selectedTemplateId, selectedDatasetId } = forms;
    const {
        data: overview,
        isLoading,
        isError,
        error,
        refetch,
    } = useQuery({
        queryKey: queryKeys.ai.overview,
        queryFn: getAiOverview,
        staleTime: QUERY_STALE_TIMES.aiOverview,
    });
    const {
        data: reviews = [],
        isLoading: reviewsLoading,
        isError: reviewsIsError,
        error: reviewsError,
        refetch: refetchReviews,
    } = useQuery({
        queryKey: queryKeys.ai.reviews,
        queryFn: listAiReviews,
        staleTime: QUERY_STALE_TIMES.aiReviews,
    });
    const {
        data: evaluationRuns = [],
        isLoading: evaluationRunsLoading,
        isError: evaluationRunsIsError,
        error: evaluationRunsError,
        refetch: refetchEvaluationRuns,
    } = useQuery({
        queryKey: queryKeys.ai.evaluationRuns,
        queryFn: listAiEvaluationRuns,
        staleTime: QUERY_STALE_TIMES.aiEvaluationRuns,
        refetchInterval: (query) => {
            const runs = query.state.data ?? [];
            return runs.some((run) => run.status === "running") ? 3_000 : false;
        },
    });

    const promptTemplates = overview?.prompt_templates ?? [];
    const documents = overview?.documents ?? [];
    const datasets = overview?.datasets ?? [];
    const recentRuns = overview?.recent_runs ?? [];
    const providers = overview?.providers ?? [];

    const {
        data: selectedTemplateVersions = [],
        isLoading: versionsLoading,
        isError: versionsIsError,
        error: versionsError,
        refetch: refetchVersions,
    } = useQuery({
        queryKey: queryKeys.ai.promptVersions(selectedTemplateId),
        queryFn: () => listPromptVersions(selectedTemplateId),
        enabled: selectedTemplateId.length > 0,
        staleTime: QUERY_STALE_TIMES.aiPromptVersions,
    });
    const {
        data: selectedDatasetCases = [],
        isLoading: datasetCasesLoading,
        isError: datasetCasesIsError,
        error: datasetCasesError,
        refetch: refetchDatasetCases,
    } = useQuery({
        queryKey: queryKeys.ai.datasetCases(selectedDatasetId),
        queryFn: () => listAiDatasetCases(selectedDatasetId),
        enabled: selectedDatasetId.length > 0,
        staleTime: QUERY_STALE_TIMES.aiDatasetCases,
    });

    const templateKeyOptions = promptTemplates.map((template) => ({
        id: template.id,
        key: template.key,
        name: template.name,
    }));

    const promptAssetMutations = useAiPromptAssetMutations(forms);
    const runEvaluationMutations = useAiRunEvaluationMutations(forms);

    return { overview, isLoading, isError, error, refetch, reviews, reviewsLoading, reviewsIsError, reviewsError, refetchReviews, evaluationRuns, evaluationRunsLoading, evaluationRunsIsError, evaluationRunsError, refetchEvaluationRuns,
        ...forms,
        promptTemplates, documents, datasets, recentRuns, providers, selectedTemplateVersions, versionsLoading, versionsIsError, versionsError, refetchVersions, selectedDatasetCases, datasetCasesLoading, datasetCasesIsError, datasetCasesError, refetchDatasetCases, templateKeyOptions,
        ...promptAssetMutations, ...runEvaluationMutations, showToast };
}
export type AiStudioModel = ReturnType<typeof useAiStudioView>;
