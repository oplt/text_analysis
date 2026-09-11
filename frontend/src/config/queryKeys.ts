import type { QueryClient } from "@tanstack/react-query";

export const queryKeys = {
    auth: {
        all: ["auth"] as const,
        me: ["auth", "me"] as const,
    },
    users: {
        all: ["users"] as const,
        me: ["users", "me"] as const,
        profile: ["users", "profile"] as const,
        sessions: ["users", "sessions"] as const,
        directory: ["users", "directory"] as const,
    },
    notifications: {
        all: ["notifications"] as const,
        preferences: ["notifications", "preferences"] as const,
    },
    projects: {
        all: ["projects"] as const,
        detail: (projectId: string) => ["projects", projectId] as const,
        tasks: (projectId: string) => ["projects", projectId, "tasks"] as const,
    },
    calendar: {
        all: ["calendar", "items"] as const,
        items: (start: string, end: string) => ["calendar", "items", start, end] as const,
    },
    admin: {
        all: ["admin"] as const,
        users: (page: number, search: string) => ["admin", "users", page, search] as const,
    },
    platform: {
        all: ["platform"] as const,
        metadata: ["platform", "metadata"] as const,
        plans: ["platform", "plans"] as const,
        subscription: ["platform", "subscription"] as const,
        apiKeys: ["platform", "api-keys"] as const,
        webhooks: ["platform", "webhooks"] as const,
        featureFlags: ["platform", "feature-flags"] as const,
        admin: {
            config: ["platform", "admin", "config"] as const,
            plans: ["platform", "admin", "plans"] as const,
            featureFlags: ["platform", "admin", "feature-flags"] as const,
            emailTemplates: ["platform", "admin", "email-templates"] as const,
        },
    },
    settings: {
        config: ["settings", "config"] as const,
        database: ["settings", "database"] as const,
    },
    observability: {
        links: ["observability", "links"] as const,
        status: ["observability", "status"] as const,
    },
    ai: {
        all: ["ai"] as const,
        overview: ["ai", "overview"] as const,
        reviews: ["ai", "reviews"] as const,
        evaluationRuns: ["ai", "evaluation-runs"] as const,
        promptVersions: (templateId: string) => ["ai", "prompt-versions", templateId] as const,
        datasetCases: (datasetId: string) => ["ai", "dataset-cases", datasetId] as const,
    },
    textResearch: {
        all: ["text-research"] as const,
        corpora: (projectId: string) => ["text-research", projectId, "corpora"] as const,
        corpus: (corpusId: string) => ["text-research", "corpus", corpusId] as const,
        documents: (corpusId: string, params?: Record<string, unknown>) =>
            ["text-research", "corpus", corpusId, "documents", params ?? {}] as const,
        codebooks: (projectId: string) => ["text-research", projectId, "codebooks"] as const,
        /** Version-aware: keyed by codebook id (frozen versions keep stable ids). */
        labels: (codebookId: string) => ["text-research", "codebook", codebookId, "labels"] as const,
        dashboard: (corpusId: string) => ["text-research", "corpus", corpusId, "dashboard"] as const,
        /** Prefix for invalidating every annotation-queue page/filter. */
        annotationQueueRoot: ["text-research", "annotation-queue"] as const,
        annotationQueue: (status?: string, offset = 0) =>
            ["text-research", "annotation-queue", status ?? "all", offset] as const,
        annotationProgress: (corpusId: string) =>
            ["text-research", "corpus", corpusId, "annotation-progress"] as const,
        runs: (projectId: string, corpusId?: string, runType?: string) =>
            ["text-research", projectId, "runs", corpusId ?? "all", runType ?? "all"] as const,
        run: (runId: string) => ["text-research", "run", runId] as const,
        runProvenance: (runId: string) => ["text-research", "run", runId, "provenance"] as const,
        /** Prefix for all classifier lists under a project. */
        classifiersRoot: (projectId: string) =>
            ["text-research", projectId, "classifiers"] as const,
        classifiers: (projectId: string, corpusId?: string) =>
            ["text-research", projectId, "classifiers", corpusId ?? "all"] as const,
        /** Prefix for model registry lists under a project (any corpus/lifecycle filter). */
        modelsRoot: (projectId: string) => ["text-research", projectId, "models"] as const,
        models: (projectId: string, corpusId?: string, lifecycleStatus?: string) =>
            [
                "text-research",
                projectId,
                "models",
                corpusId ?? "all",
                lifecycleStatus ?? "all",
            ] as const,
        model: (modelId: string) => ["text-research", "model", modelId] as const,
        predictionSets: (corpusId: string) =>
            ["text-research", "corpus", corpusId, "prediction-sets"] as const,
        predictionSet: (predictionSetId: string) =>
            ["text-research", "prediction-set", predictionSetId] as const,
        /** Prefix for prediction-set detail pages (filters/offset). */
        predictionSetPageRoot: (predictionSetId: string) =>
            ["text-research", "prediction-set-page", predictionSetId] as const,
        datasetSnapshots: (projectId: string, corpusId?: string) =>
            ["text-research", projectId, "snapshots", corpusId ?? "all"] as const,
        exportManifest: (corpusId: string) =>
            ["text-research", "corpus", corpusId, "export-manifest"] as const,
        preprocessingProfiles: (projectId: string) =>
            ["text-research", projectId, "preprocessing-profiles"] as const,
        cleaningProfiles: (projectId: string) =>
            ["text-research", projectId, "cleaning-profiles"] as const,
        dictionaries: (projectId: string) =>
            ["text-research", projectId, "dictionaries"] as const,
        metadataFacets: (corpusId: string) =>
            ["text-research", "metadata-facets", corpusId] as const,
        disagreements: (corpusId: string, codebookId: string) =>
            ["text-research", "corpus", corpusId, "disagreements", codebookId] as const,
        adjudications: (corpusId: string) =>
            ["text-research", "corpus", corpusId, "adjudications"] as const,
        topicLabels: (runId: string) =>
            ["text-research", "topic-labels", runId] as const,
        coefficients: (modelId: string) =>
            ["text-research", "coefficients", modelId] as const,
        contextualDatasets: (projectId: string) =>
            ["text-research", projectId, "contextual-datasets"] as const,
        contextualDataset: (datasetId: string) =>
            ["text-research", "contextual-dataset", datasetId] as const,
        /** Prefix for all active-learning queue pages for a model. */
        activeLearningRoot: (modelId: string) =>
            ["text-research", "active-learning", modelId] as const,
        activeLearningQueue: (
            modelId: string,
            params?: { offset?: number; limit?: number; campaignId?: string }
        ) =>
            [
                "text-research",
                "active-learning",
                modelId,
                params?.offset ?? 0,
                params?.limit ?? 20,
                params?.campaignId ?? "none",
            ] as const,
        modelLifecycleEvents: (modelId: string) =>
            ["text-research", "model", modelId, "lifecycle-events"] as const,
        annotationCampaigns: (projectId: string, corpusId?: string) =>
            ["text-research", projectId, "annotation-campaigns", corpusId ?? "all"] as const,
        assistantScope: (corpusId: string) =>
            ["text-research", "assistant", "scope", corpusId] as const,
        assistantThreads: (corpusId: string) =>
            ["text-research", "assistant", "threads", corpusId] as const,
        assistantConversation: (threadId: string) =>
            ["text-research", "assistant", "conversation", threadId] as const,
    },
    rag: {
        all: ["rag"] as const,
        documents: (projectId?: string, offset = 0) =>
            ["rag", "documents", projectId ?? "all", offset] as const,
        document: (documentId: string) => ["rag", "document", documentId] as const,
        chunks: (documentId: string, offset = 0) =>
            ["rag", "chunks", documentId, offset] as const,
        job: (jobId: string) => ["rag", "job", jobId] as const,
        queries: (offset = 0) => ["rag", "queries", offset] as const,
    },
    memory: {
        all: ["memory"] as const,
        list: (params?: Record<string, unknown>) => ["memory", "list", params ?? {}] as const,
        detail: (memoryId: string) => ["memory", "detail", memoryId] as const,
        audit: (offset = 0) => ["memory", "audit", offset] as const,
        search: (query: string, params?: Record<string, unknown>) =>
            ["memory", "search", query, params ?? {}] as const,
    },
    agent: {
        all: ["agent"] as const,
        runs: ["agent", "runs"] as const,
        run: (runId: string) => ["agent", "run", runId] as const,
    },
} as const;

export async function invalidateUserIdentity(queryClient: QueryClient) {
    await Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.auth.me }),
        queryClient.invalidateQueries({ queryKey: queryKeys.users.me }),
    ]);
}

export async function invalidateAiOverview(queryClient: QueryClient) {
    await queryClient.invalidateQueries({ queryKey: queryKeys.ai.overview });
}
