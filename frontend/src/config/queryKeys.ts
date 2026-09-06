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
