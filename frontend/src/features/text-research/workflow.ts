import type { DashboardSummary, ResearchTabSlug, UnitType } from "./types";

export type WorkflowStatus = "complete" | "current" | "incomplete" | "blocked" | "warning";

export type WorkflowStageId =
    | "corpus"
    | "prepare"
    | "codebook"
    | "annotate"
    | "reliability"
    | "analyze"
    | "topics"
    | "classify"
    | "validate"
    | "explore"
    | "contextual"
    | "export";

export type WorkflowStageDefinition = {
    id: WorkflowStageId;
    label: string;
    /** Route segment under `/research/:projectId/` */
    route: ResearchTabSlug | "prepare";
    description: string;
};

/** Ordered research workflow — primary navigation for text research. */
export const RESEARCH_WORKFLOW_STAGES: WorkflowStageDefinition[] = [
    {
        id: "corpus",
        label: "Corpus",
        route: "corpus",
        description: "Collect and link source documents",
    },
    {
        id: "prepare",
        label: "Prepare",
        route: "prepare",
        description: "Segment documents into research text units",
    },
    {
        id: "codebook",
        label: "Codebook",
        route: "codebook",
        description: "Operationalize concepts and manage versioned labels",
    },
    {
        id: "annotate",
        label: "Annotate",
        route: "annotation",
        description: "Code units with the project codebook",
    },
    {
        id: "reliability",
        label: "Reliability",
        route: "reliability",
        description: "Measure inter-annotator agreement",
    },
    {
        id: "analyze",
        label: "Analyze",
        route: "analysis",
        description: "Run frequency, keyness, and related analyses",
    },
    {
        id: "topics",
        label: "Topics",
        route: "topics",
        description: "Fit topic models on the corpus",
    },
    {
        id: "classify",
        label: "Classify",
        route: "classification",
        description: "Train and apply text classifiers",
    },
    {
        id: "validate",
        label: "Validate",
        route: "robustness",
        description: "Run controlled robustness checks",
    },
    {
        id: "explore",
        label: "Explore",
        route: "explorer",
        description: "Inspect results across documents and units",
    },
    {
        id: "contextual",
        label: "Contextual",
        route: "contextual",
        description: "Join descriptive indicators to discourse prevalence",
    },
    {
        id: "export",
        label: "Export",
        route: "exports",
        description: "Export datasets, runs, and artifacts",
    },
];

export type WorkflowStageState = WorkflowStageDefinition & {
    status: WorkflowStatus;
    detail: string | null;
    blockedReason: string | null;
};

type WorkflowInput = {
    activeRoute: string;
    hasCorpus: boolean;
    hasCodebook: boolean;
    labelCount: number;
    codebook?: { name: string; version: string; is_frozen: boolean } | null;
    unitType: UnitType;
    summary: DashboardSummary | null | undefined;
};

function runCount(summary: DashboardSummary | null | undefined, type: string): number {
    return summary?.analysis_run_counts_by_type?.[type] ?? 0;
}

function formatKappa(summary: DashboardSummary | null | undefined): string | null {
    const metrics = summary?.latest_reliability?.metrics;
    if (!metrics || typeof metrics !== "object") return null;

    const mean = metrics.mean_cohens_kappa;
    if (typeof mean === "number" && Number.isFinite(mean)) {
        return `κ ${mean.toFixed(2)}`;
    }

    for (const value of Object.values(metrics)) {
        if (value && typeof value === "object" && "cohens_kappa" in value) {
            const kappa = (value as { cohens_kappa?: { kappa?: number } }).cohens_kappa?.kappa;
            if (typeof kappa === "number" && Number.isFinite(kappa)) {
                return `κ ${kappa.toFixed(2)}`;
            }
        }
    }

    return "κ available";
}

function applyCurrent(
    stages: WorkflowStageState[],
    activeRoute: string
): WorkflowStageState[] {
    const activeId = RESEARCH_WORKFLOW_STAGES.find((stage) => stage.route === activeRoute)?.id;
    if (!activeId) return stages;

    return stages.map((stage) => {
        if (stage.id !== activeId) return stage;
        if (stage.status === "blocked") return stage;
        return { ...stage, status: "current" };
    });
}

export function deriveWorkflowStages(input: WorkflowInput): WorkflowStageState[] {
    const { summary, unitType, hasCorpus, hasCodebook, labelCount, codebook = null, activeRoute } = input;
    const documentCount = summary?.document_count ?? 0;
    const unitCount = summary?.text_unit_counts?.[unitType] ?? 0;
    const anyUnits = Object.values(summary?.text_unit_counts ?? {}).some((count) => count > 0);
    const annotationTasks = summary?.annotation_task_count ?? 0;
    const annotationCompleted = summary?.annotation_completed_count ?? 0;
    const annotationRate = summary?.annotation_completion_rate ?? 0;
    const modelCount = summary?.trained_model_count ?? 0;
    const reliabilityRuns = runCount(summary, "reliability");
    const topicRuns = runCount(summary, "topic_model");
    const analysisRuns =
        runCount(summary, "frequency_analysis") +
        runCount(summary, "keyness") +
        runCount(summary, "dfm") +
        runCount(summary, "ngram_analysis") +
        runCount(summary, "dictionary_analysis") +
        runCount(summary, "kwic") +
        runCount(summary, "cooccurrence") +
        runCount(summary, "comparative_analysis") +
        runCount(summary, "corpus_stats");
    const robustnessRuns = runCount(summary, "robustness");
    const exportRuns = runCount(summary, "export");
    const kappaLabel = formatKappa(summary);
    const hasLabels = hasCodebook && labelCount > 0;

    const stages: WorkflowStageState[] = RESEARCH_WORKFLOW_STAGES.map((stage) => {
        switch (stage.id) {
            case "corpus": {
                if (!hasCorpus) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Create or select a corpus first",
                    };
                }
                if (documentCount > 0) {
                    return {
                        ...stage,
                        status: "complete",
                        detail: `${documentCount} document${documentCount === 1 ? "" : "s"}`,
                        blockedReason: null,
                    };
                }
                return {
                    ...stage,
                    status: "incomplete",
                    detail: "0 documents",
                    blockedReason: null,
                };
            }
            case "prepare": {
                if (!hasCorpus || documentCount === 0) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Add documents to the corpus before segmentation",
                    };
                }
                if (unitCount > 0 || anyUnits) {
                    const counts = summary?.text_unit_counts ?? {};
                    const count =
                        unitCount > 0
                            ? unitCount
                            : Object.values(counts).reduce((a, b) => a + b, 0);
                    return {
                        ...stage,
                        status: "complete",
                        detail: `${count.toLocaleString()} ${unitType} units`,
                        blockedReason: null,
                    };
                }
                return {
                    ...stage,
                    status: "incomplete",
                    detail: `Segment into ${unitType} units`,
                    blockedReason: null,
                };
            }
            case "annotate": {
                if (!anyUnits && unitCount === 0) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Prepare text units before annotation",
                    };
                }
                if (!hasLabels) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Create a codebook with labels",
                    };
                }
                if (annotationTasks === 0) {
                    return {
                        ...stage,
                        status: "incomplete",
                        detail: "No tasks yet",
                        blockedReason: null,
                    };
                }
                const detail = `${annotationCompleted}/${annotationTasks} completed`;
                if (annotationRate >= 1) {
                    return { ...stage, status: "complete", detail, blockedReason: null };
                }
                if (annotationCompleted > 0) {
                    return { ...stage, status: "warning", detail, blockedReason: null };
                }
                return { ...stage, status: "incomplete", detail, blockedReason: null };
            }
            case "codebook": {
                if (!hasCodebook || !codebook) {
                    return {
                        ...stage,
                        status: "incomplete",
                        detail: "No codebook selected",
                        blockedReason: null,
                    };
                }
                return {
                    ...stage,
                    status: labelCount > 0 ? "complete" : "incomplete",
                    detail: `${codebook.name} · v${codebook.version} · ${codebook.is_frozen ? "frozen" : "draft"} · ${labelCount} label${labelCount === 1 ? "" : "s"}`,
                    blockedReason: null,
                };
            }
            case "reliability": {
                if (annotationCompleted === 0) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Complete overlapping annotations first",
                    };
                }
                if (summary?.latest_reliability || reliabilityRuns > 0) {
                    return {
                        ...stage,
                        status: "complete",
                        detail: kappaLabel,
                        blockedReason: null,
                    };
                }
                return {
                    ...stage,
                    status: "incomplete",
                    detail: "Agreement not computed",
                    blockedReason: null,
                };
            }
            case "analyze": {
                if (!anyUnits && unitCount === 0) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Prepare text units before analysis",
                    };
                }
                if (analysisRuns > 0) {
                    return {
                        ...stage,
                        status: "complete",
                        detail: `${analysisRuns} analysis run${analysisRuns === 1 ? "" : "s"}`,
                        blockedReason: null,
                    };
                }
                return {
                    ...stage,
                    status: "incomplete",
                    detail: "No analysis runs yet",
                    blockedReason: null,
                };
            }
            case "topics": {
                if (!anyUnits && unitCount === 0) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Prepare text units before topic modeling",
                    };
                }
                if (topicRuns > 0) {
                    return {
                        ...stage,
                        status: "complete",
                        detail: `${topicRuns} topic run${topicRuns === 1 ? "" : "s"}`,
                        blockedReason: null,
                    };
                }
                return {
                    ...stage,
                    status: "incomplete",
                    detail: "No topic models yet",
                    blockedReason: null,
                };
            }
            case "classify": {
                if (!hasLabels) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Need a codebook with labels",
                    };
                }
                if (annotationCompleted === 0) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Need labeled units before training",
                    };
                }
                if (modelCount > 0) {
                    return {
                        ...stage,
                        status: "complete",
                        detail: `${modelCount} trained model${modelCount === 1 ? "" : "s"}`,
                        blockedReason: null,
                    };
                }
                return {
                    ...stage,
                    status: "incomplete",
                    detail: "No trained models",
                    blockedReason: null,
                };
            }
            case "validate": {
                if (modelCount === 0 && analysisRuns === 0) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Train a model or run analysis first",
                    };
                }
                if (robustnessRuns > 0) {
                    return {
                        ...stage,
                        status: "complete",
                        detail: `${robustnessRuns} validation run${robustnessRuns === 1 ? "" : "s"}`,
                        blockedReason: null,
                    };
                }
                return {
                    ...stage,
                    status: "incomplete",
                    detail: "No robustness runs",
                    blockedReason: null,
                };
            }
            case "explore": {
                if (!anyUnits && unitCount === 0) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Prepare text units before exploring",
                    };
                }
                if (analysisRuns + topicRuns + modelCount > 0) {
                    return {
                        ...stage,
                        status: "complete",
                        detail: "Results ready to inspect",
                        blockedReason: null,
                    };
                }
                return {
                    ...stage,
                    status: "incomplete",
                    detail: "Run analysis to explore",
                    blockedReason: null,
                };
            }
            case "contextual": {
                if (!hasLabels) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Create a codebook with labels before contextual analysis",
                    };
                }
                if (analysisRuns + topicRuns + modelCount === 0) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Run an analysis before joining contextual indicators",
                    };
                }
                return {
                    ...stage,
                    status: "incomplete",
                    detail: "Join indicators when ready",
                    blockedReason: null,
                };
            }
            case "export": {
                if (!hasCorpus) {
                    return {
                        ...stage,
                        status: "blocked",
                        detail: null,
                        blockedReason: "Select a corpus to export",
                    };
                }
                if (exportRuns > 0) {
                    return {
                        ...stage,
                        status: "complete",
                        detail: `${exportRuns} export${exportRuns === 1 ? "" : "s"}`,
                        blockedReason: null,
                    };
                }
                return {
                    ...stage,
                    status: "incomplete",
                    detail: documentCount > 0 ? `${documentCount} docs ready` : "Ready when needed",
                    blockedReason: null,
                };
            }
            default:
                return {
                    ...stage,
                    status: "incomplete",
                    detail: null,
                    blockedReason: null,
                };
        }
    });

    return applyCurrent(stages, activeRoute);
}

export function stageIdFromPath(pathname: string, projectId: string): WorkflowStageId | "dashboard" | null {
    const prefix = `/research/${projectId}/`;
    if (!pathname.startsWith(prefix)) return null;
    const slug = pathname.slice(prefix.length).split("/")[0];
    if (slug === "dashboard") return "dashboard";
    const stage = RESEARCH_WORKFLOW_STAGES.find((item) => item.route === slug);
    return stage?.id ?? null;
}
