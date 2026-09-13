/**
 * Phase 2 research information architecture.
 * Groups equal-weight tabs into researcher-facing phases while preserving route URLs.
 */

export type ResearchNavGroupId =
    | "overview"
    | "data"
    | "coding"
    | "analysis"
    | "models"
    | "outputs";

export type ResearchNavItem = {
    id: string;
    label: string;
    /** Path segment under `/research/:projectId/` */
    route: string;
    /** Optional `?tab=` for page-level tabs (Prepare, Analysis). */
    tab?: string;
    description?: string;
};

export type ResearchNavGroup = {
    id: ResearchNavGroupId;
    label: string;
    description: string;
    items: ResearchNavItem[];
};

/** Canonical grouped research navigation (Phase 2). */
export const RESEARCH_NAV_GROUPS: ResearchNavGroup[] = [
    {
        id: "overview",
        label: "Overview",
        description: "Project status and recommended next step",
        items: [
            {
                id: "dashboard",
                label: "Dashboard",
                route: "dashboard",
                description: "Research overview for the active corpus",
            },
        ],
    },
    {
        id: "data",
        label: "Data",
        description: "Collect, segment, and prepare the corpus",
        items: [
            { id: "corpus", label: "Corpus", route: "corpus", description: "Documents and metadata" },
            {
                id: "prepare",
                label: "Prepare",
                route: "prepare",
                tab: "segment",
                description: "Segment documents into text units",
            },
            {
                id: "ingestion",
                label: "Ingestion QA",
                route: "prepare",
                tab: "ingestion",
                description: "Check ingestion quality",
            },
            {
                id: "cleaning",
                label: "Cleaning",
                route: "prepare",
                tab: "cleaning",
                description: "Cleaning profiles and rules",
            },
            {
                id: "preprocessing",
                label: "Preprocessing",
                route: "prepare",
                tab: "preprocessing",
                description: "Preprocessing profiles",
            },
        ],
    },
    {
        id: "coding",
        label: "Coding & Quality",
        description: "Codebook, annotation, and agreement",
        items: [
            { id: "codebook", label: "Codebook", route: "codebook" },
            { id: "annotation", label: "Annotation", route: "annotation" },
            { id: "reliability", label: "Reliability", route: "reliability" },
        ],
    },
    {
        id: "analysis",
        label: "Analysis",
        description: "Quantitative and exploratory tools",
        items: [
            { id: "analysis", label: "Analysis", route: "analysis" },
            { id: "dictionaries", label: "Dictionaries", route: "dictionaries" },
            { id: "comparative", label: "Comparative", route: "comparative" },
            { id: "topics", label: "Topics", route: "topics" },
            { id: "contextual", label: "Contextual", route: "contextual" },
            { id: "robustness", label: "Robustness", route: "robustness" },
            { id: "explorer", label: "Explorer", route: "explorer" },
        ],
    },
    {
        id: "models",
        label: "Models",
        description: "Supervised models and monitoring",
        items: [
            { id: "classification", label: "Classification", route: "classification" },
            { id: "active-learning", label: "Active Learning", route: "active-learning" },
            { id: "models", label: "Model Registry", route: "models" },
            { id: "predictions", label: "Predictions", route: "predictions" },
            { id: "drift", label: "Drift", route: "drift" },
        ],
    },
    {
        id: "outputs",
        label: "Outputs",
        description: "Runs and exports",
        items: [
            { id: "runs", label: "Runs", route: "runs" },
            { id: "exports", label: "Exports", route: "exports" },
        ],
    },
];

export type ResolvedResearchNav = {
    group: ResearchNavGroup;
    item: ResearchNavItem;
};

function tabFromSearch(search: string): string | null {
    const params = new URLSearchParams(search.startsWith("?") ? search.slice(1) : search);
    const tab = params.get("tab");
    return tab && tab.length > 0 ? tab : null;
}

/** Build in-app path for a nav item (preserves existing URLs). */
export function pathForNavItem(projectId: string, item: ResearchNavItem): string {
    const base = `/research/${projectId}/${item.route}`;
    if (!item.tab) return base;
    // Prepare default tab is segment — omit query when default for cleaner URLs.
    if (item.route === "prepare" && item.tab === "segment") return base;
    return `${base}?tab=${encodeURIComponent(item.tab)}`;
}

/**
 * Resolve the active group/item from the current location.
 * Prefer the most specific match (route + tab) over route-only items.
 */
export function resolveResearchNav(
    pathname: string,
    projectId: string,
    search = ""
): ResolvedResearchNav | null {
    const prefix = `/research/${projectId}/`;
    if (!pathname.startsWith(prefix)) return null;
    const slug = pathname.slice(prefix.length).split("/")[0] || "dashboard";
    const tab = tabFromSearch(search);

    let fallback: ResolvedResearchNav | null = null;

    for (const group of RESEARCH_NAV_GROUPS) {
        for (const item of group.items) {
            if (item.route !== slug) continue;
            if (item.tab) {
                const effectiveTab =
                    slug === "prepare" && (!tab || tab === "segment")
                        ? "segment"
                        : tab;
                if (effectiveTab && item.tab === effectiveTab) {
                    return { group, item };
                }
                // Prefer default prepare item while scanning non-matching tabs.
                if (!fallback && item.tab === "segment") fallback = { group, item };
                continue;
            }
            if (!fallback) fallback = { group, item };
        }
    }

    return fallback;
}

export function researchNavGroupById(id: ResearchNavGroupId): ResearchNavGroup | undefined {
    return RESEARCH_NAV_GROUPS.find((group) => group.id === id);
}

/** Flat list of all destination routes covered by grouped nav (for parity checks). */
export function allResearchNavRoutes(): string[] {
    return [...new Set(RESEARCH_NAV_GROUPS.flatMap((g) => g.items.map((i) => i.route)))];
}
