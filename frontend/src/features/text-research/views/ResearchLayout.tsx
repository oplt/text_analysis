import { useEffect, useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Drawer,
    useMediaQuery,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { ChatBubbleOutline as AskIcon } from "@mui/icons-material";
import { Outlet, useLocation, useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ContextInspector } from "../../../components/ui/ContextInspector";
import { PageShell } from "../../../components/ui/PageShell";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { WorkspaceSplit } from "../../../components/ui/WorkspaceSplit";
import { getProject } from "../../../api/projects";
import { queryKeys } from "../../../config/queryKeys";
import { CorpusAssistantPanel } from "../components/assistant/CorpusAssistantPanel";
import { ResearchMemosPanel } from "../components/ResearchMemosPanel";
import { CorpusDocumentDrawer } from "../components/CorpusDocumentDrawer";
import { ResearchContextBar } from "../components/ResearchContextBar";
import { ResearchNavBar } from "../components/ResearchNavBar";
import { ResearchWorkflowDrawer } from "../components/ResearchWorkflowDrawer";
import { ResearchProvider, useResearchContext } from "../hooks/useResearchContext";
import { useResearchWorkflow } from "../hooks/useResearchWorkflow";
import { setLastResearchProjectId } from "../researchProjectStorage";
import { RESEARCH_TABS } from "../types";
import {
    pathForNavItem,
    resolveResearchNav,
    type ResearchNavGroupId,
    type ResearchNavItem,
} from "../researchNavigation";
import {
    RESEARCH_WORKFLOW_STAGES,
    stageIdFromPath,
    type WorkflowStageState,
} from "../workflow";
import { stageActionLabel } from "../workflowDisplayModel";
import { getDocument, type AssistantCitation } from "../../../api/textResearch";
import type { CorpusDocument } from "../types";

function routeFromPath(pathname: string, projectId: string): string {
    const prefix = `/research/${projectId}/`;
    if (!pathname.startsWith(prefix)) return "dashboard";
    const slug = pathname.slice(prefix.length).split("/")[0];
    if (slug === "prepare") return "prepare";
    if (RESEARCH_TABS.some((tab) => tab.slug === slug)) return slug;
    if (RESEARCH_WORKFLOW_STAGES.some((stage) => stage.route === slug)) return slug;
    return "dashboard";
}

function ResearchLayoutInner() {
    const { projectId = "" } = useParams();
    const navigate = useNavigate();
    const location = useLocation();
    const theme = useTheme();
    const showInlineContext = useMediaQuery(theme.breakpoints.up("lg"));
    const ctx = useResearchContext();
    const activeRoute = routeFromPath(location.pathname, projectId);
    const activeStageId = stageIdFromPath(location.pathname, projectId);
    const { stages } = useResearchWorkflow(activeRoute);
    const resolvedNav = useMemo(
        () => resolveResearchNav(location.pathname, projectId, location.search),
        [location.pathname, location.search, projectId]
    );
    const [browseGroupId, setBrowseGroupId] = useState<ResearchNavGroupId | null>(null);
    const [workflowOpen, setWorkflowOpen] = useState(false);
    const [citationDoc, setCitationDoc] = useState<CorpusDocument | null>(null);
    const [citationSnippet, setCitationSnippet] = useState<string | null>(null);
    const [citationPage, setCitationPage] = useState<number | null>(null);
    const [citationCharStart, setCitationCharStart] = useState<number | null>(null);
    const [citationCharEnd, setCitationCharEnd] = useState<number | null>(null);
    const [citationSourceSpanIds, setCitationSourceSpanIds] = useState<string[] | null>(null);
    const [citationOffsetScope, setCitationOffsetScope] = useState<
        "parsed_document" | "page" | "canonical_document" | null
    >(null);
    const [citationOpenError, setCitationOpenError] = useState<string | null>(null);
    const [citationCoordinate, setCitationCoordinate] = useState<AssistantCitation | null>(null);

    const projectQuery = useQuery({
        queryKey: queryKeys.projects.detail(projectId),
        queryFn: () => getProject(projectId),
        enabled: Boolean(projectId),
    });

    useEffect(() => {
        if (projectId) {
            setLastResearchProjectId(projectId);
        }
    }, [projectId]);

    useEffect(() => {
        setBrowseGroupId(null);
    }, [location.pathname, location.search]);

    const nextStage =
        stages.find((stage) => stage.status === "current" || stage.status === "warning") ??
        stages.find((stage) => stage.status === "incomplete");

    function handleSelectStage(stage: WorkflowStageState) {
        if (stage.status === "blocked") {
            openWorkflow();
            return;
        }
        navigate(`/research/${projectId}/${stage.route}`);
    }

    function handleNavigateItem(item: ResearchNavItem) {
        navigate(pathForNavItem(projectId, item));
        setBrowseGroupId(null);
    }

    async function handleOpenCitation(citation: AssistantCitation) {
        if (citation.source_status === "historical_source_unavailable") {
            setCitationOpenError(
                "historical_source_unavailable: the cited evidence revision is no longer resolvable."
            );
            return;
        }
        if (!citation.corpus_document_id) {
            setCitationOpenError("This citation no longer has an accessible corpus document.");
            return;
        }
        try {
            const document = await getDocument(citation.corpus_document_id);
            setCitationDoc(document);
            setCitationSnippet(citation.snippet);
            setCitationPage(citation.page_number ?? null);
            setCitationCharStart(citation.char_start ?? null);
            setCitationCharEnd(citation.char_end ?? null);
            setCitationSourceSpanIds(citation.source_span_ids ?? null);
            setCitationOffsetScope(citation.offset_scope ?? null);
            setCitationCoordinate(citation);
        } catch {
            setCitationOpenError(
                "historical_source_unavailable: could not open this citation source."
            );
        }
    }

    function openWorkflow() {
        if (!showInlineContext) ctx.setAskPanelOpen(false);
        setWorkflowOpen(true);
    }

    function openAskCorpus() {
        if (!showInlineContext) setWorkflowOpen(false);
        ctx.setAskPanelOpen(true);
    }

    const contextDrawerOpen = ctx.askPanelOpen && !showInlineContext;
    const projectName = projectQuery.data?.name ?? projectId;

    const askCorpusBody = <CorpusAssistantPanel onOpenCitation={handleOpenCitation} />;

    const workspaceActions = (
        <>
            {nextStage ? (
                <Button
                    size="small"
                    variant="contained"
                    onClick={() => handleSelectStage(nextStage)}
                    disabled={nextStage.status === "blocked"}
                >
                    Next: {stageActionLabel(nextStage)}
                </Button>
            ) : null}
            {!showInlineContext ? (
                <Button
                    variant="outlined"
                    size="small"
                    startIcon={<AskIcon />}
                    onClick={openAskCorpus}
                >
                    Ask Corpus
                </Button>
            ) : null}
        </>
    );

    return (
        <PageShell width="full" dense>
            <ResearchContextBar projectName={projectName} actions={workspaceActions} />

            <ResearchNavBar
                projectId={projectId}
                resolved={resolvedNav}
                browseGroupId={browseGroupId}
                onBrowseGroup={setBrowseGroupId}
                onNavigate={handleNavigateItem}
                onOpenAll={openWorkflow}
            />

            <WorkspaceSplit
                ratio="8-4"
                sideFrom="lg"
                side={
                    showInlineContext ? (
                        <ContextInspector
                            title="Ask Corpus"
                            description="Context, corpus questions, and retrieved evidence."
                            sticky
                            sx={{ width: "100%", maxWidth: "none", minWidth: 0 }}
                        >
                            {askCorpusBody}
                        </ContextInspector>
                    ) : undefined
                }
                main={
                    <>
                        {ctx.corporaError ? (
                            <Alert severity="error" sx={{ mb: 2 }}>
                                Failed to load corpora for this project.
                            </Alert>
                        ) : null}
                        <QueryBoundary
                            isLoading={ctx.corporaLoading && ctx.corpora.length === 0}
                            variant="inline"
                        >
                            <ResearchMemosPanel onOpenCitation={handleOpenCitation} />
                            <Outlet />
                        </QueryBoundary>
                    </>
                }
            />

            {citationOpenError ? (
                <Alert severity="error" onClose={() => setCitationOpenError(null)}>
                    {citationOpenError}
                </Alert>
            ) : null}

            <Drawer
                anchor="right"
                open={contextDrawerOpen}
                onClose={() => ctx.setAskPanelOpen(false)}
                PaperProps={{
                    sx: {
                        width: { xs: "100%", sm: "min(100vw, 400px)", md: 400 },
                        maxWidth: "100%",
                        p: 2,
                    },
                }}
            >
                <ContextInspector
                    title="Ask Corpus"
                    description="Context, corpus questions, and retrieved evidence."
                    sticky={false}
                    dismissible
                    onDismiss={() => ctx.setAskPanelOpen(false)}
                    sx={{ width: "100%", maxWidth: "none", minWidth: 0 }}
                >
                    {askCorpusBody}
                </ContextInspector>
            </Drawer>

            <CorpusDocumentDrawer
                open={Boolean(citationDoc)}
                document={citationDoc}
                corpusId={ctx.selectedCorpus?.id ?? ""}
                highlightSnippet={citationSnippet}
                pageHint={citationPage}
                charStart={citationCharStart}
                charEnd={citationCharEnd}
                sourceSpanIds={citationSourceSpanIds}
                offsetScope={citationOffsetScope}
                sourceCoordinate={citationCoordinate}
                onClose={() => {
                    setCitationDoc(null);
                    setCitationSnippet(null);
                    setCitationPage(null);
                    setCitationCharStart(null);
                    setCitationCharEnd(null);
                    setCitationSourceSpanIds(null);
                    setCitationOffsetScope(null);
                    setCitationCoordinate(null);
                }}
            />

            <ResearchWorkflowDrawer
                open={workflowOpen}
                onClose={() => setWorkflowOpen(false)}
                stages={stages}
                activeStageId={
                    activeStageId && activeStageId !== "dashboard" ? activeStageId : false
                }
                onSelectStage={handleSelectStage}
                onNavigateItem={handleNavigateItem}
                resolved={resolvedNav}
                onOpenDashboard={() => navigate(`/research/${projectId}/dashboard`)}
                dashboardSelected={activeRoute === "dashboard"}
            />
        </PageShell>
    );
}

export default function ResearchLayout() {
    return (
        <ResearchProvider>
            <ResearchLayoutInner />
        </ResearchProvider>
    );
}
