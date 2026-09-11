import { useEffect, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Drawer,
    IconButton,
    Stack,
    Typography,
    useMediaQuery,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import {
    Close as CloseIcon,
    ChatBubbleOutline as AskIcon,
} from "@mui/icons-material";
import { Outlet, useLocation, useNavigate, useParams } from "react-router-dom";
import { PageShell } from "../../../components/ui/PageShell";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { CorpusAssistantPanel } from "../components/assistant/CorpusAssistantPanel";
import { ResearchMemosPanel } from "../components/ResearchMemosPanel";
import { CorpusDocumentDrawer } from "../components/CorpusDocumentDrawer";
import { ResearchWorkflowDrawer } from "../components/ResearchWorkflowDrawer";
import { ResearchWorkflowStrip } from "../components/ResearchWorkflowStrip";
import { ResearchProvider, useResearchContext } from "../hooks/useResearchContext";
import { useResearchWorkflow } from "../hooks/useResearchWorkflow";
import { setLastResearchProjectId } from "../researchProjectStorage";
import { RESEARCH_TABS } from "../types";
import {
    RESEARCH_WORKFLOW_STAGES,
    stageIdFromPath,
    type WorkflowStageState,
} from "../workflow";
import { getDocument, type AssistantCitation } from "../../../api/textResearch";
import type { CorpusDocument } from "../types";

const ANALYSIS_SUBLINKS = [
    { tab: "overview", label: "Corpus tools" },
    { tab: "statistical", label: "Statistical models" },
    { tab: "measurement", label: "Measurement" },
] as const;

const ANALYSIS_RELATED_LINKS = [
    { path: "dictionaries", label: "Dictionaries" },
    { path: "comparative", label: "Comparative prevalence" },
] as const;

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
    const isWide = useMediaQuery(theme.breakpoints.up("xl"));
    const showInlineContext = isWide;
    const ctx = useResearchContext();
    const activeRoute = routeFromPath(location.pathname, projectId);
    const activeStageId = stageIdFromPath(location.pathname, projectId);
    const { stages } = useResearchWorkflow(activeRoute);
    const [contextOpen, setContextOpen] = useState(false);
    const [workflowOpen, setWorkflowOpen] = useState(false);
    const [citationDoc, setCitationDoc] = useState<CorpusDocument | null>(null);
    const [citationSnippet, setCitationSnippet] = useState<string | null>(null);
    const [citationPage, setCitationPage] = useState<number | null>(null);
    const [citationCharStart, setCitationCharStart] = useState<number | null>(null);
    const [citationCharEnd, setCitationCharEnd] = useState<number | null>(null);
    const [citationSourceSpanIds, setCitationSourceSpanIds] = useState<string[] | null>(null);

    useEffect(() => {
        if (ctx.askPanelOpenNonce > 0) {
            setContextOpen(true);
        }
    }, [ctx.askPanelOpenNonce]);

    useEffect(() => {
        if (projectId) {
            setLastResearchProjectId(projectId);
        }
    }, [projectId]);

    function handleSelectStage(stage: WorkflowStageState) {
        if (stage.status === "blocked") {
            setWorkflowOpen(true);
            return;
        }
        navigate(`/research/${projectId}/${stage.route}`);
    }

    async function handleOpenCitation(citation: AssistantCitation) {
        if (!citation.corpus_document_id) return;
        try {
            const document = await getDocument(citation.corpus_document_id);
            setCitationDoc(document);
            setCitationSnippet(citation.snippet);
            setCitationPage(citation.page_number ?? null);
            setCitationCharStart(citation.char_start ?? null);
            setCitationCharEnd(citation.char_end ?? null);
            setCitationSourceSpanIds(citation.source_span_ids ?? null);
        } catch {
            // Preserve the existing no-op behavior when a citation is no longer accessible.
        }
    }

    const contextDrawerOpen = contextOpen && !showInlineContext;

    const contextPanel = (
        <SectionCard
            title="Ask Corpus"
            description="Context, corpus questions, and retrieved evidence."
            variant="subtle"
            compact
            sx={{ mt: 0 }}
            action={
                !showInlineContext ? (
                    <IconButton
                        aria-label="Close Ask Corpus panel"
                        size="small"
                        onClick={() => setContextOpen(false)}
                    >
                        <CloseIcon fontSize="small" />
                    </IconButton>
                ) : undefined
            }
        >
            <CorpusAssistantPanel onOpenCitation={handleOpenCitation} />
        </SectionCard>
    );

    return (
        <PageShell width="full" dense>
            {!showInlineContext ? (
                <Stack direction="row" spacing={1} alignItems="center" justifyContent="flex-end">
                    <Button
                        variant="outlined"
                        size="small"
                        startIcon={<AskIcon />}
                        onClick={() => setContextOpen(true)}
                    >
                        Ask Corpus
                    </Button>
                </Stack>
            ) : null}

            <Box sx={{ mb: 1.5 }}>
                <ResearchWorkflowStrip
                    stages={stages}
                    activeStageId={
                        activeStageId && activeStageId !== "dashboard" ? activeStageId : false
                    }
                    onSelectStage={handleSelectStage}
                    onOpenWorkflow={() => setWorkflowOpen(true)}
                />
            </Box>

            {!workflowOpen ? (
                <Button
                    variant="contained"
                    size="small"
                    onClick={() => setWorkflowOpen(true)}
                    aria-label="Open all workflow stages"
                    sx={{
                        position: "fixed",
                        right: { xs: 12, md: 16 },
                        top: "50%",
                        transform: "translateY(-50%)",
                        zIndex: (t) => t.zIndex.speedDial,
                        writingMode: "vertical-rl",
                        textOrientation: "mixed",
                        minWidth: 40,
                        px: 1,
                        py: 1.5,
                        borderRadius: 2,
                        boxShadow: 3,
                        letterSpacing: 0.04,
                    }}
                >
                    
                    
                    Stages
                </Button>
            ) : null}

            <Box
                sx={{
                    display: "grid",
                    gridTemplateColumns: {
                        xs: "minmax(0, 1fr)",
                        xl: "minmax(0, 1fr) minmax(380px, 440px)",
                    },
                    gap: { xs: 1.5, lg: 2 },
                    alignItems: "start",
                }}
            >
                <Box component="main" sx={{ minWidth: 0 }}>
                    {ctx.corporaError ? (
                        <Alert severity="error" sx={{ mb: 2 }}>
                            Failed to load corpora for this project.
                        </Alert>
                    ) : null}
                    {activeRoute === "analysis" ? (
                        <Stack
                            direction="row"
                            spacing={1}
                            flexWrap="wrap"
                            useFlexGap
                            sx={{ mb: 1.5 }}
                            aria-label="Analysis section"
                        >
                            <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ alignSelf: "center", mr: 0.5 }}
                            >
                                Overview
                            </Typography>
                            {ANALYSIS_SUBLINKS.map((link) => (
                                <Button
                                    key={link.tab}
                                    size="small"
                                    variant={
                                        location.search.includes(`tab=${link.tab}`) ||
                                        (link.tab === "overview" &&
                                            !location.search.includes("tab="))
                                            ? "contained"
                                            : "outlined"
                                    }
                                    onClick={() =>
                                        navigate(
                                            link.tab === "overview"
                                                ? `/research/${projectId}/analysis`
                                                : `/research/${projectId}/analysis?tab=${link.tab}`
                                        )
                                    }
                                >
                                    {link.label}
                                </Button>
                            ))}
                            {ANALYSIS_RELATED_LINKS.map((link) => (
                                <Button
                                    key={link.path}
                                    size="small"
                                    variant="text"
                                    onClick={() =>
                                        navigate(`/research/${projectId}/${link.path}`)
                                    }
                                >
                                    {link.label}
                                </Button>
                            ))}
                        </Stack>
                    ) : null}
                    <QueryBoundary
                        isLoading={ctx.corporaLoading && ctx.corpora.length === 0}
                        variant="inline"
                    >
                        <ResearchMemosPanel onOpenCitation={handleOpenCitation} />
                        <Outlet />
                    </QueryBoundary>
                </Box>

                {showInlineContext ? (
                    <Box
                        component="aside"
                        aria-label="Ask Corpus"
                        sx={{
                            position: "sticky",
                            top: 16,
                            width: { xl: 400 },
                            maxWidth: 440,
                            minWidth: 360,
                        }}
                    >
                        {contextPanel}
                    </Box>
                ) : null}
            </Box>

            <Drawer
                anchor="right"
                open={contextDrawerOpen}
                onClose={() => setContextOpen(false)}
                PaperProps={{ sx: { width: { xs: "100%", sm: 400 }, p: 2 } }}
            >
                {contextPanel}
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
                onClose={() => {
                    setCitationDoc(null);
                    setCitationSnippet(null);
                    setCitationPage(null);
                    setCitationCharStart(null);
                    setCitationCharEnd(null);
                    setCitationSourceSpanIds(null);
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
                onOpenDashboard={() => navigate(`/research/${projectId}/dashboard`)}
                dashboardSelected={activeRoute === "dashboard"}
            />

            <Box
                component="footer"
                sx={{ mt: 1, py: 1, borderTop: 1, borderColor: "divider" }}
            >
                <Typography variant="caption" color="text.secondary">
                    Corpus: {ctx.selectedCorpus?.name ?? "not selected"} · Unit: {ctx.unitType} ·
                    Codebook: {ctx.selectedCodebook?.name ?? "not selected"} · Research state is
                    persisted in analysis runs.
                    {citationSnippet ? ` · Opened evidence: ${citationSnippet.slice(0, 80)}…` : ""}
                </Typography>
            </Box>
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
