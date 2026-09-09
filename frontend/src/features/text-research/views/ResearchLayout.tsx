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
    ArrowBack as BackIcon,
    Close as CloseIcon,
    Science as LabIcon,
    Tune as ContextIcon,
} from "@mui/icons-material";
import { Outlet, useLocation, useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getProject } from "../../../api/projects";
import { PageHeader } from "../../../components/ui/PageHeader";
import { PageShell } from "../../../components/ui/PageShell";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { ResearchContextBar } from "../components/ResearchShared";
import { ResearchWorkflowNavigator } from "../components/ResearchWorkflowNavigator";
import { ResearchProvider, useResearchContext } from "../hooks/useResearchContext";
import { useResearchWorkflow } from "../hooks/useResearchWorkflow";
import { setLastResearchProjectId } from "../researchProjectStorage";
import { RESEARCH_TABS } from "../types";
import {
    RESEARCH_WORKFLOW_STAGES,
    stageIdFromPath,
    type WorkflowStageState,
} from "../workflow";

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
    const isDesktop = useMediaQuery(theme.breakpoints.up("lg"));
    const showInlineContext = isWide;
    const ctx = useResearchContext();
    const activeRoute = routeFromPath(location.pathname, projectId);
    const activeStageId = stageIdFromPath(location.pathname, projectId);
    const { stages } = useResearchWorkflow(activeRoute);
    const [contextOpen, setContextOpen] = useState(false);

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

    function handleSelectStage(stage: WorkflowStageState) {
        navigate(`/research/${projectId}/${stage.route}`);
    }

    const drawerOpen = contextOpen && !showInlineContext;

    const contextPanel = (
        <SectionCard
            title="Workspace context"
            description="Corpus, codebook, and unit selection remain visible while you work."
            variant="subtle"
            compact
            sx={{ mt: 0 }}
            action={
                !showInlineContext ? (
                    <IconButton aria-label="Close context" size="small" onClick={() => setContextOpen(false)}>
                        <CloseIcon fontSize="small" />
                    </IconButton>
                ) : undefined
            }
        >
            <ResearchContextBar />
        </SectionCard>
    );

    return (
        <PageShell width="full" dense>
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                <Button
                    variant="outlined"
                    size="small"
                    startIcon={<BackIcon />}
                    onClick={() => navigate(`/projects/${projectId}`)}
                >
                    Back to project
                </Button>
                <Button variant="text" size="small" onClick={() => navigate("/research")}>
                    Switch project
                </Button>
                {!showInlineContext ? (
                    <Button
                        variant="outlined"
                        size="small"
                        startIcon={<ContextIcon />}
                        onClick={() => setContextOpen(true)}
                        sx={{ ml: { sm: "auto" } }}
                    >
                        Context
                    </Button>
                ) : null}
            </Stack>

            <PageHeader
                dense
                icon={<LabIcon />}
                title="Text Research"
                description={`${projectQuery.data?.name ?? "Research workspace"} — corpus, annotation, analysis, and exports.`}
            />

            <Box
                sx={{
                    display: "grid",
                    gridTemplateColumns: {
                        xs: "minmax(0, 1fr)",
                        lg: showInlineContext
                            ? "minmax(160px, 180px) minmax(0, 1fr) minmax(220px, 260px)"
                            : "minmax(160px, 180px) minmax(0, 1fr)",
                        xl: "minmax(170px, 200px) minmax(0, 1fr) minmax(240px, 280px)",
                    },
                    gap: { xs: 1.5, lg: 2 },
                    alignItems: "start",
                }}
            >
                {isDesktop ? (
                    <Box
                        component="nav"
                        aria-label="Research workflow"
                        sx={{ position: "sticky", top: 16 }}
                    >
                        <SectionCard variant="subtle" compact sx={{ mt: 0 }}>
                            <ResearchWorkflowNavigator
                                stages={stages}
                                activeStageId={
                                    activeStageId && activeStageId !== "dashboard"
                                        ? activeStageId
                                        : false
                                }
                                onSelectStage={handleSelectStage}
                                onOpenDashboard={() => navigate(`/research/${projectId}/dashboard`)}
                                dashboardSelected={activeRoute === "dashboard"}
                                orientation="vertical"
                            />
                        </SectionCard>
                    </Box>
                ) : (
                    <SectionCard variant="flat" compact sx={{ mt: 0 }}>
                        <ResearchWorkflowNavigator
                            stages={stages}
                            activeStageId={
                                activeStageId && activeStageId !== "dashboard"
                                    ? activeStageId
                                    : false
                            }
                            onSelectStage={handleSelectStage}
                            onOpenDashboard={() => navigate(`/research/${projectId}/dashboard`)}
                            dashboardSelected={activeRoute === "dashboard"}
                            orientation="horizontal"
                        />
                    </SectionCard>
                )}

                <Box component="main" sx={{ minWidth: 0 }}>
                    {ctx.corporaError ? (
                        <Alert severity="error" sx={{ mb: 2 }}>
                            Failed to load corpora for this project.
                        </Alert>
                    ) : null}
                    <QueryBoundary
                        isLoading={ctx.corporaLoading && ctx.corpora.length === 0}
                        variant="inline"
                    >
                        <Outlet />
                    </QueryBoundary>
                </Box>

                {showInlineContext ? (
                    <Box
                        component="aside"
                        aria-label="Research context"
                        sx={{ position: "sticky", top: 16 }}
                    >
                        {contextPanel}
                    </Box>
                ) : null}
            </Box>

            <Drawer
                anchor="right"
                open={drawerOpen}
                onClose={() => setContextOpen(false)}
                PaperProps={{ sx: { width: { xs: "100%", sm: 360 }, p: 2 } }}
            >
                {contextPanel}
            </Drawer>

            <Box
                component="footer"
                sx={{ mt: 1, py: 1, borderTop: 1, borderColor: "divider" }}
            >
                <Typography variant="caption" color="text.secondary">
                    Corpus: {ctx.selectedCorpus?.name ?? "not selected"} · Unit: {ctx.unitType} ·
                    Codebook: {ctx.selectedCodebook?.name ?? "not selected"} · Research state is
                    persisted in analysis runs.
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
