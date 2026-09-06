import { useEffect } from "react";
import { Alert, Box, Button, Stack, Typography } from "@mui/material";
import { ArrowBack as BackIcon, Science as LabIcon } from "@mui/icons-material";
import { Outlet, useLocation, useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getProject } from "../../../api/projects";
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
    const ctx = useResearchContext();
    const activeRoute = routeFromPath(location.pathname, projectId);
    const activeStageId = stageIdFromPath(location.pathname, projectId);
    const { stages } = useResearchWorkflow(activeRoute);

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

    return (
        <PageShell maxWidth="xl">
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
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
            </Stack>

            <Stack direction="row" spacing={1.5} alignItems="center">
                <LabIcon color="primary" />
                <Box>
                    <Typography variant="h4">Policy Text Lab</Typography>
                    <Typography variant="body2" color="text.secondary">
                        {projectQuery.data?.name ?? "Research workspace"} — corpus management,
                        annotation, analysis, and exports.
                    </Typography>
                </Box>
            </Stack>

            <Box
                sx={{
                    display: "grid",
                    gridTemplateColumns: {
                        xs: "minmax(0, 1fr)",
                        lg: "minmax(210px, 0.75fr) minmax(0, 2.2fr) minmax(230px, 0.85fr)",
                    },
                    gap: 2,
                    alignItems: "start",
                    mt: 2,
                }}
            >
                <Box component="nav" aria-label="Research workflow" sx={{ position: { lg: "sticky" }, top: { lg: 16 } }}>
                    <SectionCard sx={{ mt: 0 }}>
                        <ResearchWorkflowNavigator
                            stages={stages}
                            activeStageId={
                                activeStageId && activeStageId !== "dashboard" ? activeStageId : false
                            }
                            onSelectStage={handleSelectStage}
                            onOpenDashboard={() => navigate(`/research/${projectId}/dashboard`)}
                            dashboardSelected={activeRoute === "dashboard"}
                            orientation="vertical"
                        />
                    </SectionCard>
                </Box>

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

                <Box component="aside" aria-label="Research context" sx={{ position: { lg: "sticky" }, top: { lg: 16 } }}>
                    <SectionCard
                        title="Workspace context"
                        description="Corpus, codebook, and unit selection remain visible while you work."
                        sx={{ mt: 0 }}
                    >
                        <ResearchContextBar />
                    </SectionCard>
                </Box>
            </Box>

            <Box
                component="footer"
                sx={{ mt: 2, py: 1, borderTop: 1, borderColor: "divider" }}
            >
                <Typography variant="caption" color="text.secondary">
                    Corpus: {ctx.selectedCorpus?.name ?? "not selected"} · Unit: {ctx.unitType} · Codebook: {ctx.selectedCodebook?.name ?? "not selected"} · Research state is persisted in analysis runs.
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
