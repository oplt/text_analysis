import { Alert, Box, Button, Stack, Tab, Tabs, Typography } from "@mui/material";
import { ArrowBack as BackIcon, Science as LabIcon } from "@mui/icons-material";
import { Outlet, useLocation, useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getProject } from "../../../api/projects";
import { PageShell } from "../../../components/ui/PageShell";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { ResearchContextBar } from "../components/ResearchShared";
import { ResearchProvider, useResearchContext } from "../hooks/useResearchContext";
import { RESEARCH_TABS, type ResearchTabSlug } from "../types";

function tabFromPath(pathname: string, projectId: string): ResearchTabSlug {
    const prefix = `/research/${projectId}/`;
    if (!pathname.startsWith(prefix)) return "dashboard";
    const slug = pathname.slice(prefix.length).split("/")[0] as ResearchTabSlug;
    return RESEARCH_TABS.some((t) => t.slug === slug) ? slug : "dashboard";
}

function ResearchLayoutInner() {
    const { projectId = "" } = useParams();
    const navigate = useNavigate();
    const location = useLocation();
    const ctx = useResearchContext();
    const activeTab = tabFromPath(location.pathname, projectId);

    const projectQuery = useQuery({
        queryKey: queryKeys.projects.detail(projectId),
        queryFn: () => getProject(projectId),
        enabled: Boolean(projectId),
    });

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

            <SectionCard title="Workspace context" description="Select the active corpus and codebook for downstream tabs.">
                <ResearchContextBar />
            </SectionCard>

            {ctx.corporaError ? (
                <Alert severity="error">Failed to load corpora for this project.</Alert>
            ) : null}

            <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
                <Tabs
                    value={activeTab}
                    onChange={(_, value: ResearchTabSlug) =>
                        navigate(`/research/${projectId}/${value}`)
                    }
                    variant="scrollable"
                    scrollButtons="auto"
                >
                    {RESEARCH_TABS.map((tab) => (
                        <Tab key={tab.slug} label={tab.label} value={tab.slug} />
                    ))}
                </Tabs>
            </Box>

            <QueryBoundary
                isLoading={ctx.corporaLoading && ctx.corpora.length === 0}
                variant="inline"
            >
                <Outlet />
            </QueryBoundary>
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
