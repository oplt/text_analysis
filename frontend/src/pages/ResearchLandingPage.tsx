import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
    Box,
    Button,
    CircularProgress,
    List,
    ListItemButton,
    ListItemText,
    Stack,
    Typography,
} from "@mui/material";
import {
    FolderOpen as ProjectIcon,
    Science as LabIcon,
} from "@mui/icons-material";
import { listProjects } from "../api/projects";
import { EmptyState } from "../components/ui/EmptyState";
import { PageShell } from "../components/ui/PageShell";
import { QueryBoundary } from "../components/ui/QueryBoundary";
import { SectionCard } from "../components/ui/SectionCard";
import { queryKeys } from "../config/queryKeys";
import {
    getLastResearchProjectId,
    researchLabPath,
    setLastResearchProjectId,
} from "../features/text-research/researchProjectStorage";

export default function ResearchLandingPage() {
    const navigate = useNavigate();
    const projectsQuery = useQuery({
        queryKey: queryKeys.projects.all,
        queryFn: listProjects,
    });

    const projects = useMemo(() => projectsQuery.data ?? [], [projectsQuery.data]);
    const lastProjectId = getLastResearchProjectId();
    const lastProject = useMemo(
        () => projects.find((project) => project.id === lastProjectId) ?? null,
        [projects, lastProjectId]
    );

    function openProject(projectId: string) {
        setLastResearchProjectId(projectId);
        navigate(researchLabPath(projectId));
    }

    return (
        <PageShell maxWidth="md">
            <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 2 }}>
                <LabIcon color="primary" />
                <Box>
                    <Typography variant="h4">Text Research</Typography>
                    <Typography variant="body2" color="text.secondary">
                        Choose a project to open the research workspace.
                    </Typography>
                </Box>
            </Stack>

            <QueryBoundary
                isLoading={projectsQuery.isLoading}
                isError={projectsQuery.isError}
                error={projectsQuery.error}
                onRetry={() => void projectsQuery.refetch()}
            >
                {projects.length === 0 ? (
                    <EmptyState
                        icon={<ProjectIcon fontSize="large" />}
                        title="No projects yet"
                        description="Create a project first, then open Text Research to build corpora, annotate, and analyze."
                        action={
                            <Button variant="contained" onClick={() => navigate("/projects")}>
                                Go to projects
                            </Button>
                        }
                    />
                ) : (
                    <Stack spacing={2}>
                        {lastProject ? (
                            <SectionCard
                                title="Continue"
                                description="Resume the last research project you opened."
                                action={
                                    <Button
                                        variant="contained"
                                        color="secondary"
                                        onClick={() => openProject(lastProject.id)}
                                    >
                                        Open {lastProject.name}
                                    </Button>
                                }
                            >
                                <Typography variant="body2" color="text.secondary">
                                    {lastProject.description || "No description"}
                                </Typography>
                            </SectionCard>
                        ) : null}

                        <SectionCard
                            title="Select a project"
                            description="Research corpora, codebooks, and models are scoped to a project."
                        >
                            {projectsQuery.isFetching && !projectsQuery.isLoading ? (
                                <CircularProgress size={20} sx={{ mb: 1 }} />
                            ) : null}
                            <List disablePadding>
                                {projects.map((project) => (
                                    <ListItemButton
                                        key={project.id}
                                        onClick={() => openProject(project.id)}
                                        sx={{ borderRadius: 2, mb: 0.5 }}
                                    >
                                        <ListItemText
                                            primary={project.name}
                                            secondary={project.description || "Open Text Research"}
                                        />
                                    </ListItemButton>
                                ))}
                            </List>
                        </SectionCard>
                    </Stack>
                )}
            </QueryBoundary>
        </PageShell>
    );
}
