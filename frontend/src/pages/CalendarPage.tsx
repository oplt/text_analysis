import { useQuery } from "@tanstack/react-query";
import { Button, Stack } from "@mui/material";
import { ArrowForward as ArrowForwardIcon } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { listProjects } from "../api/projects";
import { queryKeys } from "../config/queryKeys";
import { DashboardCalendar } from "../components/dashboard/DashboardCalendar";
import { PageShell } from "../components/ui/PageShell";
import { QueryBoundary } from "../components/ui/QueryBoundary";
import { usePlatformMetadata } from "../hooks/usePlatformMetadata";

export default function CalendarPage() {
    const navigate = useNavigate();
    const { data: platformMetadata } = usePlatformMetadata();
    const {
        data: projects,
        isLoading: projectsLoading,
        isError: projectsIsError,
        error: projectsError,
        refetch: refetchProjects,
    } = useQuery({
        queryKey: queryKeys.projects.all,
        queryFn: listProjects,
    });

    const coreDomainPlural = platformMetadata?.core_domain_plural ?? "Projects";

    return (
        <PageShell maxWidth="xl">
            <Stack direction="row" justifyContent="flex-end" sx={{ mb: 2 }}>
                <Button
                    variant="outlined"
                    endIcon={<ArrowForwardIcon />}
                    onClick={() => navigate("/projects")}
                >
                    Open {coreDomainPlural}
                </Button>
            </Stack>

            <QueryBoundary
                isLoading={projectsLoading}
                isError={projectsIsError}
                error={projectsError}
                errorFallback={`Failed to load ${coreDomainPlural.toLowerCase()}.`}
                onRetry={() => void refetchProjects()}
                loadingFallback={
                    <DashboardCalendar
                        projects={[]}
                        projectsLoading
                        onOpenProjects={() => navigate("/projects")}
                        allowedViews={["day", "week", "month", "twelve_month"]}
                        initialView="month"
                    />
                }
            >
                <DashboardCalendar
                    projects={projects ?? []}
                    projectsLoading={false}
                    onOpenProjects={() => navigate("/projects")}
                    allowedViews={["day", "week", "month", "twelve_month"]}
                    initialView="month"
                />
            </QueryBoundary>
        </PageShell>
    );
}
