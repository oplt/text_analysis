import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
    Alert,
    Box,
    Button,
    CircularProgress,
    Paper,
    Skeleton,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import {
    AddCircleOutline as AddCircleOutlineIcon,
    FolderOpen as FolderOpenIcon,
    ArrowForward as ArrowForwardIcon,
} from "@mui/icons-material";
import { alpha } from "@mui/material/styles";
import { useNavigate } from "react-router-dom";
import { createProject, listProjects } from "../api/projects";
import { queryKeys } from "../config/queryKeys";
import { useSnackbar } from "../app/snackbarContext";
import { EmptyState } from "../components/ui/EmptyState";
import { PageShell } from "../components/ui/PageShell";
import { QueryBoundary } from "../components/ui/QueryBoundary";
import { SectionCard } from "../components/ui/SectionCard";
import { usePlatformMetadata } from "../hooks/usePlatformMetadata";
import { formatDate } from "../utils/formatters";

const projectSchema = z.object({
    name: z.string().min(2, "Name must be at least 2 characters").max(255),
    description: z.string().optional(),
});

type ProjectValues = z.infer<typeof projectSchema>;

export default function ProjectsPage() {
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const { showToast } = useSnackbar();
    const { data: platformMetadata } = usePlatformMetadata();
    const { data: projects, isLoading, isError, error, refetch } = useQuery({
        queryKey: queryKeys.projects.all,
        queryFn: listProjects,
    });
    const {
        register,
        handleSubmit,
        reset,
        formState: { errors },
    } = useForm<ProjectValues>({ resolver: zodResolver(projectSchema) });

    const mutation = useMutation({
        mutationFn: createProject,
        onSuccess: async (project) => {
            await queryClient.invalidateQueries({ queryKey: queryKeys.projects.all });
            reset();
            showToast({ message: "Project created successfully.", severity: "success" });
            navigate(`/projects/${project.id}`);
        },
    });

    const coreDomainSingular = platformMetadata?.core_domain_singular ?? "Project";
    const coreDomainPlural = platformMetadata?.core_domain_plural ?? "Projects";

    return (
        <PageShell maxWidth="xl">
            <Box
                sx={{
                    display: "grid",
                    gap: 2,
                    gridTemplateColumns: { xs: "1fr", lg: "minmax(320px, 400px) minmax(0, 1fr)" },
                    alignItems: "start",
                }}
            >
                <SectionCard
                    title={`Create ${coreDomainSingular}`}
                    description={`Start a new ${coreDomainSingular.toLowerCase()} with a clear name and optional context.`}
                >
                    <Box component="form" onSubmit={handleSubmit((values) => mutation.mutate(values))}>
                        <Stack spacing={2}>
                            <TextField
                                label={`${coreDomainSingular} name`}
                                placeholder={`Acme ${coreDomainSingular}`}
                                {...register("name")}
                                error={!!errors.name}
                                helperText={errors.name?.message}
                                fullWidth
                            />
                            <TextField
                                label="Description"
                                placeholder={`What is this ${coreDomainSingular.toLowerCase()} for?`}
                                {...register("description")}
                                error={!!errors.description}
                                helperText={errors.description?.message}
                                fullWidth
                                multiline
                                minRows={4}
                            />
                            {mutation.isError && (
                                <Alert severity="error">
                                    {mutation.error instanceof Error
                                        ? mutation.error.message
                                        : `Failed to create ${coreDomainSingular.toLowerCase()}.`}
                                </Alert>
                            )}
                            <Button
                                type="submit"
                                variant="contained"
                                disabled={mutation.isPending}
                                startIcon={mutation.isPending ? <CircularProgress size={16} /> : <AddCircleOutlineIcon />}
                            >
                                {mutation.isPending ? "Creating..." : `Create ${coreDomainSingular}`}
                            </Button>
                        </Stack>
                    </Box>
                </SectionCard>

                <SectionCard
                    title={`Your ${coreDomainPlural}`}
                    description={`Browse the current ${coreDomainPlural.toLowerCase()} and scan for missing descriptions or naming gaps.`}
                >
                    <QueryBoundary
                        isLoading={isLoading}
                        isError={isError}
                        error={error}
                        errorFallback={`Failed to load ${coreDomainPlural.toLowerCase()}.`}
                        onRetry={() => void refetch()}
                        loadingFallback={
                            <Box
                                sx={{
                                    display: "grid",
                                    gap: 1.5,
                                    gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))" },
                                }}
                            >
                                {Array.from({ length: 4 }).map((_, index) => (
                                    <Skeleton key={index} variant="rounded" height={162} sx={{ borderRadius: 4 }} />
                                ))}
                            </Box>
                        }
                        isEmpty={!projects || projects.length === 0}
                        emptyFallback={
                            <EmptyState
                                icon={<FolderOpenIcon />}
                                title={`No ${coreDomainPlural.toLowerCase()} yet`}
                                description={`Create the first ${coreDomainSingular.toLowerCase()} to give the workspace structure and momentum.`}
                            />
                        }
                    >
                        <Box
                            sx={{
                                display: "grid",
                                gap: 1.5,
                                gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))" },
                            }}
                        >
                            {projects?.map((project) => (
                                <Paper
                                    key={project.id}
                                    sx={(theme) => ({
                                        p: 2.5,
                                        borderRadius: 4,
                                        border: `1px solid ${theme.palette.divider}`,
                                        backgroundColor: theme.palette.background.paper,
                                    })}
                                >
                                    <Stack spacing={1.25}>
                                        <Stack direction="row" spacing={1.25} alignItems="center">
                                            <Box
                                                sx={(theme) => ({
                                                    width: 42,
                                                    height: 42,
                                                    borderRadius: 3,
                                                    display: "grid",
                                                    placeItems: "center",
                                                    color: "primary.main",
                                                    backgroundColor: alpha(theme.palette.primary.main, theme.palette.mode === "dark" ? 0.16 : 0.1),
                                                })}
                                            >
                                                <FolderOpenIcon />
                                            </Box>
                                            <Box>
                                                <Typography variant="subtitle1">{project.name}</Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    Created {formatDate(project.created_at)}
                                                </Typography>
                                            </Box>
                                        </Stack>
                                        <Typography variant="body2" color="text.secondary">
                                            {project.description || `No description yet for this ${coreDomainSingular.toLowerCase()}.`}
                                        </Typography>
                                        <Box>
                                            <Button
                                                variant="text"
                                                endIcon={<ArrowForwardIcon />}
                                                onClick={() => navigate(`/projects/${project.id}`)}
                                                sx={{ px: 0 }}
                                            >
                                                Open workspace
                                            </Button>
                                        </Box>
                                    </Stack>
                                </Paper>
                            ))}
                        </Box>
                    </QueryBoundary>
                </SectionCard>
            </Box>
        </PageShell>
    );
}
