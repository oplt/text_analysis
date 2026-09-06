import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Box, Skeleton, Stack } from "@mui/material";
import { ProtectedRoute } from "../components/guards/ProtectedRoute";
import { useAuth } from "../hooks/useAuth";
import AuthHomePage from "../pages/AuthHomePage";

const AppLayout = lazy(() =>
    import("../components/layout/AppLayout").then((module) => ({ default: module.AppLayout }))
);
const DashboardPage = lazy(() => import("../pages/DashboardPage"));
const CalendarPage = lazy(() => import("../pages/CalendarPage"));
const ProjectsPage = lazy(() => import("../pages/ProjectsPage"));
const ProjectDetailPage = lazy(() => import("../pages/ProjectDetailPage"));
const PlatformPage = lazy(() => import("../pages/PlatformPage"));
const ProfilePage = lazy(() => import("../pages/ProfilePage"));
const NotificationsPage = lazy(() => import("../pages/NotificationsPage"));
const ObservabilityPage = lazy(() => import("../pages/ObservabilityPage"));
const ResetPasswordPage = lazy(() => import("../pages/ResetPasswordPage"));
const VerifyEmailPage = lazy(() => import("../pages/VerifyEmailPage"));
const AdminUsersPage = lazy(() => import("../pages/AdminUsersPage"));
const AdminPlatformPage = lazy(() => import("../pages/AdminPlatformPage"));
const AdminSettingsPage = lazy(() => import("../pages/AdminSettingsPage"));
const AiStudioPage = lazy(() => import("../pages/AiStudioPage"));
const ResearchPage = lazy(() => import("../pages/ResearchPage"));
const DashboardView = lazy(() =>
    import("../features/text-research/views/DashboardView").then((m) => ({ default: m.default }))
);
const CorpusView = lazy(() =>
    import("../features/text-research/views/CorpusView").then((m) => ({ default: m.default }))
);
const AnnotationView = lazy(() =>
    import("../features/text-research/views/AnnotationView").then((m) => ({ default: m.default }))
);
const ReliabilityView = lazy(() =>
    import("../features/text-research/views/ReliabilityView").then((m) => ({ default: m.default }))
);
const AnalysisView = lazy(() =>
    import("../features/text-research/views/AnalysisView").then((m) => ({ default: m.default }))
);
const ClassificationView = lazy(() =>
    import("../features/text-research/views/ClassificationView").then((m) => ({ default: m.default }))
);
const TopicsView = lazy(() =>
    import("../features/text-research/views/TopicsView").then((m) => ({ default: m.default }))
);
const RobustnessView = lazy(() =>
    import("../features/text-research/views/RobustnessView").then((m) => ({ default: m.default }))
);
const ExplorerView = lazy(() =>
    import("../features/text-research/views/ExplorerView").then((m) => ({ default: m.default }))
);
const RunsView = lazy(() =>
    import("../features/text-research/views/RunsView").then((m) => ({ default: m.default }))
);

function PageLoader() {
    return (
        <Box sx={{ px: { xs: 2, md: 3 }, py: { xs: 3, md: 4 } }}>
            <Stack spacing={3}>
                <Skeleton variant="rounded" height={170} sx={{ borderRadius: 2 }} />
                <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
                    <Skeleton variant="rounded" height={168} sx={{ borderRadius: 2, flex: 1 }} />
                    <Skeleton variant="rounded" height={168} sx={{ borderRadius: 2, flex: 1 }} />
                    <Skeleton variant="rounded" height={168} sx={{ borderRadius: 2, flex: 1 }} />
                </Stack>
                <Skeleton variant="rounded" height={260} sx={{ borderRadius: 5 }} />
            </Stack>
        </Box>
    );
}

function SuspensePage({ children }: { children: React.ReactNode }) {
    return <Suspense fallback={<PageLoader />}>{children}</Suspense>;
}

export function AppRouter() {
    const { isReady, isAuthenticated, isAdmin } = useAuth();

    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<AuthHomePage />} />
                <Route path="/reset-password" element={<SuspensePage><ResetPasswordPage /></SuspensePage>} />
                <Route path="/verify-email" element={<SuspensePage><VerifyEmailPage /></SuspensePage>} />

                <Route
                    element={
                        <ProtectedRoute isReady={isReady} isAuthenticated={isAuthenticated}>
                            <Suspense fallback={<PageLoader />}>
                                <AppLayout />
                            </Suspense>
                        </ProtectedRoute>
                    }
                >
                    <Route path="/dashboard" element={<SuspensePage><DashboardPage /></SuspensePage>} />
                    <Route path="/calendar" element={<SuspensePage><CalendarPage /></SuspensePage>} />
                    <Route path="/projects" element={<SuspensePage><ProjectsPage /></SuspensePage>} />
                    <Route path="/projects/:projectId" element={<SuspensePage><ProjectDetailPage /></SuspensePage>} />
                    <Route path="/research/:projectId" element={<SuspensePage><ResearchPage /></SuspensePage>}>
                        <Route index element={<Navigate to="dashboard" replace />} />
                        <Route path="dashboard" element={<SuspensePage><DashboardView /></SuspensePage>} />
                        <Route path="corpus" element={<SuspensePage><CorpusView /></SuspensePage>} />
                        <Route path="annotation" element={<SuspensePage><AnnotationView /></SuspensePage>} />
                        <Route path="reliability" element={<SuspensePage><ReliabilityView /></SuspensePage>} />
                        <Route path="analysis" element={<SuspensePage><AnalysisView /></SuspensePage>} />
                        <Route path="classification" element={<SuspensePage><ClassificationView /></SuspensePage>} />
                        <Route path="topics" element={<SuspensePage><TopicsView /></SuspensePage>} />
                        <Route path="robustness" element={<SuspensePage><RobustnessView /></SuspensePage>} />
                        <Route path="explorer" element={<SuspensePage><ExplorerView /></SuspensePage>} />
                        <Route path="runs" element={<SuspensePage><RunsView /></SuspensePage>} />
                        <Route path="exports" element={<Navigate to="../runs" replace />} />
                    </Route>
                    <Route path="/platform" element={<SuspensePage><PlatformPage /></SuspensePage>} />
                    <Route path="/ai" element={<SuspensePage><AiStudioPage /></SuspensePage>} />
                    <Route path="/observability" element={<SuspensePage><ObservabilityPage /></SuspensePage>} />
                    <Route path="/profile" element={<SuspensePage><ProfilePage /></SuspensePage>} />
                    <Route path="/notifications" element={<SuspensePage><NotificationsPage /></SuspensePage>} />
                    <Route
                        path="/admin/users"
                        element={
                            <ProtectedRoute
                                isReady={isReady}
                                isAuthenticated={isAuthenticated}
                                isAdmin={isAdmin}
                                requireAdmin
                            >
                                <SuspensePage><AdminUsersPage /></SuspensePage>
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/admin/platform"
                        element={
                            <ProtectedRoute
                                isReady={isReady}
                                isAuthenticated={isAuthenticated}
                                isAdmin={isAdmin}
                                requireAdmin
                            >
                                <SuspensePage><AdminPlatformPage /></SuspensePage>
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/admin/settings"
                        element={
                            <ProtectedRoute
                                isReady={isReady}
                                isAuthenticated={isAuthenticated}
                                isAdmin={isAdmin}
                                requireAdmin
                            >
                                <SuspensePage><AdminSettingsPage /></SuspensePage>
                            </ProtectedRoute>
                        }
                    />
                    <Route path="/app" element={<Navigate to="/dashboard" replace />} />
                </Route>

                <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
        </BrowserRouter>
    );
}
