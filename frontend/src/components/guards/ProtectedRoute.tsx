import { Navigate } from "react-router-dom";
import { Box, CircularProgress } from "@mui/material";

type Props = {
    isReady: boolean;
    isAuthenticated: boolean;
    isAdmin?: boolean;
    isMfaEnabled?: boolean;
    requireAdmin?: boolean;
    requireMfa?: boolean;
    redirectTo?: string;
    children?: React.ReactNode;
};

export function ProtectedRoute({
    isReady,
    isAuthenticated,
    isAdmin = false,
    isMfaEnabled = false,
    requireAdmin = false,
    requireMfa = false,
    redirectTo = "/",
    children,
}: Props) {
    if (!isReady) {
        return (
            <Box sx={{ display: "grid", placeItems: "center", minHeight: "100vh" }}>
                <CircularProgress />
            </Box>
        );
    }
    if (!isAuthenticated) return <Navigate to={redirectTo} replace />;
    if (requireAdmin && !isAdmin) return <Navigate to="/dashboard" replace />;
    if (requireMfa && !isMfaEnabled) return <Navigate to="/profile" replace />;
    return <>{children}</>;
}
