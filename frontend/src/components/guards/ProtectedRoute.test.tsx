import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { describe, expect, it } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { ProtectedRoute } from "./ProtectedRoute";

type GuardRoute = {
    path: string;
    redirectPath: string;
    redirectLabel: string;
};

const ADMIN_ROUTES: GuardRoute[] = [
    { path: "/admin/users", redirectPath: "/dashboard", redirectLabel: "Dashboard page" },
    { path: "/admin/platform", redirectPath: "/dashboard", redirectLabel: "Dashboard page" },
    { path: "/admin/settings", redirectPath: "/dashboard", redirectLabel: "Dashboard page" },
];

const WORKSPACE_ROUTES: GuardRoute[] = [
    { path: "/ai", redirectPath: "/", redirectLabel: "Landing page" },
    { path: "/projects", redirectPath: "/", redirectLabel: "Landing page" },
];

function renderGuard(
    path: string,
    props: React.ComponentProps<typeof ProtectedRoute>,
) {
    render(
        <MemoryRouter initialEntries={[path]}>
            <Routes>
                <Route
                    path={path}
                    element={
                        <ProtectedRoute {...props}>
                            <div>Protected content</div>
                        </ProtectedRoute>
                    }
                />
                <Route path="/" element={<div>Landing page</div>} />
                <Route path="/dashboard" element={<div>Dashboard page</div>} />
                <Route path="/profile" element={<div>Profile page</div>} />
            </Routes>
        </MemoryRouter>
    );
}

describe("ProtectedRoute", () => {
    it("redirects unauthenticated users", async () => {
        renderGuard("/admin/users", {
            isReady: true,
            isAuthenticated: false,
            redirectTo: "/",
        });

        expect(await screen.findByText("Landing page")).toBeInTheDocument();
    });

    it("redirects non-admin users away from admin routes", async () => {
        renderGuard("/admin/users", {
            isReady: true,
            isAuthenticated: true,
            isAdmin: false,
            requireAdmin: true,
        });

        expect(await screen.findByText("Dashboard page")).toBeInTheDocument();
    });

    it("renders children for authorized users", async () => {
        renderGuard("/admin/users", {
            isReady: true,
            isAuthenticated: true,
            isAdmin: true,
            requireAdmin: true,
        });

        expect(await screen.findByText("Protected content")).toBeInTheDocument();
    });

    it("redirects admins without MFA to profile when MFA is required", async () => {
        renderGuard("/admin/users", {
            isReady: true,
            isAuthenticated: true,
            isAdmin: true,
            isMfaEnabled: false,
            requireAdmin: true,
            requireMfa: true,
        });

        expect(await screen.findByText("Profile page")).toBeInTheDocument();
    });

    it.each(ADMIN_ROUTES)(
        "blocks non-admin access to $path",
        async ({ path, redirectLabel }) => {
            renderGuard(path, {
                isReady: true,
                isAuthenticated: true,
                isAdmin: false,
                requireAdmin: true,
            });

            expect(await screen.findByText(redirectLabel)).toBeInTheDocument();
        }
    );

    it.each(WORKSPACE_ROUTES)(
        "redirects signed-out users from $path",
        async ({ path, redirectLabel }) => {
            renderGuard(path, {
                isReady: true,
                isAuthenticated: false,
            });

            expect(await screen.findByText(redirectLabel)).toBeInTheDocument();
        }
    );
});
