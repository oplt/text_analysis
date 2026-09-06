import { createContext, useContext } from "react";
import type { AuthUser } from "../../../api/auth";

export type AuthContextValue = {
    isReady: boolean;
    isAuthenticated: boolean;
    isAdmin: boolean;
    isMfaEnabled: boolean;
    currentUser: AuthUser | null;
    logout: () => Promise<void>;
    setAuthenticated: (user: AuthUser) => void;
};

export const AuthContext = createContext<AuthContextValue>({
    isReady: false,
    isAuthenticated: false,
    isAdmin: false,
    isMfaEnabled: false,
    currentUser: null,
    logout: async () => undefined,
    setAuthenticated: () => undefined,
});

export function useAuth() {
    return useContext(AuthContext);
}
