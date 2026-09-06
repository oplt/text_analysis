import {
    type PropsWithChildren,
} from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { logout as logoutRequest, me, type AuthUser } from "../../../api/auth";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES } from "../../../config/queryTiming";
import { AuthContext } from "./authContext";

export function AuthProvider({ children }: PropsWithChildren) {
    const queryClient = useQueryClient();
    const {
        data: currentUser = null,
        isPending,
        isError,
    } = useQuery<AuthUser | null>({
        queryKey: queryKeys.auth.me,
        queryFn: me,
        retry: false,
        staleTime: QUERY_STALE_TIMES.userProfile,
    });

    const isAuthenticated = currentUser !== null;
    const isReady = !isPending || isError;

    async function logout() {
        await logoutRequest().catch(() => undefined);
        queryClient.setQueryData(queryKeys.auth.me, null);
    }

    function setAuthenticated(user: AuthUser) {
        queryClient.setQueryData(queryKeys.auth.me, user);
    }

    return (
        <AuthContext.Provider
            value={{
                isReady,
                isAuthenticated,
                isAdmin: currentUser?.is_admin ?? false,
                isMfaEnabled: currentUser?.mfa_enabled ?? false,
                currentUser,
                logout,
                setAuthenticated,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
}
