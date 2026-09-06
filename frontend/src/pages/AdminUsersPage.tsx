import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
    Box,
    Chip,
    CircularProgress,
    InputAdornment,
    Skeleton,
    Stack,
    Switch,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TablePagination,
    TableRow,
    TextField,
    Tooltip,
    Typography,
    useMediaQuery,
} from "@mui/material";
import { Search as SearchIcon, PeopleAlt as PeopleAltIcon } from "@mui/icons-material";
import { useTheme } from "@mui/material/styles";
import { listAdminUsers, updateUserStatus, type AdminUserListResponse } from "../api/admin";
import { AdminSettingsTabs } from "../components/layout/AdminSettingsTabs";
import { EmptyState } from "../components/ui/EmptyState";
import { PageShell } from "../components/ui/PageShell";
import { QueryBoundary } from "../components/ui/QueryBoundary";
import { SectionCard } from "../components/ui/SectionCard";
import { StatCard } from "../components/ui/StatCard";
import { queryKeys } from "../config/queryKeys";
import { useDebounce } from "../hooks/useDebounce";
import { useMutationErrorToast } from "../hooks/useMutationErrorToast";
import { formatDate } from "../utils/formatters";

export default function AdminUsersPage() {
    const queryClient = useQueryClient();
    const toastMutationError = useMutationErrorToast();
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down("md"));
    const [search, setSearch] = useState("");
    const [page, setPage] = useState(0);
    const pageSize = 20;
    const debouncedSearch = useDebounce(search, 300);
    const usersQueryKey = queryKeys.admin.users(page, debouncedSearch);

    const { data, isLoading, isError, error, refetch } = useQuery({
        queryKey: usersQueryKey,
        queryFn: () =>
            listAdminUsers({
                page: page + 1,
                page_size: pageSize,
                search: debouncedSearch || undefined,
            }),
    });
    const statusMutation = useMutation({
        mutationFn: ({ id, is_active }: { id: string; is_active: boolean }) =>
            updateUserStatus(id, { is_active }),
        onMutate: async ({ id, is_active }) => {
            await queryClient.cancelQueries({ queryKey: queryKeys.admin.all });
            const previous = queryClient.getQueryData<AdminUserListResponse>(usersQueryKey);
            queryClient.setQueryData<AdminUserListResponse>(usersQueryKey, (old) =>
                old
                    ? {
                          ...old,
                          items: old.items.map((user) =>
                              user.id === id ? { ...user, is_active } : user
                          ),
                      }
                    : old
            );
            return { previous, queryKey: usersQueryKey };
        },
        onError: (mutationError, _vars, context) => {
            if (context?.previous) {
                queryClient.setQueryData(context.queryKey, context.previous);
            }
            toastMutationError(mutationError, "Failed to update user status.");
        },
        onSettled: () => void queryClient.invalidateQueries({ queryKey: queryKeys.admin.all }),
    });

    const users = data?.items ?? [];
    const activeCount = users.filter((user) => user.is_active).length;
    const verifiedCount = users.filter((user) => user.is_verified).length;

    return (
        <PageShell maxWidth="xl">
            <AdminSettingsTabs />

            <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 2 }}>
                <TextField
                    size="small"
                    placeholder="Search users..."
                    value={search}
                    onChange={(event) => {
                        setSearch(event.target.value);
                        setPage(0);
                    }}
                    InputProps={{
                        startAdornment: (
                            <InputAdornment position="start">
                                <SearchIcon fontSize="small" />
                            </InputAdornment>
                        ),
                    }}
                    sx={{ width: { xs: "100%", sm: 320 } }}
                />
            </Box>

            <Box
                sx={{
                    display: "grid",
                    gap: 2,
                    gridTemplateColumns: { xs: "1fr", md: "repeat(3, minmax(0, 1fr))" },
                }}
            >
                <StatCard
                    label="Total users"
                    value={data?.total ?? 0}
                    description="Accounts matching the current search and pagination scope"
                    icon={<PeopleAltIcon />}
                    loading={isLoading}
                />
                <StatCard
                    label="Active on page"
                    value={activeCount}
                    description="Users currently allowed to access the product"
                    icon={<PeopleAltIcon />}
                    loading={isLoading}
                    color="success"
                />
                <StatCard
                    label="Verified on page"
                    value={verifiedCount}
                    description="Accounts with confirmed email identity"
                    icon={<PeopleAltIcon />}
                    loading={isLoading}
                    color="secondary"
                />
            </Box>

            <SectionCard title="User directory" description="Review roles, verification, and status changes from one place.">
                <QueryBoundary
                    isLoading={isLoading}
                    isError={isError}
                    error={error}
                    errorFallback="Failed to load users."
                    onRetry={() => void refetch()}
                    loadingFallback={
                        <Stack spacing={1.5}>
                            {Array.from({ length: 5 }).map((_, index) => (
                                <Skeleton key={index} variant="rounded" height={88} sx={{ borderRadius: 4 }} />
                            ))}
                        </Stack>
                    }
                    isEmpty={users.length === 0}
                    emptyFallback={
                        <EmptyState
                            icon={<PeopleAltIcon />}
                            title="No users found"
                            description="Try broadening the search or check if the current filters are too narrow."
                        />
                    }
                >
                    {isMobile ? (
                        <Stack spacing={1.5}>
                            {users.map((user) => {
                                const isUpdatingThisUser =
                                    statusMutation.isPending &&
                                    statusMutation.variables?.id === user.id;
                                return (
                                    <Box
                                        key={user.id}
                                        sx={(currentTheme) => ({
                                            p: 2.25,
                                            borderRadius: 4,
                                            border: `1px solid ${currentTheme.palette.divider}`,
                                        })}
                                    >
                                        <Stack spacing={1.25}>
                                            <Stack direction="row" justifyContent="space-between" spacing={1}>
                                                <Box sx={{ minWidth: 0 }}>
                                                    <Typography variant="subtitle2" noWrap>{user.full_name ?? "Unnamed user"}</Typography>
                                                    <Typography variant="body2" color="text.secondary" noWrap>
                                                        {user.email}
                                                    </Typography>
                                                </Box>
                                                <Switch
                                                    checked={user.is_active}
                                                    size="small"
                                                    disabled={isUpdatingThisUser}
                                                    inputProps={{
                                                        "aria-label": `${user.is_active ? "Deactivate" : "Activate"} ${user.full_name ?? user.email}`,
                                                    }}
                                                    onChange={(event) =>
                                                        statusMutation.mutate({
                                                            id: user.id,
                                                            is_active: event.target.checked,
                                                        })
                                                    }
                                                />
                                            </Stack>
                                            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                                {user.roles.map((role) => (
                                                    <Chip key={role} label={role} size="small" />
                                                ))}
                                                <Chip
                                                    label={user.is_verified ? "Verified" : "Unverified"}
                                                    size="small"
                                                    color={user.is_verified ? "success" : "warning"}
                                                    variant="outlined"
                                                />
                                            </Stack>
                                            <Typography variant="caption" color="text.secondary">
                                                Joined {formatDate(user.created_at)}
                                            </Typography>
                                        </Stack>
                                    </Box>
                                );
                            })}
                        </Stack>
                    ) : (
                        <TableContainer>
                            <Table>
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Email</TableCell>
                                        <TableCell>Name</TableCell>
                                        <TableCell>Roles</TableCell>
                                        <TableCell>Verified</TableCell>
                                        <TableCell>Joined</TableCell>
                                        <TableCell align="center">Active</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {users.map((user) => {
                                        const isUpdatingThisUser =
                                            statusMutation.isPending &&
                                            statusMutation.variables?.id === user.id;
                                        return (
                                            <TableRow key={user.id} hover>
                                                <TableCell>{user.email}</TableCell>
                                                <TableCell>{user.full_name ?? "—"}</TableCell>
                                                <TableCell>
                                                    <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                                                        {user.roles.map((role) => (
                                                            <Chip key={role} label={role} size="small" />
                                                        ))}
                                                    </Stack>
                                                </TableCell>
                                                <TableCell>
                                                    <Chip
                                                        label={user.is_verified ? "Verified" : "Unverified"}
                                                        size="small"
                                                        color={user.is_verified ? "success" : "warning"}
                                                        variant="outlined"
                                                    />
                                                </TableCell>
                                                <TableCell>{formatDate(user.created_at)}</TableCell>
                                                <TableCell align="center">
                                                    <Tooltip title={user.is_active ? "Deactivate" : "Activate"}>
                                                        <Box component="span">
                                                            {isUpdatingThisUser ? (
                                                                <CircularProgress size={18} />
                                                            ) : (
                                                                <Switch
                                                                    checked={user.is_active}
                                                                    size="small"
                                                                    inputProps={{
                                                                        "aria-label": `${user.is_active ? "Deactivate" : "Activate"} ${user.full_name ?? user.email}`,
                                                                    }}
                                                                    onChange={(event) =>
                                                                        statusMutation.mutate({
                                                                            id: user.id,
                                                                            is_active: event.target.checked,
                                                                        })
                                                                    }
                                                                />
                                                            )}
                                                        </Box>
                                                    </Tooltip>
                                                </TableCell>
                                            </TableRow>
                                        );
                                    })}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    )}
                </QueryBoundary>

                <TablePagination
                    component="div"
                    count={data?.total ?? 0}
                    page={page}
                    rowsPerPage={pageSize}
                    rowsPerPageOptions={[pageSize]}
                    onPageChange={(_, nextPage) => setPage(nextPage)}
                />
            </SectionCard>
        </PageShell>
    );
}
