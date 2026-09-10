import { Box, Button, Skeleton, Stack } from "@mui/material";
import { ChevronLeft as ChevronLeftIcon, ChevronRight as ChevronRightIcon } from "@mui/icons-material";
import { QueryErrorAlert } from "../ui/QueryBoundary";
import { SectionCard } from "../ui/SectionCard";
import { CalendarItemDrawer } from "../../features/calendar/components/CalendarItemDrawer";
import { CalendarViews } from "../../features/calendar/components/CalendarViews";
import { getMonthGridColumns, shiftCalendarAnchor } from "../../features/calendar/calendarRange";
import { VIEW_OPTIONS, type DashboardCalendarProps } from "../../features/calendar/calendarDisplay";
import { useDashboardCalendar } from "../../features/calendar/hooks/useDashboardCalendar";

export function DashboardCalendar(props: DashboardCalendarProps) {
    const m = useDashboardCalendar(props);
    return (
        <Box sx={{ height: "100%", minHeight: 0, display: "flex", flexDirection: "column" }}>
            <SectionCard
                title="Workspace calendar"
                description="Switch between focused day planning, weekly scheduling, monthly scanning, and a 12-month horizon."
                sx={props.sx}
                contentSx={props.contentSx}
                action={
                    <Stack spacing={1} alignItems={{ xs: "stretch", sm: "flex-end" }}>
                        {m.allowedViews.length > 1 && (
                            <Stack
                                direction="row"
                                spacing={0.75}
                                flexWrap="wrap"
                                useFlexGap
                                justifyContent="flex-end"
                            >
                                {VIEW_OPTIONS.filter((option) =>
                                    m.allowedViews.includes(option.value)
                                ).map((option) => (
                                    <Button
                                        key={option.value}
                                        size="small"
                                        variant={m.viewMode === option.value ? "contained" : "outlined"}
                                        onClick={() => m.setViewMode(option.value)}
                                    >
                                        {option.label}
                                    </Button>
                                ))}
                            </Stack>
                        )}
                        <Stack direction="row" spacing={0.75} justifyContent="flex-end">
                            <Button
                                size="small"
                                variant="text"
                                startIcon={<ChevronLeftIcon />}
                                onClick={() =>
                                    m.setAnchorDate((current) =>
                                        shiftCalendarAnchor(current, m.viewMode, -1)
                                    )
                                }
                            >
                                Back
                            </Button>
                            <Button
                                size="small"
                                variant="text"
                                endIcon={<ChevronRightIcon />}
                                onClick={() =>
                                    m.setAnchorDate((current) =>
                                        shiftCalendarAnchor(current, m.viewMode, 1)
                                    )
                                }
                            >
                                Forward
                            </Button>
                        </Stack>
                    </Stack>
                }
            >
                {m.itemsQuery.isError && (
                    <Box sx={{ mb: 2 }}>
                        <QueryErrorAlert
                            error={m.itemsQuery.error}
                            fallback="Failed to load calendar items."
                            onRetry={() => void m.itemsQuery.refetch()}
                        />
                    </Box>
                )}
                {m.itemsQuery.isLoading ? (
                    m.viewMode === "day" || m.viewMode === "week" ? (
                        <Stack spacing={2}>
                            <Skeleton variant="rounded" height={340} sx={{ borderRadius: 4 }} />
                            <Skeleton variant="rounded" height={220} sx={{ borderRadius: 4 }} />
                        </Stack>
                    ) : (
                        <Box
                            sx={{
                                display: "grid",
                                gap: 1.5,
                                gridTemplateColumns: getMonthGridColumns(m.viewMode),
                            }}
                        >
                            {m.visibleMonths.map((month) => (
                                <Skeleton
                                    key={month.format("YYYY-MM")}
                                    variant="rounded"
                                    height={m.viewMode === "twelve_month" ? 360 : 420}
                                    sx={{ borderRadius: 4 }}
                                />
                            ))}
                        </Box>
                    )
                ) : (
                    <CalendarViews m={m} />
                )}
            </SectionCard>
            <CalendarItemDrawer m={m} />
        </Box>
    );
}
