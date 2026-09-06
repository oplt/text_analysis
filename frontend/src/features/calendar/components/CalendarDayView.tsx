import dayjs from "dayjs";
import { Box, Button, Stack, Typography } from "@mui/material";
import { alpha, useTheme } from "@mui/material/styles";
import { formatDateOnly, humanizeKey } from "../../../utils/formatters";
import { formatItemTime, getMinutesFromTimeString, getWeekItemColor } from "../calendarDisplay";
import { DayItems } from "./DayItems";
import type { DashboardCalendarModel } from "../hooks/useDashboardCalendar";
import { MonthDateCalendar } from "./MonthDateCalendar";

export function CalendarDayView({ m }: { m: DashboardCalendarModel }) {
const theme = useTheme();
const { anchorDate, itemsByDate, openDay } = m;
    const dayKey = anchorDate.format("YYYY-MM-DD");
    const dayItems = itemsByDate[dayKey] ?? [];
    const timedItems = dayItems
        .filter((item) => item.start_time)
        .map((item) => {
            const startMinutes = getMinutesFromTimeString(item.start_time ?? "09:00");
            const endMinutes = item.end_time
                ? getMinutesFromTimeString(item.end_time)
                : startMinutes + 60;
            const clampedEnd = Math.max(endMinutes, startMinutes + 30);

            return {
                ...item,
                startMinutes,
                endMinutes: clampedEnd,
            };
        });
    const allDayItems = dayItems.filter((item) => !item.start_time);
    const hourSlots = Array.from({ length: 24 }, (_, index) => index);
    const rowHeight = 56;
    const gridStartMinutes = 0;
    const now = dayjs();
    const isToday = anchorDate.isSame(now, "day");
    const nowMinutes = now.hour() * 60 + now.minute();
    const nowTop = (nowMinutes / 60) * rowHeight;
    const selectedDetailItem = timedItems[0] ?? allDayItems[0] ?? dayItems[0] ?? null;

    return (
        <Box
            sx={{
                display: "grid",
                gap: 2,
                gridTemplateColumns: { xs: "1fr", xl: "minmax(0, 1.5fr) 360px" },
            }}
        >
            <Box
                sx={(theme) => ({
                    borderRadius: 4,
                    border: `1px solid ${theme.palette.divider}`,
                    overflow: "hidden",
                    backgroundColor: theme.palette.background.paper,
                })}
            >
                <Stack
                    direction={{ xs: "column", sm: "row" }}
                    justifyContent="space-between"
                    spacing={1.5}
                    sx={(theme) => ({
                        px: 2,
                        py: 1.5,
                        borderBottom: `1px solid ${theme.palette.divider}`,
                        backgroundColor: theme.palette.background.default,
                    })}
                >
                    <Box>
                        <Typography variant="h6">{formatDateOnly(dayKey)}</Typography>
                        <Typography variant="body2" color="text.secondary">
                            Focused view for one day of scheduled work.
                        </Typography>
                    </Box>
                    <Button variant="contained" onClick={() => openDay(anchorDate)}>
                        Add item
                    </Button>
                </Stack>

                {allDayItems.length > 0 && (
                    <Box
                        sx={(theme) => ({
                            px: 2,
                            py: 1.25,
                            borderBottom: `1px solid ${theme.palette.divider}`,
                            backgroundColor: theme.palette.background.paper,
                        })}
                    >
                        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 700 }}>
                            All day
                        </Typography>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 1 }}>
                            {allDayItems.map((item) => {
                                const colors = getWeekItemColor(item, theme);
                                return (
                                    <Box
                                        key={item.id}
                                        onClick={() => openDay(anchorDate)}
                                        sx={{
                                            px: 1.25,
                                            py: 0.8,
                                            borderRadius: 2,
                                            border: `1px solid ${colors.border}`,
                                            backgroundColor: colors.bg,
                                            color: colors.text,
                                            cursor: "pointer",
                                        }}
                                    >
                                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                            {item.title}
                                        </Typography>
                                    </Box>
                                );
                            })}
                        </Stack>
                    </Box>
                )}

                <Box
                    sx={{
                        display: "grid",
                        gridTemplateColumns: "88px minmax(0, 1fr)",
                        minHeight: rowHeight * hourSlots.length,
                        maxHeight: 760,
                        overflow: "auto",
                    }}
                >
                    <Box
                        sx={(theme) => ({
                            borderRight: `1px solid ${theme.palette.divider}`,
                            backgroundColor: theme.palette.background.default,
                        })}
                    >
                        {hourSlots.map((hour) => (
                            <Box
                                key={hour}
                                sx={(theme) => ({
                                    height: rowHeight,
                                    px: 1.25,
                                    pt: 0.55,
                                    borderBottom: `1px solid ${theme.palette.divider}`,
                                })}
                            >
                                <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
                                    {dayjs().hour(hour).minute(0).format("h A")}
                                </Typography>
                            </Box>
                        ))}
                    </Box>
                    <Box
                        sx={(theme) => ({
                            position: "relative",
                            backgroundColor: theme.palette.background.paper,
                        })}
                    >
                        {hourSlots.map((hour) => (
                            <Box
                                key={hour}
                                sx={(theme) => ({
                                    height: rowHeight,
                                    borderBottom: `1px solid ${theme.palette.divider}`,
                                })}
                            />
                        ))}

                        {isToday && (
                            <Box
                                sx={{
                                    position: "absolute",
                                    left: 0,
                                    right: 0,
                                    top: nowTop,
                                    height: 2,
                                    backgroundColor: "primary.main",
                                    zIndex: 2,
                                    "&::before": {
                                        content: '""',
                                        position: "absolute",
                                        left: -6,
                                        top: "50%",
                                        width: 10,
                                        height: 10,
                                        borderRadius: "999px",
                                        backgroundColor: "primary.main",
                                        transform: "translateY(-50%)",
                                    },
                                }}
                            />
                        )}

                        {timedItems.map((item) => {
                            const colors = getWeekItemColor(item, theme);
                            const top = ((item.startMinutes - gridStartMinutes) / 60) * rowHeight;
                            const height = Math.max(
                                ((item.endMinutes - item.startMinutes) / 60) * rowHeight,
                                44
                            );

                            return (
                                <Box
                                    key={item.id}
                                    onClick={() => openDay(anchorDate)}
                                    sx={{
                                        position: "absolute",
                                        top,
                                        left: 12,
                                        right: 12,
                                        height,
                                        px: 1.25,
                                        py: 1,
                                        borderRadius: 2,
                                        border: `1px solid ${colors.border}`,
                                        backgroundColor: colors.bg,
                                        color: colors.text,
                                        cursor: "pointer",
                                        overflow: "hidden",
                                        zIndex: 1,
                                    }}
                                >
                                    <Typography variant="body2" sx={{ fontWeight: 700, lineHeight: 1.3 }}>
                                        {item.title}
                                    </Typography>
                                    <Typography variant="body2" sx={{ mt: 0.4 }}>
                                        {formatItemTime(item)}
                                    </Typography>
                                </Box>
                            );
                        })}
                    </Box>
                </Box>
            </Box>

            <Box
                sx={(theme) => ({
                    borderRadius: 4,
                    border: `1px solid ${theme.palette.divider}`,
                    overflow: "hidden",
                    backgroundColor: theme.palette.background.paper,
                })}
            >
                <Box
                    sx={(theme) => ({
                        p: 1.25,
                        borderBottom: `1px solid ${theme.palette.divider}`,
                        backgroundColor: alpha(
                            theme.palette.background.paper,
                            theme.palette.mode === "dark" ? 0.9 : 0.78
                        ),
                    })}
                >
                    <MonthDateCalendar m={m} date={anchorDate} />
                </Box>
                <Box sx={{ p: 2 }}>
                    {selectedDetailItem ? (
                        <Stack spacing={2}>
                            <Box>
                                <Typography variant="h6">{selectedDetailItem.title}</Typography>
                                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                                    {selectedDetailItem.description || "No extra notes for this item."}
                                </Typography>
                            </Box>
                            <Stack spacing={1}>
                                <Typography variant="body2" color="text.secondary">
                                    {formatDateOnly(dayKey)}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    {formatItemTime(selectedDetailItem)}
                                </Typography>
                                {selectedDetailItem.project_name && (
                                    <Typography variant="body2" color="text.secondary">
                                        Project: {selectedDetailItem.project_name}
                                    </Typography>
                                )}
                                {selectedDetailItem.priority && (
                                    <Typography variant="body2" color="text.secondary">
                                        Priority: {humanizeKey(selectedDetailItem.priority)}
                                    </Typography>
                                )}
                            </Stack>
                            <Button variant="outlined" onClick={() => openDay(anchorDate)}>
                                Edit or add item
                            </Button>
                        </Stack>
                    ) : (
                        <DayItems
                            items={dayItems}
                            emptyTitle="Nothing scheduled for this day"
                            emptyDescription="Use Add item to create an event, appointment, or task."
                        />
                    )}
                </Box>
            </Box>
        </Box>
    );
}
