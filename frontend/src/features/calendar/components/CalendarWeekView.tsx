import dayjs from "dayjs";
import { Box, Stack, Typography } from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { formatTimeValue, getMinutesFromTimeString, getWeekItemColor } from "../calendarDisplay";
import type { DashboardCalendarModel } from "../hooks/useDashboardCalendar";

export function CalendarWeekView({ m }: { m: DashboardCalendarModel }) {
const theme = useTheme();
const { anchorDate, itemsByDate, openDay } = m;
    const weekDays = Array.from({ length: 7 }, (_, index) =>
        anchorDate.startOf("week").add(index, "day")
    );
    const hourSlots = Array.from({ length: 10 }, (_, index) => 9 + index);
    const rowHeight = 88;
    const dayColumnWidth = 194;

    const timedWeekItems = weekDays.map((day) => {
        const dateKey = day.format("YYYY-MM-DD");
        return (itemsByDate[dateKey] ?? [])
            .filter((item) => item.start_time)
            .map((item) => {
                const startMinutes = getMinutesFromTimeString(item.start_time ?? "09:00");
                const endMinutes = item.end_time
                    ? getMinutesFromTimeString(item.end_time)
                    : startMinutes + 60;
                const clampedStart = Math.max(9 * 60, startMinutes);
                const clampedEnd = Math.max(clampedStart + 30, endMinutes);

                return {
                    ...item,
                    top: ((clampedStart - 9 * 60) / 60) * rowHeight,
                    height: Math.max(((clampedEnd - clampedStart) / 60) * rowHeight, 44),
                };
            });
    });

    const allDayWeekItems = weekDays.map((day) => {
        const dateKey = day.format("YYYY-MM-DD");
        return (itemsByDate[dateKey] ?? []).filter((item) => !item.start_time);
    });

    return (
        <Box sx={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
            <Box
                sx={(theme) => ({
                    borderRadius: 4,
                    border: `1px solid ${theme.palette.divider}`,
                    overflow: "hidden",
                    backgroundColor: theme.palette.background.paper,
                    minWidth: 980,
                })}
            >
            <Box
                sx={{
                    display: "grid",
                    gridTemplateColumns: `88px repeat(7, minmax(${dayColumnWidth}px, 1fr))`,
                }}
            >
                <Box
                    sx={(theme) => ({
                        borderRight: `1px solid ${theme.palette.divider}`,
                        borderBottom: `1px solid ${theme.palette.divider}`,
                        backgroundColor: theme.palette.background.default,
                    })}
                />
                {weekDays.map((day) => {
                    const isToday = day.isSame(dayjs(), "day");
                    return (
                        <Box
                            key={day.format("YYYY-MM-DD")}
                            sx={(theme) => ({
                                minHeight: 88,
                                px: 2,
                                py: 1.5,
                                borderRight: `1px solid ${theme.palette.divider}`,
                                borderBottom: `1px solid ${theme.palette.divider}`,
                                backgroundColor: theme.palette.background.default,
                                display: "flex",
                                alignItems: "flex-start",
                                justifyContent: "center",
                            })}
                        >
                            <Stack direction="row" spacing={1} alignItems="center">
                                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                    {day.format("ddd D")}
                                </Typography>
                                {isToday && (
                                    <Box
                                        sx={(theme) => ({
                                            width: 32,
                                            height: 32,
                                            borderRadius: "999px",
                                            display: "grid",
                                            placeItems: "center",
                                            backgroundColor: theme.palette.primary.main,
                                            color: theme.palette.primary.contrastText,
                                            fontSize: 14,
                                            fontWeight: 700,
                                        })}
                                    >
                                        {day.format("D")}
                                    </Box>
                                )}
                            </Stack>
                        </Box>
                    );
                })}

                <Box
                    sx={(theme) => ({
                        minHeight: 64,
                        px: 1.5,
                        py: 1,
                        borderRight: `1px solid ${theme.palette.divider}`,
                        borderBottom: `1px solid ${theme.palette.divider}`,
                        backgroundColor: theme.palette.background.default,
                    })}
                >
                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 700 }}>
                        All day
                    </Typography>
                </Box>
                {weekDays.map((day, dayIndex) => {
                    const allDayItems = allDayWeekItems[dayIndex];
                    return (
                        <Box
                            key={`${day.format("YYYY-MM-DD")}-all-day`}
                            sx={(theme) => ({
                                minHeight: 64,
                                p: 1,
                                borderRight: `1px solid ${theme.palette.divider}`,
                                borderBottom: `1px solid ${theme.palette.divider}`,
                                backgroundColor: theme.palette.background.paper,
                            })}
                        >
                            <Stack spacing={0.75}>
                                {allDayItems.slice(0, 2).map((item) => {
                                    const colors = getWeekItemColor(item, theme);
                                    return (
                                        <Box
                                            key={item.id}
                                            onClick={() => openDay(day)}
                                            sx={{
                                                px: 1.25,
                                                py: 0.75,
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
                                {allDayItems.length > 2 && (
                                    <Typography variant="caption" color="text.secondary">
                                        +{allDayItems.length - 2} more
                                    </Typography>
                                )}
                            </Stack>
                        </Box>
                    );
                })}

                <Box
                    sx={(theme) => ({
                        position: "relative",
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
                                pt: 0.8,
                                borderBottom: `1px solid ${theme.palette.divider}`,
                            })}
                        >
                            <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
                                {dayjs().hour(hour).minute(0).format("h A")}
                            </Typography>
                        </Box>
                    ))}
                </Box>
                {weekDays.map((day, dayIndex) => (
                    <Box
                        key={`${day.format("YYYY-MM-DD")}-grid`}
                        sx={(theme) => ({
                            position: "relative",
                            height: rowHeight * hourSlots.length,
                            borderRight: `1px solid ${theme.palette.divider}`,
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
                        {timedWeekItems[dayIndex].map((item) => {
                            const colors = getWeekItemColor(item, theme);
                            return (
                                <Box
                                    key={item.id}
                                    onClick={() => openDay(day)}
                                    sx={{
                                        position: "absolute",
                                        top: item.top,
                                        left: 8,
                                        right: 8,
                                        height: item.height,
                                        px: 1.25,
                                        py: 1,
                                        borderRadius: 2,
                                        border: `1px solid ${colors.border}`,
                                        backgroundColor: colors.bg,
                                        color: colors.text,
                                        cursor: "pointer",
                                        overflow: "hidden",
                                    }}
                                >
                                    <Typography variant="body2" sx={{ fontWeight: 700, lineHeight: 1.3 }}>
                                        {item.title}
                                    </Typography>
                                    <Typography variant="body2" sx={{ mt: 0.4 }}>
                                        {formatTimeValue(item.start_time ?? "09:00")}
                                    </Typography>
                                </Box>
                            );
                        })}
                    </Box>
                ))}
            </Box>
            </Box>
        </Box>
    );
}
