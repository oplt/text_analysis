import { Box } from "@mui/material";
import { alpha } from "@mui/material/styles";
import { getMonthGridColumns } from "../calendarRange";
import type { DashboardCalendarModel } from "../hooks/useDashboardCalendar";
import { CalendarDayView } from "./CalendarDayView";
import { CalendarWeekView } from "./CalendarWeekView";
import { MonthDateCalendar } from "./MonthDateCalendar";

function CalendarMonthViews({ m }: { m: DashboardCalendarModel }) {
    return <Box sx={{ display: "grid", gap: 1.5, gridTemplateColumns: getMonthGridColumns(m.viewMode) }}>
        {m.visibleMonths.map((month) => <Box key={month.format("YYYY-MM")} sx={(theme) => ({
            borderRadius: 4,
            border: `1px solid ${theme.palette.divider}`,
            p: 1.25,
            backgroundColor: alpha(theme.palette.background.paper, theme.palette.mode === "dark" ? 0.9 : 0.78),
        })}><MonthDateCalendar m={m} date={month} /></Box>)}
    </Box>;
}

export function CalendarViews({ m }: { m: DashboardCalendarModel }) {
    if (m.viewMode === "day") return <CalendarDayView m={m} />;
    if (m.viewMode === "week") return <CalendarWeekView m={m} />;
    return <CalendarMonthViews m={m} />;
}
