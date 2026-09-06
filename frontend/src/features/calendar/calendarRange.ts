import type { Dayjs } from "dayjs";

export type CalendarViewMode = "day" | "week" | "month" | "twelve_month";

export function getMonthGridColumns(viewMode: CalendarViewMode) {
    return viewMode === "twelve_month"
        ? ({ xs: "1fr", md: "repeat(2, minmax(0, 1fr))", xl: "repeat(4, minmax(0, 1fr))" } as const)
        : ({ xs: "1fr" } as const);
}

export function getCalendarQueryRange(viewMode: CalendarViewMode, anchorDate: Dayjs) {
    if (viewMode === "day") {
        const dateKey = anchorDate.startOf("day").format("YYYY-MM-DD");
        return { start: dateKey, end: dateKey };
    }
    if (viewMode === "week") {
        return {
            start: anchorDate.startOf("week").format("YYYY-MM-DD"),
            end: anchorDate.endOf("week").format("YYYY-MM-DD"),
        };
    }
    const startMonth = anchorDate.startOf("month");
    const endMonth = viewMode === "twelve_month" ? startMonth.add(11, "month") : startMonth;
    return {
        start: startMonth.startOf("week").format("YYYY-MM-DD"),
        end: endMonth.endOf("month").endOf("week").format("YYYY-MM-DD"),
    };
}

export function shiftCalendarAnchor(current: Dayjs, viewMode: CalendarViewMode, direction: 1 | -1) {
    if (viewMode === "day") return current.add(direction, "day");
    if (viewMode === "week") return current.add(direction, "week");
    if (viewMode === "twelve_month") return current.add(direction * 12, "month");
    return current.add(direction, "month");
}
