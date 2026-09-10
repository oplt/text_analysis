import dayjs from "dayjs";
import { describe, expect, it } from "vitest";

import { getCalendarQueryRange, getMonthGridColumns, shiftCalendarAnchor } from "./calendarRange";

describe("calendar range behavior", () => {
    const anchor = dayjs("2026-07-15");

    it("uses exact day range", () => {
        expect(getCalendarQueryRange("day", anchor)).toEqual({ start: "2026-07-15", end: "2026-07-15" });
    });

    it("pads month and twelve-month ranges to calendar weeks", () => {
        expect(getCalendarQueryRange("month", anchor)).toEqual({ start: "2026-06-28", end: "2026-08-01" });
        expect(getCalendarQueryRange("twelve_month", anchor)).toEqual({ start: "2026-06-28", end: "2027-07-03" });
    });

    it("shifts each view by its navigation unit", () => {
        expect(shiftCalendarAnchor(anchor, "day", 1).format("YYYY-MM-DD")).toBe("2026-07-16");
        expect(shiftCalendarAnchor(anchor, "week", -1).format("YYYY-MM-DD")).toBe("2026-07-08");
        expect(shiftCalendarAnchor(anchor, "twelve_month", 1).format("YYYY-MM-DD")).toBe("2027-07-15");
    });

    it("lays out twelve-month view as three months per row", () => {
        expect(getMonthGridColumns("twelve_month")).toBe("repeat(3, minmax(0, 1fr))");
        expect(getMonthGridColumns("month")).toEqual({ xs: "1fr" });
    });
});
