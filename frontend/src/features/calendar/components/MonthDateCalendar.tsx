import type { Dayjs } from "dayjs";
import { memo, useCallback } from "react";
import { Box, Stack } from "@mui/material";
import { alpha } from "@mui/material/styles";
import { DateCalendar, PickersDay, type PickersDayProps } from "@mui/x-date-pickers";
import type {} from "@mui/x-date-pickers/AdapterDayjs";
import { getDateCalendarSx } from "../calendarDisplay";
import type { DashboardCalendarModel } from "../hooks/useDashboardCalendar";

type CalendarDayCellProps = PickersDayProps & {
    daySize: number;
    itemTypes: string[];
    openDay: (day: Dayjs) => void;
};

const CalendarDayCell = memo(function CalendarDayCell({ daySize, itemTypes, openDay, ...props }: CalendarDayCellProps) {
    const { day, outsideCurrentMonth, ...other } = props;

    return (
        <Box sx={{ position: "relative" }}>
            <PickersDay
                {...other}
                day={day}
                outsideCurrentMonth={outsideCurrentMonth}
                disableMargin
                onClick={(event) => {
                    other.onClick?.(event);
                    openDay(day);
                }}
                sx={(theme) => ({
                    width: daySize,
                    height: daySize,
                    fontWeight: 700,
                    borderRadius: "999px",
                    color: outsideCurrentMonth
                        ? theme.palette.text.disabled
                        : theme.palette.text.primary,
                    backgroundColor: "transparent",
                    border: 0,
                    "&.Mui-selected": {
                        backgroundColor: theme.palette.primary.main,
                        color: theme.palette.primary.contrastText,
                    },
                    "&.Mui-selected:hover": {
                        backgroundColor: theme.palette.primary.dark,
                    },
                    "&.MuiPickersDay-today": {
                        border: 0,
                        backgroundColor: alpha(
                            theme.palette.secondary.main,
                            theme.palette.mode === "dark" ? 0.2 : 0.1
                        ),
                    },
                    "&:hover": {
                        backgroundColor: alpha(
                            theme.palette.primary.main,
                            theme.palette.mode === "dark" ? 0.14 : 0.08
                        ),
                    },
                })}
            />
            {itemTypes.length > 0 && (
                <Stack
                    direction="row"
                    spacing={0.35}
                    justifyContent="center"
                    alignItems="center"
                    sx={{
                        position: "absolute",
                        left: "50%",
                        bottom: -2,
                        transform: "translateX(-50%)",
                        pointerEvents: "none",
                    }}
                >
                    {itemTypes.slice(0, 3).map((itemType, index) => (
                        <Box
                            key={`${itemType}-${index}`}
                            sx={(theme) => ({
                                width: 5,
                                height: 5,
                                borderRadius: "999px",
                                backgroundColor:
                                    itemType === "task"
                                        ? theme.palette.success.main
                                        : itemType === "appointment"
                                            ? theme.palette.secondary.main
                                            : theme.palette.primary.main,
                            })}
                        />
                    ))}
                </Stack>
            )}
        </Box>
    );
});

export function MonthDateCalendar({ m, date }: { m: DashboardCalendarModel; date: Dayjs }) {
    const { itemsByDate, openDay, daySize, selectedDate, setAnchorDate } = m;
    const CalendarDay = useCallback((props: PickersDayProps) => {
        const itemTypes = (itemsByDate[props.day.format("YYYY-MM-DD")] ?? []).map((item) => item.type);
        return <CalendarDayCell {...props} daySize={daySize} itemTypes={itemTypes} openDay={openDay} />;
    }, [daySize, itemsByDate, openDay]);

function renderDateCalendar(date: Dayjs) {
    return (
        <DateCalendar
            value={selectedDate.isSame(date, "month") ? selectedDate : null}
            onChange={(newValue) => {
                if (newValue) {
                    setAnchorDate(newValue.startOf("day"));
                }
            }}
            referenceDate={date}
            views={["day"]}
            showDaysOutsideCurrentMonth
            fixedWeekNumber={6}
            reduceAnimations
            slots={{ day: CalendarDay }}
            sx={getDateCalendarSx(daySize)}
        />
    );
}
    return renderDateCalendar(date);
}
