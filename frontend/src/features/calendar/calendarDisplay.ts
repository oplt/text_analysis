import { alpha, type Theme } from "@mui/material/styles";
import type { CalendarItem, CalendarItemType } from "../../api/calendar";
import type { Project, ProjectTaskPriority } from "../../api/projects";
import type { CalendarViewMode } from "./calendarRange";
export type DashboardCalendarProps = { projects: Project[]; projectsLoading: boolean; onOpenProjects: () => void; allowedViews?: CalendarViewMode[]; initialView?: CalendarViewMode };
export type CalendarDraft = { type: CalendarItemType; title: string; description: string; start_time: string; end_time: string; project_id: string; priority: ProjectTaskPriority };
export type CalendarFieldErrors = { title?: string; project_id?: string; start_time?: string; end_time?: string; general?: string };
export const VIEW_OPTIONS: Array<{ value: CalendarViewMode; label: string }> = [{ value: "day", label: "Day" }, { value: "week", label: "Week" }, { value: "month", label: "Month" }, { value: "twelve_month", label: "12M" }];
export const ITEM_TYPE_OPTIONS: Array<{ value: CalendarItemType; label: string }> = [{ value: "event", label: "Event" }, { value: "appointment", label: "Appointment" }, { value: "task", label: "Task" }];
export function buildEmptyDraft(projects: Project[], type: CalendarItemType = "event"): CalendarDraft { return { type, title: "", description: "", start_time: "", end_time: "", project_id: projects[0]?.id ?? "", priority: "medium" }; }
export function formatTimeValue(value: string) { return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date(`1970-01-01T${value}`)); }
export function formatItemTime(item: CalendarItem) { if (!item.start_time) return item.type === "task" ? "Due anytime" : "All day"; return item.end_time ? `${formatTimeValue(item.start_time)} to ${formatTimeValue(item.end_time)}` : formatTimeValue(item.start_time); }
export function getMinutesFromTimeString(value: string) { const [hours = "0", minutes = "0"] = value.split(":"); return Number(hours) * 60 + Number(minutes); }
export function getWeekItemColor(item: CalendarItem, theme: Theme) { const palette = item.type === "task" ? theme.palette.success : item.type === "appointment" ? theme.palette.secondary : theme.palette.primary; const isDark = theme.palette.mode === "dark"; return { bg: alpha(palette.main, isDark ? .22 : .12), border: alpha(palette.main, isDark ? .45 : .35), text: isDark ? palette.light : palette.dark }; }
export function getDateCalendarSx(daySize: number) { return { width: "100%", maxWidth: "none", m: 0, "& .MuiPickersCalendarHeader-root": { px: 1, mb: .5 }, "& .MuiPickersCalendarHeader-switchViewButton": { display: "none" }, "& .MuiPickersCalendarHeader-label": { fontSize: "1rem", fontWeight: 700 }, "& .MuiDayCalendar-header": { justifyContent: "space-between", px: .75 }, "& .MuiDayCalendar-weekDayLabel": { width: daySize, color: "text.secondary", fontWeight: 700 }, "& .MuiDayCalendar-weekContainer": { justifyContent: "space-between", mt: .5 } } as const; }

