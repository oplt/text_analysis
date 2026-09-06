import { apiFetch } from "./client";
import type { ProjectTaskPriority, ProjectTaskStatus } from "./projects";

export type CalendarItemType = "event" | "appointment" | "task";
export type CalendarItemSource = "planner" | "task";

export type CalendarItem = {
    id: string;
    source: CalendarItemSource;
    type: CalendarItemType;
    title: string;
    description: string | null;
    date: string;
    start_time: string | null;
    end_time: string | null;
    project_id: string | null;
    project_name: string | null;
    priority: ProjectTaskPriority | null;
    status: ProjectTaskStatus | null;
    created_at: string;
};

export async function listCalendarItems(
    startDate: string,
    endDate: string
): Promise<CalendarItem[]> {
    const params = new URLSearchParams({
        start_date: startDate,
        end_date: endDate,
    });
    return apiFetch(`/calendar/items?${params.toString()}`);
}

export async function createCalendarItem(payload: {
    type: CalendarItemType;
    title: string;
    description?: string | null;
    date: string;
    start_time?: string | null;
    end_time?: string | null;
    project_id?: string | null;
    priority?: ProjectTaskPriority | null;
    assignee_id?: string | null;
}): Promise<CalendarItem> {
    return apiFetch("/calendar/items", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}
