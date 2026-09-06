import { Box, Chip, MenuItem, Paper, Stack, TextField, Typography } from "@mui/material";
import { CalendarMonth as CalendarIcon, DragIndicator as DragIcon } from "@mui/icons-material";
import { alpha } from "@mui/material/styles";
import type { ProjectTask, ProjectTaskStatus } from "../../../api/projects";
import { formatDateOnly, formatDateTime, humanizeKey } from "../../../utils/formatters";
import { TASK_STATUS_OPTIONS } from "../projectTaskModel";

function Meta({ task }: { task: ProjectTask }) { return <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap><Chip label={humanizeKey(task.status)} size="small" variant="outlined" /><Chip label={humanizeKey(task.priority)} size="small" variant="outlined" />{task.assignee && <Chip label={task.assignee.full_name || task.assignee.email} size="small" variant="outlined" />}{task.due_date && <Chip icon={<CalendarIcon fontSize="small" />} label={formatDateOnly(task.due_date)} size="small" variant="outlined" color={task.status !== "done" && new Date(`${task.due_date}T23:59:59`) < new Date() ? "warning" : "default"} />}</Stack>; }

type BoardProps = { task: ProjectTask; selected: boolean; isDragging: boolean; onSelect: () => void; onDragStart: () => void; onDropBefore: () => void; onMove: (status: ProjectTaskStatus) => void };
export function TaskBoardCard({ task, selected, isDragging, onSelect, onDragStart, onDropBefore, onMove }: BoardProps) {
    return <Paper draggable aria-grabbed={isDragging} onClick={onSelect} onDragStart={(e) => { e.dataTransfer.setData("text/plain", task.id); e.dataTransfer.effectAllowed = "move"; onDragStart(); }} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); e.stopPropagation(); onDropBefore(); }} sx={(t) => ({ p: 1.75, borderRadius: 3, cursor: "grab", border: `1px solid ${selected ? t.palette.primary.main : t.palette.divider}`, backgroundColor: selected ? alpha(t.palette.primary.main, t.palette.mode === "dark" ? 0.18 : 0.08) : t.palette.background.paper })}>
        <Stack spacing={1.25}><Stack direction="row" justifyContent="space-between" spacing={1}><Typography variant="subtitle2">{task.title}</Typography><DragIcon fontSize="small" color="action" /></Stack><Typography variant="body2" color="text.secondary">{task.description || "No task notes yet."}</Typography><Meta task={task} />
            <TextField select size="small" label="Move to" value="" onClick={(e) => e.stopPropagation()} onChange={(e) => { e.stopPropagation(); onMove(e.target.value as ProjectTaskStatus); }} fullWidth SelectProps={{ displayEmpty: true, renderValue: () => "Choose column" }} inputProps={{ "aria-label": `Move ${task.title} to another column` }}>{TASK_STATUS_OPTIONS.filter((o) => o.value !== task.status).map((o) => <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>)}</TextField>
        </Stack></Paper>;
}
export function TaskListCard({ task, selected, onSelect }: { task: ProjectTask; selected: boolean; onSelect: () => void }) {
    return <Paper onClick={onSelect} sx={(t) => ({ p: 2, borderRadius: 4, cursor: "pointer", border: `1px solid ${selected ? t.palette.primary.main : t.palette.divider}`, backgroundColor: selected ? alpha(t.palette.primary.main, t.palette.mode === "dark" ? 0.18 : 0.06) : t.palette.background.paper })}><Stack spacing={1.25}><Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" spacing={1}><Box><Typography variant="subtitle1">{task.title}</Typography><Typography variant="body2" color="text.secondary">{task.description || "No task notes yet."}</Typography></Box><Typography variant="caption" color="text.secondary">Updated {formatDateTime(task.updated_at)}</Typography></Stack><Meta task={task} /></Stack></Paper>;
}
