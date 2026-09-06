import { Box, Button, Chip, IconButton, Stack, TextField, Typography } from "@mui/material";
import { DeleteOutline as DeleteIcon } from "@mui/icons-material";
import { alpha } from "@mui/material/styles";
import type { ConfigEntry, DatabaseSetting } from "../../../api/settings";
import { formatDateTime } from "../../../utils/formatters";

export function ConfigEntryEditor({ item, value, onChange }: { item: ConfigEntry; value: string; onChange: (value: string) => void }) {
    return (
        <Box sx={(theme) => ({ p: 2, borderRadius: 4, border: `1px solid ${theme.palette.divider}`, backgroundColor: alpha(theme.palette.background.paper, 0.78) })}>
            <Stack spacing={1.25}>
                <Stack direction={{ xs: "column", sm: "row" }} justifyContent="space-between" spacing={1}>
                    <Box>
                        <Typography variant="subtitle2">{item.key}</Typography>
                        {item.description && <Typography variant="body2" color="text.secondary">{item.description}</Typography>}
                    </Box>
                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                        <Chip label={item.value_type} size="small" variant="outlined" />
                        {item.is_custom && <Chip label="custom" size="small" variant="outlined" />}
                        {item.requires_restart && <Chip label="restart recommended" size="small" color="warning" variant="outlined" />}
                    </Stack>
                </Stack>
                <TextField type={item.is_secret ? "password" : "text"} value={value}
                    onChange={(event) => onChange(event.target.value)}
                    helperText={item.is_secret ? "Stored value is masked. Enter a new value to replace it." : undefined} fullWidth />
            </Stack>
        </Box>
    );
}

type DatabaseEditorProps = {
    item: DatabaseSetting;
    draft: { value: string; description: string };
    onDraftChange: (draft: { value: string; description: string }) => void;
    onSave: () => void;
    onDelete: () => void;
    isSaving: boolean;
    isDeleting: boolean;
};

export function DatabaseSettingEditor({ item, draft, onDraftChange, onSave, onDelete, isSaving, isDeleting }: DatabaseEditorProps) {
    return (
        <Box sx={(theme) => ({ p: 2.25, borderRadius: 4, border: `1px solid ${theme.palette.divider}` })}>
            <Stack spacing={1.5}>
                <Stack direction="row" justifyContent="space-between" spacing={1.5}>
                    <Box><Typography variant="subtitle2">{item.key}</Typography><Typography variant="caption" color="text.secondary">Updated {formatDateTime(item.updated_at)}</Typography></Box>
                    <IconButton aria-label={`Delete ${item.key}`} color="error" onClick={onDelete} disabled={isDeleting}><DeleteIcon /></IconButton>
                </Stack>
                <TextField label="Value" value={draft.value} onChange={(event) => onDraftChange({ ...draft, value: event.target.value })} fullWidth multiline minRows={3} />
                <TextField label="Description" value={draft.description} onChange={(event) => onDraftChange({ ...draft, description: event.target.value })} fullWidth multiline minRows={3} />
                <Button variant="contained" disabled={isSaving} onClick={onSave}>{isSaving ? "Saving..." : "Save setting"}</Button>
            </Stack>
        </Box>
    );
}
