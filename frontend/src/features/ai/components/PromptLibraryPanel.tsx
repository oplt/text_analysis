import { Box, Button, Chip, Stack, TextField, Typography } from "@mui/material";
import { PsychologyAlt as PromptIcon } from "@mui/icons-material";
import { EmptyState } from "../../../components/ui/EmptyState";
import { SectionCard } from "../../../components/ui/SectionCard";
import type { AiStudioModel } from "../hooks/useAiStudioView";

export function PromptLibraryPanel({ m }: { m: AiStudioModel }) {
    const { templateForm, setTemplateForm, createTemplateMutation, promptTemplates, setSelectedTemplateId, selectedTemplateId } = m;
    return (
    <SectionCard title="Prompt library" description="Create reusable prompt templates and publish versioned variants with rollout and pricing metadata.">
        <Stack spacing={2}>
            <Stack direction={{ xs: "column", md: "row" }} spacing={1.5}>
                <TextField label="Template key" value={templateForm.key} onChange={(event) => setTemplateForm((current) => ({ ...current, key: event.target.value }))} fullWidth />
                <TextField label="Name" value={templateForm.name} onChange={(event) => setTemplateForm((current) => ({ ...current, name: event.target.value }))} fullWidth />
            </Stack>
            <TextField label="Description" value={templateForm.description} onChange={(event) => setTemplateForm((current) => ({ ...current, description: event.target.value }))} fullWidth multiline minRows={2} />
            <Button
                variant="contained"
                onClick={() => createTemplateMutation.mutate(templateForm)}
                disabled={createTemplateMutation.isPending || !templateForm.key.trim() || !templateForm.name.trim()}
            >
                {createTemplateMutation.isPending ? "Creating..." : "Create prompt template"}
            </Button>
            {promptTemplates.length > 0 ? (
                <Stack spacing={1.25}>
                    {promptTemplates.map((template) => (
                        <Box key={template.id} sx={(theme) => ({ p: 2, borderRadius: 4, border: `1px solid ${theme.palette.divider}` })}>
                            <Stack spacing={1}>
                                <Stack direction={{ xs: "column", sm: "row" }} justifyContent="space-between" spacing={1}>
                                    <Box>
                                        <Typography variant="subtitle1">{template.name}</Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            {template.key}
                                        </Typography>
                                    </Box>
                                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                        {template.is_active && <Chip label="active" size="small" color="success" variant="outlined" />}
                                        {template.active_version_id && <Chip label="version pinned" size="small" variant="outlined" />}
                                    </Stack>
                                </Stack>
                                {template.description && (
                                    <Typography variant="body2" color="text.secondary">
                                        {template.description}
                                    </Typography>
                                )}
                                <Button variant="outlined" size="small" onClick={() => setSelectedTemplateId(template.id)}>
                                    {selectedTemplateId === template.id ? "Selected" : "Manage versions"}
                                </Button>
                            </Stack>
                        </Box>
                    ))}
                </Stack>
            ) : (
                <EmptyState
                    icon={<PromptIcon />}
                    title="No prompts yet"
                    description="Create your first prompt template to start building reusable AI behaviors."
                />
            )}
        </Stack>
    </SectionCard>
    );
}

