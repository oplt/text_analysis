import { Box, Button, Chip, FormControlLabel, MenuItem, Skeleton, Stack, Switch, TextField, Typography } from "@mui/material";
import { QueryErrorAlert } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import type { AiStudioModel } from "../hooks/useAiStudioView";
import { firstSchemaError, promptVersionDraftSchema } from "../studioUtils";

export function VersionBuilderPanel({ m }: { m: AiStudioModel }) {
    const { selectedTemplateId, setSelectedTemplateId, promptTemplates, versionForm, setVersionForm, providers, createVersionMutation, parseVariableDefinitions, versionsIsError, versionsError, refetchVersions, versionsLoading, selectedTemplateVersions, activateVersionMutation, publishVersionMutation } = m;
    return (
    <SectionCard title="Version builder" description="Attach deployable versions to a prompt template with model selection, rollout state, and provider pricing.">
        <Stack spacing={2}>
            <TextField
                select
                label="Selected template"
                value={selectedTemplateId}
                onChange={(event) => setSelectedTemplateId(event.target.value)}
                fullWidth
            >
                {promptTemplates.map((template) => (
                    <MenuItem key={template.id} value={template.id}>
                        {template.name}
                    </MenuItem>
                ))}
            </TextField>
            <TextField
                select
                label="Provider"
                value={versionForm.provider_key}
                onChange={(event) => setVersionForm((current) => ({ ...current, provider_key: event.target.value }))}
                fullWidth
            >
                {providers.map((provider) => (
                    <MenuItem key={provider.key} value={provider.key}>
                        {provider.label}
                    </MenuItem>
                ))}
            </TextField>
            <TextField label="Model name" value={versionForm.model_name} onChange={(event) => setVersionForm((current) => ({ ...current, model_name: event.target.value }))} fullWidth />
            <TextField label="System prompt" value={versionForm.system_prompt} onChange={(event) => setVersionForm((current) => ({ ...current, system_prompt: event.target.value }))} fullWidth multiline minRows={3} />
            <TextField label="User prompt template" value={versionForm.user_prompt_template} onChange={(event) => setVersionForm((current) => ({ ...current, user_prompt_template: event.target.value }))} fullWidth multiline minRows={5} />
            <TextField label="Variable names (comma separated)" value={versionForm.variable_names} onChange={(event) => setVersionForm((current) => ({ ...current, variable_names: event.target.value }))} fullWidth />
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                <TextField select label="Response format" value={versionForm.response_format} onChange={(event) => setVersionForm((current) => ({ ...current, response_format: event.target.value as "text" | "json" }))} fullWidth>
                    <MenuItem value="text">Text</MenuItem>
                    <MenuItem value="json">JSON</MenuItem>
                </TextField>
                <TextField label="Temperature" value={versionForm.temperature} onChange={(event) => setVersionForm((current) => ({ ...current, temperature: event.target.value }))} fullWidth />
            </Stack>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                <TextField label="Rollout %" value={versionForm.rollout_percentage} onChange={(event) => setVersionForm((current) => ({ ...current, rollout_percentage: event.target.value }))} fullWidth />
                <FormControlLabel control={<Switch checked={versionForm.is_published} onChange={(event) => setVersionForm((current) => ({ ...current, is_published: event.target.checked }))} />} label="Published" />
            </Stack>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                <TextField label="Input cost / million" value={versionForm.input_cost_per_million} onChange={(event) => setVersionForm((current) => ({ ...current, input_cost_per_million: event.target.value }))} fullWidth />
                <TextField label="Output cost / million" value={versionForm.output_cost_per_million} onChange={(event) => setVersionForm((current) => ({ ...current, output_cost_per_million: event.target.value }))} fullWidth />
            </Stack>
            <Button
                variant="contained"
                disabled={createVersionMutation.isPending || !selectedTemplateId || !versionForm.model_name.trim() || !versionForm.user_prompt_template.trim()}
                onClick={() => {
                    const result = promptVersionDraftSchema.safeParse(versionForm);
                    if (!result.success) {
                        m.showToast({ message: firstSchemaError(result), severity: "error" });
                        return;
                    }
                    createVersionMutation.mutate({
                        templateId: selectedTemplateId,
                        payload: {
                            provider_key: result.data.provider_key,
                            model_name: result.data.model_name,
                            system_prompt: result.data.system_prompt,
                            user_prompt_template: result.data.user_prompt_template,
                            variable_definitions: parseVariableDefinitions(result.data.variable_names),
                            response_format: result.data.response_format,
                            temperature: Number(result.data.temperature),
                            rollout_percentage: Number(result.data.rollout_percentage),
                            is_published: result.data.is_published,
                            input_cost_per_million: Number(result.data.input_cost_per_million),
                            output_cost_per_million: Number(result.data.output_cost_per_million),
                        },
                    });
                }}
            >
                {createVersionMutation.isPending ? "Saving..." : "Create prompt version"}
            </Button>
            {selectedTemplateId && versionsIsError && (
                <QueryErrorAlert
                    error={versionsError}
                    fallback="Failed to load prompt versions."
                    onRetry={() => void refetchVersions()}
                />
            )}
            {selectedTemplateId && versionsLoading ? (
                <Stack spacing={1.25}>
                    {Array.from({ length: 2 }).map((_, index) => (
                        <Skeleton key={index} variant="rounded" height={96} sx={{ borderRadius: 3 }} />
                    ))}
                </Stack>
            ) : selectedTemplateVersions.length > 0 ? (
                <Stack spacing={1.25}>
                    {selectedTemplateVersions.map((version) => (
                        <Box key={version.id} sx={(theme) => ({ p: 2, borderRadius: 4, border: `1px solid ${theme.palette.divider}` })}>
                            <Stack spacing={1}>
                                <Stack direction={{ xs: "column", sm: "row" }} justifyContent="space-between" spacing={1}>
                                    <Typography variant="subtitle2">
                                        v{version.version_number} • {version.provider_key}/{version.model_name}
                                    </Typography>
                                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                        {version.is_published && <Chip label="published" size="small" color="success" variant="outlined" />}
                                        <Chip label={`${version.rollout_percentage}% rollout`} size="small" variant="outlined" />
                                    </Stack>
                                </Stack>
                                <Typography variant="body2" color="text.secondary">
                                    {version.user_prompt_template.slice(0, 180)}
                                </Typography>
                                <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                                    <Button size="small" variant="outlined" onClick={() => activateVersionMutation.mutate({ templateId: selectedTemplateId, versionId: version.id })}>
                                        Set active
                                    </Button>
                                    <Button
                                        size="small"
                                        variant="outlined"
                                        onClick={() => publishVersionMutation.mutate({ templateId: selectedTemplateId, versionId: version.id, isPublished: !version.is_published })}
                                    >
                                        {version.is_published ? "Unpublish" : "Publish"}
                                    </Button>
                                </Stack>
                            </Stack>
                        </Box>
                    ))}
                </Stack>
            ) : null}
        </Stack>
    </SectionCard>
    );
}
