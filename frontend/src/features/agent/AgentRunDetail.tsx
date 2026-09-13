import { Alert, Box, Chip, Stack, Typography } from "@mui/material";
import { formatAgentCostMicros, type AgentRun } from "../../api/agent";
import { AdvancedSettings } from "../../components/ui/AdvancedSettings";
import { JsonBlock } from "../../components/ui/JsonBlock";
import { KeyValueList } from "../../components/ui/KeyValueList";
import { SectionCard } from "../../components/ui/SectionCard";
import { recordToKeyValueItems } from "../../components/ui/jsonDisplay";
import { buildAgentTraceSteps, formatElapsed } from "./agentTraceModel";

type AgentRunDetailProps = {
    run: AgentRun;
};

/** Compact run summary used outside the tabbed Agent workspace. */
export function AgentRunDetail({ run }: AgentRunDetailProps) {
    const steps = buildAgentTraceSteps(run).slice(0, 4);
    const outputItems = recordToKeyValueItems(run.output_json);
    return (
        <SectionCard
            title="Run detail"
            description="Structured summary — prefer the Agent Trace / Sources / Output tabs for full inspection."
        >
            <Stack spacing={1.5}>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <Chip size="small" label={`run ${run.id}`} />
                    <Chip
                        size="small"
                        color={run.status === "completed" ? "success" : "default"}
                        label={run.status}
                    />
                    <Chip
                        size="small"
                        variant="outlined"
                        label={`${run.provider_key}/${run.model_name}`}
                    />
                </Stack>

                {(run.retrieval_degraded || run.memory_degraded || run.error_message) && (
                    <Alert severity={run.error_message ? "error" : "warning"}>
                        {run.error_message ??
                            run.degradation_reason ??
                            (run.retrieval_degraded
                                ? "Retrieval degraded"
                                : "Memory degraded")}
                    </Alert>
                )}

                <Box>
                    <Typography variant="subtitle2" gutterBottom>
                        Trace preview
                    </Typography>
                    {steps.map((step) => (
                        <Typography key={step.id} variant="body2" color="text.secondary">
                            {step.title}
                            {step.durationMs != null ? ` · ${formatElapsed(step.durationMs)}` : ""}
                        </Typography>
                    ))}
                </Box>

                <Box>
                    <Typography variant="subtitle2" gutterBottom>
                        Output
                    </Typography>
                    {run.output_text?.trim() ? (
                        <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
                            {run.output_text}
                        </Typography>
                    ) : outputItems.length ? (
                        <KeyValueList dense items={outputItems} />
                    ) : (
                        <Typography variant="body2" color="text.secondary">
                            —
                        </Typography>
                    )}
                    {run.output_json ? (
                        <AdvancedSettings title="Raw JSON" sx={{ mt: 1 }}>
                            <JsonBlock data={run.output_json} />
                        </AdvancedSettings>
                    ) : null}
                </Box>

                <Typography variant="body2" color="text.secondary">
                    Latency: {run.latency_ms ?? "—"} ms · Tokens: {run.total_tokens} · Est. cost:{" "}
                    {formatAgentCostMicros(run.estimated_cost_micros)}
                </Typography>
            </Stack>
        </SectionCard>
    );
}
