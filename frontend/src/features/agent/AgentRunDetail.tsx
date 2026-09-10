import { Alert, Box, Chip, Stack, Typography } from "@mui/material";
import { formatAgentCostMicros, type AgentRun } from "../../api/agent";
import { SectionCard } from "../../components/ui/SectionCard";

type AgentRunDetailProps = {
    run: AgentRun;
};

export function AgentRunDetail({ run }: AgentRunDetailProps) {
    return (
        <SectionCard
            title="Run detail"
            description="Response, retrieval/memory degradation, provider usage, and identifiers."
        >
            <Stack spacing={1.5}>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <Chip size="small" label={`run ${run.id}`} />
                    <Chip size="small" variant="outlined" label={`memory ${run.memory_run_id}`} />
                    <Chip size="small" color={run.status === "completed" ? "success" : "default"} label={run.status} />
                    <Chip size="small" variant="outlined" label={`${run.provider_key}/${run.model_name}`} />
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
                        Response
                    </Typography>
                    <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
                        {run.output_text ||
                            (run.output_json ? JSON.stringify(run.output_json, null, 2) : "—")}
                    </Typography>
                </Box>

                <Typography variant="body2" color="text.secondary">
                    Latency: {run.latency_ms ?? "—"} ms · Tokens: {run.total_tokens} (in{" "}
                    {run.input_tokens} / out {run.output_tokens}) · Est. cost:{" "}
                    {formatAgentCostMicros(run.estimated_cost_micros)}
                </Typography>

                <Box>
                    <Typography variant="subtitle2" gutterBottom>
                        Retrieved chunks
                    </Typography>
                    {run.retrieved_chunk_ids.length === 0 ? (
                        <Typography variant="body2" color="text.secondary">
                            None
                        </Typography>
                    ) : (
                        <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                            {run.retrieved_chunk_ids.map((chunkId) => (
                                <Chip key={chunkId} size="small" variant="outlined" label={chunkId.slice(0, 12)} />
                            ))}
                        </Stack>
                    )}
                    {run.injection_chunks_filtered > 0 ? (
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                            Filtered {run.injection_chunks_filtered} injection chunk(s)
                        </Typography>
                    ) : null}
                </Box>

                {run.retrieval_query ? (
                    <Typography variant="caption" color="text.secondary">
                        Retrieval query: {run.retrieval_query}
                    </Typography>
                ) : null}
            </Stack>
        </SectionCard>
    );
}
