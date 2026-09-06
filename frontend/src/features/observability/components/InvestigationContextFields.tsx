import { Box, MenuItem, TextField } from "@mui/material";
import { SectionCard } from "../../../components/ui/SectionCard";
import { ENVIRONMENTS, SERVICES, TIME_RANGES } from "../observabilityModel";
import type { ObservabilityViewModel } from "../hooks/useObservabilityView";

type Props = Pick<ObservabilityViewModel, "filters" | "setters">;

export function InvestigationContextFields({ filters, setters }: Props) {
    return (
        <SectionCard
            title="Investigation context"
            description="These fields are passed as safe Grafana query parameters when a dashboard supports them."
        >
            <Box sx={{
                display: "grid",
                gap: 2,
                gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))", xl: "repeat(3, minmax(0, 1fr))" },
            }}>
                <TextField select label="Service" value={filters.service} onChange={(event) => setters.setService(event.target.value)}>
                    {SERVICES.map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                </TextField>
                <TextField select label="Environment" value={filters.environment} onChange={(event) => setters.setEnvironment(event.target.value)}>
                    {ENVIRONMENTS.map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                </TextField>
                <TextField select label="Time range" value={filters.timeRangeIndex} onChange={(event) => setters.setTimeRangeIndex(Number(event.target.value))}>
                    {TIME_RANGES.map((item, index) => <MenuItem key={item.from} value={index}>{item.label}</MenuItem>)}
                </TextField>
                <TextField label="Route" value={filters.route} onChange={(event) => setters.setRoute(event.target.value)} placeholder="/api/v1/users" />
                <TextField label="Job" value={filters.jobName} onChange={(event) => setters.setJobName(event.target.value)} placeholder="email" />
                <TextField label="Request or trace" value={filters.requestId} onChange={(event) => setters.setRequestId(event.target.value)} placeholder="request id" />
                <TextField label="Trace ID" value={filters.traceId} onChange={(event) => setters.setTraceId(event.target.value)} placeholder="trace id" />
            </Box>
        </SectionCard>
    );
}
