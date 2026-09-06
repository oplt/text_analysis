import {
    Box,
    FormControl,
    InputLabel,
    MenuItem,
    Select,
    Stack,
    Typography,
} from "@mui/material";
import { useResearchContext } from "../hooks/useResearchContext";
import { UNIT_TYPE_OPTIONS } from "../types";

export function CorpusSelector() {
    const ctx = useResearchContext();
    return (
        <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel id="research-corpus-label">Corpus</InputLabel>
            <Select
                labelId="research-corpus-label"
                label="Corpus"
                value={ctx.selectedCorpusId || ""}
                onChange={(e) => ctx.setSelectedCorpusId(e.target.value)}
                disabled={ctx.corpora.length === 0}
            >
                {ctx.corpora.map((corpus) => (
                    <MenuItem key={corpus.id} value={corpus.id}>
                        {corpus.name}
                    </MenuItem>
                ))}
            </Select>
        </FormControl>
    );
}

export function CodebookSelector() {
    const ctx = useResearchContext();
    return (
        <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel id="research-codebook-label">Codebook</InputLabel>
            <Select
                labelId="research-codebook-label"
                label="Codebook"
                value={ctx.selectedCodebookId || ""}
                onChange={(e) => ctx.setSelectedCodebookId(e.target.value)}
                disabled={ctx.codebooks.length === 0}
            >
                {ctx.codebooks.map((codebook) => (
                    <MenuItem key={codebook.id} value={codebook.id}>
                        {codebook.name} (v{codebook.version})
                    </MenuItem>
                ))}
            </Select>
        </FormControl>
    );
}

export function UnitTypeSelector() {
    const ctx = useResearchContext();
    return (
        <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel id="research-unit-type-label">Unit type</InputLabel>
            <Select
                labelId="research-unit-type-label"
                label="Unit type"
                value={ctx.unitType}
                onChange={(e) => ctx.setUnitType(e.target.value as typeof ctx.unitType)}
            >
                {UNIT_TYPE_OPTIONS.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                        {option.label}
                    </MenuItem>
                ))}
            </Select>
        </FormControl>
    );
}

export function ResearchContextBar() {
    const ctx = useResearchContext();
    return (
        <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={2}
            alignItems={{ xs: "stretch", md: "center" }}
            flexWrap="wrap"
        >
            <CorpusSelector />
            <CodebookSelector />
            <UnitTypeSelector />
            {ctx.corpora.length === 0 && (
                <Typography variant="body2" color="text.secondary">
                    No corpora yet — create one or seed the demo corpus.
                </Typography>
            )}
        </Stack>
    );
}

export function JsonBlock({ data }: { data: unknown }) {
    return (
        <Box
            component="pre"
            sx={{
                m: 0,
                p: 2,
                borderRadius: 2,
                overflow: "auto",
                maxHeight: 420,
                fontSize: 12,
                bgcolor: "action.hover",
            }}
        >
            {JSON.stringify(data, null, 2)}
        </Box>
    );
}

export function RunStatusChip({ status }: { status: string }) {
    const color =
        status === "completed"
            ? "success"
            : status === "failed"
              ? "error"
              : status === "running"
                ? "warning"
                : "default";
    return (
        <Typography component="span" variant="caption" color={`${color}.main`} sx={{ fontWeight: 600 }}>
            {status}
        </Typography>
    );
}
