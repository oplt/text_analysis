import { Checkbox, FormControlLabel, MenuItem, Stack, TextField } from "@mui/material";

export type DfmWeighting =
    | "count"
    | "binary"
    | "tf"
    | "tfidf"
    | "sublinear_tf"
    | "log_count"
    | "bm25";

export type AdvancedDfmConfig = {
    weighting: DfmWeighting;
    k1: number;
    b: number;
    smoothIdf: boolean;
    forceSparseOnly: boolean;
    minTermFrequency: string;
    maxTermFrequency: string;
    termFrequencyType: "count" | "prop" | "rank" | "quantile";
    minDocumentFrequency: string;
    maxDocumentFrequency: string;
    documentFrequencyType: "count" | "prop" | "rank" | "quantile";
    topN: string;
};

export const DEFAULT_DFM_CONFIG: AdvancedDfmConfig = {
    weighting: "count",
    k1: 1.5,
    b: 0.75,
    smoothIdf: true,
    forceSparseOnly: false,
    minTermFrequency: "",
    maxTermFrequency: "",
    termFrequencyType: "count",
    minDocumentFrequency: "",
    maxDocumentFrequency: "",
    documentFrequencyType: "count",
    topN: "",
};

const FREQUENCY_TYPES = ["count", "prop", "rank", "quantile"] as const;

export function AdvancedDfmPanel({
    config,
    onChange,
}: {
    config: AdvancedDfmConfig;
    onChange: (config: AdvancedDfmConfig) => void;
}) {
    const update = (patch: Partial<AdvancedDfmConfig>) => onChange({ ...config, ...patch });
    const usesIdf = config.weighting === "tfidf" || config.weighting === "sublinear_tf";

    return (
        <Stack spacing={1.5}>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} flexWrap="wrap" useFlexGap>
                <TextField
                    select size="small" label="Weighting" value={config.weighting}
                    onChange={(event) => update({ weighting: event.target.value as DfmWeighting })}
                    sx={{ minWidth: 190 }}
                >
                    <MenuItem value="count">Count</MenuItem>
                    <MenuItem value="binary">Binary</MenuItem>
                    <MenuItem value="tf">Term frequency</MenuItem>
                    <MenuItem value="tfidf">TF-IDF</MenuItem>
                    <MenuItem value="sublinear_tf">Sublinear TF-IDF</MenuItem>
                    <MenuItem value="log_count">Log count</MenuItem>
                    <MenuItem value="bm25">BM25</MenuItem>
                </TextField>
                <FormControlLabel
                    control={<Checkbox checked={config.forceSparseOnly} onChange={(event) => update({ forceSparseOnly: event.target.checked })} />}
                    label="Sparse-only output"
                />
                {usesIdf ? <FormControlLabel
                    control={<Checkbox checked={config.smoothIdf} onChange={(event) => update({ smoothIdf: event.target.checked })} />}
                    label="Smooth IDF"
                /> : null}
            </Stack>

            {config.weighting === "bm25" ? (
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                    <TextField size="small" type="number" label="BM25 k1" value={config.k1} onChange={(event) => update({ k1: Math.max(0.01, Number(event.target.value) || 0.01) })} inputProps={{ min: 0.01, step: 0.1 }} />
                    <TextField size="small" type="number" label="BM25 b" value={config.b} onChange={(event) => update({ b: Math.max(0, Math.min(1, Number(event.target.value))) })} inputProps={{ min: 0, max: 1, step: 0.05 }} />
                </Stack>
            ) : null}

            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} flexWrap="wrap" useFlexGap>
                <TextField size="small" type="number" label="Min term frequency" value={config.minTermFrequency} onChange={(event) => update({ minTermFrequency: event.target.value })} />
                <TextField size="small" type="number" label="Max term frequency" value={config.maxTermFrequency} onChange={(event) => update({ maxTermFrequency: event.target.value })} />
                <TextField select size="small" label="Term frequency type" value={config.termFrequencyType} onChange={(event) => update({ termFrequencyType: event.target.value as AdvancedDfmConfig["termFrequencyType"] })}>
                    {FREQUENCY_TYPES.map((type) => <MenuItem key={type} value={type}>{type}</MenuItem>)}
                </TextField>
            </Stack>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} flexWrap="wrap" useFlexGap>
                <TextField size="small" type="number" label="Min document frequency" value={config.minDocumentFrequency} onChange={(event) => update({ minDocumentFrequency: event.target.value })} />
                <TextField size="small" type="number" label="Max document frequency" value={config.maxDocumentFrequency} onChange={(event) => update({ maxDocumentFrequency: event.target.value })} />
                <TextField select size="small" label="Document frequency type" value={config.documentFrequencyType} onChange={(event) => update({ documentFrequencyType: event.target.value as AdvancedDfmConfig["documentFrequencyType"] })}>
                    {FREQUENCY_TYPES.map((type) => <MenuItem key={type} value={type}>{type}</MenuItem>)}
                </TextField>
                <TextField size="small" type="number" label="Keep top features" value={config.topN} onChange={(event) => update({ topN: event.target.value })} inputProps={{ min: 1 }} helperText="Optional top-N trim" />
            </Stack>
        </Stack>
    );
}
