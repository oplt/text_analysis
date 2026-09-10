import { Checkbox, FormControlLabel, MenuItem, Stack, TextField, Typography } from "@mui/material";
import type { AdvancedDfmConfig, DfmWeighting } from "./advancedDfmConfig";

const FREQUENCY_TYPES = ["count", "prop", "rank", "quantile"] as const;

export function AdvancedDfmPanel({
    config,
    onChange,
    rEngine = false,
}: {
    config: AdvancedDfmConfig;
    onChange: (config: AdvancedDfmConfig) => void;
    /** R / quanteda currently supports count weighting only (no trim/TF-IDF/BM25). */
    rEngine?: boolean;
}) {
    const update = (patch: Partial<AdvancedDfmConfig>) => onChange({ ...config, ...patch });
    const usesIdf = !rEngine && (config.weighting === "tfidf" || config.weighting === "sublinear_tf");

    return (
        <Stack spacing={1.5}>
            {rEngine ? (
                <Typography variant="body2" color="text.secondary">
                    R / quanteda DFM currently supports count weighting only. Advanced trim and
                    TF-IDF/BM25 options are disabled.
                </Typography>
            ) : null}
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} flexWrap="wrap" useFlexGap>
                <TextField
                    select
                    size="small"
                    label="Weighting"
                    value={rEngine ? "count" : config.weighting}
                    disabled={rEngine}
                    onChange={(event) => update({ weighting: event.target.value as DfmWeighting })}
                    sx={{ minWidth: 190 }}
                >
                    <MenuItem value="count">Count</MenuItem>
                    <MenuItem value="binary" disabled={rEngine}>
                        Binary
                    </MenuItem>
                    <MenuItem value="tf" disabled={rEngine}>
                        Term frequency
                    </MenuItem>
                    <MenuItem value="tfidf" disabled={rEngine}>
                        TF-IDF
                    </MenuItem>
                    <MenuItem value="sublinear_tf" disabled={rEngine}>
                        Sublinear TF-IDF
                    </MenuItem>
                    <MenuItem value="log_count" disabled={rEngine}>
                        Log count
                    </MenuItem>
                    <MenuItem value="bm25" disabled={rEngine}>
                        BM25
                    </MenuItem>
                </TextField>
                <FormControlLabel
                    control={
                        <Checkbox
                            checked={rEngine ? false : config.forceSparseOnly}
                            disabled={rEngine}
                            onChange={(event) => update({ forceSparseOnly: event.target.checked })}
                        />
                    }
                    label="Sparse-only output"
                />
                {usesIdf ? (
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={config.smoothIdf}
                                onChange={(event) => update({ smoothIdf: event.target.checked })}
                            />
                        }
                        label="Smooth IDF"
                    />
                ) : null}
            </Stack>

            {!rEngine && config.weighting === "bm25" ? (
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                    <TextField
                        size="small"
                        type="number"
                        label="BM25 k1"
                        value={config.k1}
                        onChange={(event) =>
                            update({ k1: Math.max(0.01, Number(event.target.value) || 0.01) })
                        }
                        inputProps={{ min: 0.01, step: 0.1 }}
                    />
                    <TextField
                        size="small"
                        type="number"
                        label="BM25 b"
                        value={config.b}
                        onChange={(event) =>
                            update({
                                b: Math.max(0, Math.min(1, Number(event.target.value))),
                            })
                        }
                        inputProps={{ min: 0, max: 1, step: 0.05 }}
                    />
                </Stack>
            ) : null}

            {!rEngine ? (
                <>
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} flexWrap="wrap" useFlexGap>
                        <TextField
                            size="small"
                            type="number"
                            label="Min term frequency"
                            value={config.minTermFrequency}
                            onChange={(event) => update({ minTermFrequency: event.target.value })}
                        />
                        <TextField
                            size="small"
                            type="number"
                            label="Max term frequency"
                            value={config.maxTermFrequency}
                            onChange={(event) => update({ maxTermFrequency: event.target.value })}
                        />
                        <TextField
                            select
                            size="small"
                            label="Term frequency type"
                            value={config.termFrequencyType}
                            onChange={(event) =>
                                update({
                                    termFrequencyType: event.target
                                        .value as AdvancedDfmConfig["termFrequencyType"],
                                })
                            }
                        >
                            {FREQUENCY_TYPES.map((type) => (
                                <MenuItem key={type} value={type}>
                                    {type}
                                </MenuItem>
                            ))}
                        </TextField>
                    </Stack>
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} flexWrap="wrap" useFlexGap>
                        <TextField
                            size="small"
                            type="number"
                            label="Min document frequency"
                            value={config.minDocumentFrequency}
                            onChange={(event) => update({ minDocumentFrequency: event.target.value })}
                        />
                        <TextField
                            size="small"
                            type="number"
                            label="Max document frequency"
                            value={config.maxDocumentFrequency}
                            onChange={(event) => update({ maxDocumentFrequency: event.target.value })}
                        />
                        <TextField
                            select
                            size="small"
                            label="Document frequency type"
                            value={config.documentFrequencyType}
                            onChange={(event) =>
                                update({
                                    documentFrequencyType: event.target
                                        .value as AdvancedDfmConfig["documentFrequencyType"],
                                })
                            }
                        >
                            {FREQUENCY_TYPES.map((type) => (
                                <MenuItem key={type} value={type}>
                                    {type}
                                </MenuItem>
                            ))}
                        </TextField>
                        <TextField
                            size="small"
                            type="number"
                            label="Keep top features"
                            value={config.topN}
                            onChange={(event) => update({ topN: event.target.value })}
                            inputProps={{ min: 1 }}
                            helperText="Optional top-N trim"
                        />
                    </Stack>
                </>
            ) : null}
        </Stack>
    );
}
