import { Checkbox, FormControlLabel, MenuItem, Stack, TextField } from "@mui/material";
import { FormGrid } from "../../../components/ui/FormGrid";
import { HelpFieldLabel } from "../../../components/ui/HelpTooltip";
import type { AdvancedDfmConfig, DfmWeighting } from "./advancedDfmConfig";

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
            <FormGrid columns="4-4-4">
                <TextField
                    select
                    size="small"
                    label={<HelpFieldLabel termId="tfidf">Weighting</HelpFieldLabel>}
                    value={config.weighting}
                    onChange={(event) => update({ weighting: event.target.value as DfmWeighting })}
                    fullWidth
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
                    control={
                        <Checkbox
                            checked={config.forceSparseOnly}
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
                ) : (
                    <span />
                )}
            </FormGrid>

            {config.weighting === "bm25" ? (
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

            <FormGrid columns="4-4-4">
                <TextField
                    size="small"
                    type="number"
                    label="Min term frequency"
                    value={config.minTermFrequency}
                    onChange={(event) => update({ minTermFrequency: event.target.value })}
                    fullWidth
                />
                <TextField
                    size="small"
                    type="number"
                    label="Max term frequency"
                    value={config.maxTermFrequency}
                    onChange={(event) => update({ maxTermFrequency: event.target.value })}
                    fullWidth
                />
                <TextField
                    select
                    size="small"
                    label="Term frequency type"
                    value={config.termFrequencyType}
                    onChange={(event) =>
                        update({
                            termFrequencyType:
                                event.target.value as AdvancedDfmConfig["termFrequencyType"],
                        })
                    }
                    fullWidth
                >
                    {FREQUENCY_TYPES.map((type) => (
                        <MenuItem key={type} value={type}>
                            {type}
                        </MenuItem>
                    ))}
                </TextField>
            </FormGrid>
            <FormGrid columns="4-4-4">
                <TextField
                    size="small"
                    type="number"
                    label={<HelpFieldLabel termId="min_df">Min document frequency</HelpFieldLabel>}
                    value={config.minDocumentFrequency}
                    onChange={(event) => update({ minDocumentFrequency: event.target.value })}
                    fullWidth
                />
                <TextField
                    size="small"
                    type="number"
                    label={<HelpFieldLabel termId="max_df">Max document frequency</HelpFieldLabel>}
                    value={config.maxDocumentFrequency}
                    onChange={(event) => update({ maxDocumentFrequency: event.target.value })}
                    fullWidth
                />
                <TextField
                    select
                    size="small"
                    label="Document frequency type"
                    value={config.documentFrequencyType}
                    onChange={(event) =>
                        update({
                            documentFrequencyType:
                                event.target.value as AdvancedDfmConfig["documentFrequencyType"],
                        })
                    }
                    fullWidth
                >
                    {FREQUENCY_TYPES.map((type) => (
                        <MenuItem key={type} value={type}>
                            {type}
                        </MenuItem>
                    ))}
                </TextField>
            </FormGrid>
        </Stack>
    );
}
