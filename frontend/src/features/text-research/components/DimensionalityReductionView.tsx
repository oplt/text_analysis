import { useState } from "react";
import { Button, MenuItem, Stack, TextField } from "@mui/material";
import { PlayArrow as RunIcon } from "@mui/icons-material";
import { runDimensionalityReduction } from "../../../api/textResearch";
import { SectionCard } from "../../../components/ui/SectionCard";
import {
    PanelBody,
    useAnalysisRun,
    useRunMutation,
    type AdvancedAnalysisBasePayload,
} from "./advancedAnalysisShared";

export default function DimensionalityReductionView({
    basePayload,
}: {
    basePayload: AdvancedAnalysisBasePayload;
}) {
    const { ctx, run, runQuery, setRunId, showToast } = useAnalysisRun();
    const [method, setMethod] = useState<"svd" | "pca">("svd");
    const [components, setComponents] = useState(2);
    const mutation = useRunMutation(
        () =>
            runDimensionalityReduction(ctx.selectedCorpusId, {
                ...basePayload,
                method,
                n_components: components,
            }),
        "Dimensionality reduction",
        setRunId,
        showToast
    );
    return (
        <SectionCard
            title="Dimensionality reduction"
            description="Project unit vectors into two or three dimensions for exploratory inspection."
        >
            <PanelBody run={run} runQuery={runQuery}>
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                    <TextField
                        select
                        size="small"
                        label="Method"
                        value={method}
                        onChange={(event) => setMethod(event.target.value as typeof method)}
                    >
                        <MenuItem value="svd">Truncated SVD</MenuItem>
                        <MenuItem value="pca">PCA</MenuItem>
                    </TextField>
                    <TextField
                        size="small"
                        type="number"
                        label="Components"
                        value={components}
                        onChange={(event) =>
                            setComponents(
                                Math.min(3, Math.max(2, Number(event.target.value) || 2))
                            )
                        }
                        inputProps={{ min: 2, max: 3 }}
                    />
                </Stack>
                <Button
                    variant="contained"
                    startIcon={<RunIcon />}
                    onClick={() => mutation.mutate()}
                    disabled={mutation.isPending}
                    sx={{ alignSelf: "flex-start" }}
                >
                    Reduce dimensions
                </Button>
            </PanelBody>
        </SectionCard>
    );
}
