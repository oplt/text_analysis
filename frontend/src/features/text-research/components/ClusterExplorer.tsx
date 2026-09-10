import { useState } from "react";
import { Button, Checkbox, FormControlLabel, MenuItem, Stack, TextField } from "@mui/material";
import { PlayArrow as RunIcon } from "@mui/icons-material";
import { runClustering } from "../../../api/textResearch";
import { SectionCard } from "../../../components/ui/SectionCard";
import {
    PanelBody,
} from "./advancedAnalysisShared";
import {
    useAnalysisRun,
    useRunMutation,
    type AdvancedAnalysisBasePayload,
} from "./advancedAnalysisUtils";

export default function ClusterExplorer({
    basePayload,
}: {
    basePayload: AdvancedAnalysisBasePayload;
}) {
    const { ctx, run, runQuery, setRunId, showToast } = useAnalysisRun();
    const [clusters, setClusters] = useState(5);
    const [algorithm, setAlgorithm] = useState<"kmeans" | "minibatch_kmeans">("kmeans");
    const [useSvd, setUseSvd] = useState(false);
    const mutation = useRunMutation(
        () =>
            runClustering(ctx.selectedCorpusId, {
                ...basePayload,
                n_clusters: clusters,
                algorithm,
                use_svd: useSvd,
            }),
        "Clustering",
        setRunId,
        showToast
    );
    return (
        <SectionCard
            title="Cluster explorer"
            description="Group units using TF-IDF features, then inspect cluster assignments and top terms."
        >
            <PanelBody run={run} runQuery={runQuery}>
                <Stack
                    direction={{ xs: "column", sm: "row" }}
                    spacing={1}
                    alignItems={{ sm: "center" }}
                >
                    <TextField
                        size="small"
                        type="number"
                        label="Clusters"
                        value={clusters}
                        onChange={(event) =>
                            setClusters(Math.max(2, Number(event.target.value) || 2))
                        }
                        inputProps={{ min: 2 }}
                    />
                    <TextField
                        select
                        size="small"
                        label="Algorithm"
                        value={algorithm}
                        onChange={(event) =>
                            setAlgorithm(event.target.value as typeof algorithm)
                        }
                    >
                        <MenuItem value="kmeans">K-means</MenuItem>
                        <MenuItem value="minibatch_kmeans">MiniBatch K-means</MenuItem>
                    </TextField>
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={useSvd}
                                onChange={(event) => setUseSvd(event.target.checked)}
                            />
                        }
                        label="Reduce dimensions first"
                    />
                </Stack>
                <Button
                    variant="contained"
                    startIcon={<RunIcon />}
                    onClick={() => mutation.mutate()}
                    disabled={mutation.isPending}
                    sx={{ alignSelf: "flex-start" }}
                >
                    Run clustering
                </Button>
            </PanelBody>
        </SectionCard>
    );
}
