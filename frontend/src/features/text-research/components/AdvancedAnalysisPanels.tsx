import { useState } from "react";
import {
    Button,
    Checkbox,
    FormControlLabel,
    MenuItem,
    Stack,
    TextField,
} from "@mui/material";
import { PlayArrow as RunIcon } from "@mui/icons-material";
import {
    runDuplicateDetection,
    runReadability,
    runSimilarity,
} from "../../../api/textResearch";
import { SectionCard } from "../../../components/ui/SectionCard";
import {
    PanelBody,
} from "./advancedAnalysisShared";
import {
    useAnalysisRun,
    useRunMutation,
    type AdvancedAnalysisBasePayload,
} from "./advancedAnalysisUtils";

export type { AdvancedAnalysisBasePayload as BasePayload };

export function SimilarityExplorer({ basePayload }: { basePayload: AdvancedAnalysisBasePayload }) {
    const { ctx, run, runQuery, setRunId, showToast } = useAnalysisRun();
    const [method, setMethod] = useState<"tfidf_cosine" | "jaccard">("tfidf_cosine");
    const [mode, setMode] = useState<"pairwise" | "query" | "group_centroid">("pairwise");
    const [queryText, setQueryText] = useState("");
    const [groupBy, setGroupBy] = useState("organization");
    const [topK, setTopK] = useState(20);
    const mutation = useRunMutation(
        () =>
            runSimilarity(ctx.selectedCorpusId, {
                ...basePayload,
                method,
                mode,
                top_k: topK,
                ...(mode === "query" ? { query_text: queryText.trim() } : {}),
                ...(mode === "group_centroid" ? { group_by: groupBy.trim() } : {}),
            }),
        "Similarity analysis",
        setRunId,
        showToast
    );
    return (
        <SectionCard
            title="Similarity explorer"
            description="Compare text units pairwise, against a query, or between metadata-group centroids."
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
                        <MenuItem value="tfidf_cosine">TF-IDF cosine</MenuItem>
                        <MenuItem value="jaccard">Jaccard</MenuItem>
                    </TextField>
                    <TextField
                        select
                        size="small"
                        label="Mode"
                        value={mode}
                        onChange={(event) => setMode(event.target.value as typeof mode)}
                    >
                        <MenuItem value="pairwise">Pairwise</MenuItem>
                        <MenuItem value="query">Query</MenuItem>
                        <MenuItem value="group_centroid">Group centroid</MenuItem>
                    </TextField>
                    <TextField
                        size="small"
                        type="number"
                        label="Top results"
                        value={topK}
                        onChange={(event) => setTopK(Math.max(1, Number(event.target.value) || 1))}
                    />
                </Stack>
                {mode === "query" ? (
                    <TextField
                        size="small"
                        label="Query text"
                        value={queryText}
                        onChange={(event) => setQueryText(event.target.value)}
                    />
                ) : null}
                {mode === "group_centroid" ? (
                    <TextField
                        size="small"
                        label="Metadata field"
                        value={groupBy}
                        onChange={(event) => setGroupBy(event.target.value)}
                    />
                ) : null}
                <Button
                    variant="contained"
                    startIcon={<RunIcon />}
                    onClick={() => mutation.mutate()}
                    disabled={
                        mutation.isPending ||
                        (mode === "query" && !queryText.trim()) ||
                        (mode === "group_centroid" && !groupBy.trim())
                    }
                    sx={{ alignSelf: "flex-start" }}
                >
                    Run similarity
                </Button>
            </PanelBody>
        </SectionCard>
    );
}

export function DuplicateDetectionView({
    basePayload,
}: {
    basePayload: AdvancedAnalysisBasePayload;
}) {
    const { ctx, run, runQuery, setRunId, showToast } = useAnalysisRun();
    const [lexical, setLexical] = useState(true);
    const [threshold, setThreshold] = useState(0.85);
    const mutation = useRunMutation(
        () =>
            runDuplicateDetection(ctx.selectedCorpusId, {
                ...basePayload,
                methods: lexical ? ["exact", "normalized", "lexical"] : ["exact", "normalized"],
                lexical_threshold: threshold,
            }),
        "Duplicate detection",
        setRunId,
        showToast
    );
    return (
        <SectionCard
            title="Duplicate detection"
            description="Find exact, normalized, and optionally lexical near-duplicates among selected units."
        >
            <PanelBody run={run} runQuery={runQuery}>
                <Stack
                    direction={{ xs: "column", sm: "row" }}
                    spacing={1}
                    alignItems={{ sm: "center" }}
                >
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={lexical}
                                onChange={(event) => setLexical(event.target.checked)}
                            />
                        }
                        label="Include lexical matches"
                    />
                    <TextField
                        size="small"
                        type="number"
                        label="Lexical threshold"
                        value={threshold}
                        onChange={(event) =>
                            setThreshold(Math.max(0, Math.min(1, Number(event.target.value))))
                        }
                        inputProps={{ min: 0, max: 1, step: 0.05 }}
                    />
                </Stack>
                <Button
                    variant="contained"
                    startIcon={<RunIcon />}
                    onClick={() => mutation.mutate()}
                    disabled={mutation.isPending}
                    sx={{ alignSelf: "flex-start" }}
                >
                    Find duplicates
                </Button>
            </PanelBody>
        </SectionCard>
    );
}

export function ReadabilityView({ basePayload }: { basePayload: AdvancedAnalysisBasePayload }) {
    const { ctx, run, runQuery, setRunId, showToast } = useAnalysisRun();
    const mutation = useRunMutation(
        () => runReadability(ctx.selectedCorpusId, basePayload),
        "Readability analysis",
        setRunId,
        showToast
    );
    return (
        <SectionCard
            title="Readability"
            description="Calculate corpus and unit-level readability measures from the selected text units."
        >
            <PanelBody run={run} runQuery={runQuery}>
                <Button
                    variant="contained"
                    startIcon={<RunIcon />}
                    onClick={() => mutation.mutate()}
                    disabled={mutation.isPending}
                    sx={{ alignSelf: "flex-start" }}
                >
                    Calculate readability
                </Button>
            </PanelBody>
        </SectionCard>
    );
}
