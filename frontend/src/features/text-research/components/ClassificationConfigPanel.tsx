import type { ReactNode } from "react";
import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Checkbox,
    FormControlLabel,
    MenuItem,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import type { ClassificationTrainConfig, ClassifierAlgorithm } from "./classificationTrainConfig";

type ProfileOption = { id: string; name: string };

function Section({
    title,
    description,
    defaultExpanded = false,
    children,
}: {
    title: string;
    description: string;
    defaultExpanded?: boolean;
    children: ReactNode;
}) {
    return (
        <Accordion defaultExpanded={defaultExpanded} disableGutters>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Stack spacing={0.25}>
                    <Typography variant="subtitle2">{title}</Typography>
                    <Typography variant="caption" color="text.secondary">
                        {description}
                    </Typography>
                </Stack>
            </AccordionSummary>
            <AccordionDetails>
                <Stack spacing={2}>{children}</Stack>
            </AccordionDetails>
        </Accordion>
    );
}

export function ClassificationConfigPanel({
    config,
    onChange,
    profiles,
}: {
    config: ClassificationTrainConfig;
    onChange: (next: ClassificationTrainConfig) => void;
    profiles: ProfileOption[];
}) {
    const patch = (partial: Partial<ClassificationTrainConfig>) =>
        onChange({ ...config, ...partial });
    const isNb = config.algorithm === "multinomial_nb" || config.algorithm === "complement_nb";
    const isSgd = config.algorithm === "sgd_classifier";
    const isEmbedding =
        config.algorithm === "embedding_logistic" || config.algorithm === "embedding_svm";
    const isLinear =
        config.algorithm === "logistic_regression" ||
        config.algorithm === "linear_svm" ||
        isSgd ||
        isEmbedding;

    return (
        <Stack spacing={1}>
            <Section
                title="Dataset"
                description="Snapshot is selected above. Choose preprocessing and task type here."
                defaultExpanded
            >
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                    <TextField
                        select
                        size="small"
                        label="Task type"
                        value={config.taskType}
                        onChange={(e) =>
                            patch({
                                taskType: e.target.value as ClassificationTrainConfig["taskType"],
                            })
                        }
                        sx={{ minWidth: 180 }}
                        helperText="Blank = infer from labels"
                    >
                        <MenuItem value="">Infer automatically</MenuItem>
                        <MenuItem value="binary">Binary</MenuItem>
                        <MenuItem value="multiclass">Multiclass</MenuItem>
                        <MenuItem value="multilabel">Multilabel</MenuItem>
                    </TextField>
                    <TextField
                        select
                        size="small"
                        label="Preprocessing profile"
                        value={config.profileId}
                        onChange={(e) => patch({ profileId: e.target.value })}
                        sx={{ minWidth: 220 }}
                    >
                        <MenuItem value="">Default (none)</MenuItem>
                        {profiles.map((profile) => (
                            <MenuItem key={profile.id} value={profile.id}>
                                {profile.name}
                            </MenuItem>
                        ))}
                    </TextField>
                    <TextField
                        size="small"
                        label="Model name"
                        value={config.modelName}
                        onChange={(e) => patch({ modelName: e.target.value })}
                        sx={{ minWidth: 200 }}
                        placeholder="Optional"
                    />
                </Stack>
            </Section>

            <Section title="Features" description="Vectorizer and n-gram families." defaultExpanded>
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                    <TextField
                        select
                        size="small"
                        label="Vectorizer"
                        value={config.vectorizer}
                        onChange={(e) =>
                            patch({
                                vectorizer: e.target.value as "tfidf" | "count",
                            })
                        }
                        disabled={isEmbedding}
                        sx={{ minWidth: 140 }}
                    >
                        <MenuItem value="tfidf">TF-IDF</MenuItem>
                        <MenuItem value="count">Count</MenuItem>
                    </TextField>
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={config.useWordNgrams}
                                onChange={(_, checked) => patch({ useWordNgrams: checked })}
                                disabled={isEmbedding}
                            />
                        }
                        label="Word n-grams"
                    />
                    <TextField
                        size="small"
                        type="number"
                        label="Word n-gram min"
                        value={config.ngramMin}
                        onChange={(e) => patch({ ngramMin: Number(e.target.value) || 1 })}
                        inputProps={{ min: 1, max: 5, step: 1 }}
                        disabled={!config.useWordNgrams || isEmbedding}
                        sx={{ width: 130 }}
                    />
                    <TextField
                        size="small"
                        type="number"
                        label="Word n-gram max"
                        value={config.ngramMax}
                        onChange={(e) => patch({ ngramMax: Number(e.target.value) || 1 })}
                        inputProps={{ min: 1, max: 5, step: 1 }}
                        disabled={!config.useWordNgrams || isEmbedding}
                        sx={{ width: 130 }}
                    />
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={config.useCharNgrams}
                                onChange={(_, checked) => patch({ useCharNgrams: checked })}
                                disabled={isEmbedding}
                            />
                        }
                        label="Character n-grams"
                    />
                    <TextField
                        size="small"
                        type="number"
                        label="Char n-gram min"
                        value={config.charNgramMin}
                        onChange={(e) => patch({ charNgramMin: Number(e.target.value) || 3 })}
                        inputProps={{ min: 2, max: 10, step: 1 }}
                        disabled={!config.useCharNgrams || isEmbedding}
                        sx={{ width: 130 }}
                    />
                    <TextField
                        size="small"
                        type="number"
                        label="Char n-gram max"
                        value={config.charNgramMax}
                        onChange={(e) => patch({ charNgramMax: Number(e.target.value) || 5 })}
                        inputProps={{ min: 2, max: 10, step: 1 }}
                        disabled={!config.useCharNgrams || isEmbedding}
                        sx={{ width: 130 }}
                    />
                    <TextField
                        size="small"
                        type="number"
                        label="min_df"
                        value={config.minDf}
                        onChange={(e) => patch({ minDf: Number(e.target.value) || 1 })}
                        inputProps={{ min: 1, step: 1 }}
                        disabled={isEmbedding}
                        sx={{ width: 120 }}
                    />
                    <TextField
                        size="small"
                        type="number"
                        label="max_df"
                        value={config.maxDf}
                        onChange={(e) => patch({ maxDf: Number(e.target.value) || 1 })}
                        inputProps={{ min: 0.01, max: 1, step: 0.01 }}
                        disabled={isEmbedding}
                        sx={{ width: 120 }}
                    />
                    <TextField
                        size="small"
                        label="max_features"
                        value={config.maxFeatures}
                        onChange={(e) => patch({ maxFeatures: e.target.value })}
                        placeholder="unlimited"
                        disabled={isEmbedding}
                        sx={{ width: 140 }}
                        helperText="Blank = no cap"
                    />
                </Stack>
            </Section>

            <Section
                title="Feature selection"
                description="Supervised selection after DF pruning. Fit on train labels only."
                defaultExpanded
            >
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                    <TextField
                        select
                        size="small"
                        label="Method"
                        value={config.featureSelectionMethod}
                        onChange={(e) =>
                            patch({
                                featureSelectionMethod: e.target
                                    .value as ClassificationTrainConfig["featureSelectionMethod"],
                            })
                        }
                        disabled={isEmbedding}
                        sx={{ minWidth: 180 }}
                    >
                        <MenuItem value="none">None</MenuItem>
                        <MenuItem value="chi2">Chi-square</MenuItem>
                        <MenuItem value="mutual_info">Mutual information</MenuItem>
                        <MenuItem value="l1">L1 selection</MenuItem>
                    </TextField>
                    <TextField
                        size="small"
                        label="Top K"
                        value={config.featureSelectionK}
                        onChange={(e) => patch({ featureSelectionK: e.target.value })}
                        disabled={
                            isEmbedding ||
                            config.featureSelectionMethod === "none" ||
                            config.featureSelectionMethod === "l1" ||
                            Boolean(config.featureSelectionPercentile.trim())
                        }
                        sx={{ width: 140 }}
                        helperText="Integer or all"
                    />
                    <TextField
                        size="small"
                        label="Percentile"
                        value={config.featureSelectionPercentile}
                        onChange={(e) => patch({ featureSelectionPercentile: e.target.value })}
                        disabled={
                            isEmbedding ||
                            config.featureSelectionMethod === "none" ||
                            config.featureSelectionMethod === "l1"
                        }
                        sx={{ width: 140 }}
                        helperText="Optional; overrides K"
                    />
                </Stack>
            </Section>

            <Section title="Model" description="Algorithm and regularization." defaultExpanded>
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                    <TextField
                        select
                        size="small"
                        label="Algorithm"
                        value={config.algorithm}
                        onChange={(e) =>
                            patch({
                                algorithm: e.target.value as ClassifierAlgorithm,
                            })
                        }
                        sx={{ minWidth: 220 }}
                    >
                        <MenuItem value="logistic_regression">Logistic regression</MenuItem>
                        <MenuItem value="linear_svm">Linear SVM</MenuItem>
                        <MenuItem value="multinomial_nb">Multinomial NB</MenuItem>
                        <MenuItem value="complement_nb">Complement NB</MenuItem>
                        <MenuItem value="sgd_classifier">SGD classifier</MenuItem>
                        <MenuItem value="embedding_logistic">Embedding + logistic</MenuItem>
                        <MenuItem value="embedding_svm">Embedding + SVM</MenuItem>
                    </TextField>
                    {isEmbedding ? (
                        <TextField
                            select
                            size="small"
                            label="Embedding provider"
                            value={config.embeddingProvider}
                            onChange={(e) => patch({ embeddingProvider: e.target.value })}
                            sx={{ minWidth: 180 }}
                        >
                            <MenuItem value="hashing">Hashing (lexical)</MenuItem>
                            <MenuItem value="sentence_transformers">Sentence transformers</MenuItem>
                        </TextField>
                    ) : null}
                    <TextField
                        select
                        size="small"
                        label="class_weight"
                        value={config.classWeight}
                        onChange={(e) =>
                            patch({
                                classWeight: e.target.value as "none" | "balanced",
                            })
                        }
                        sx={{ width: 150 }}
                    >
                        <MenuItem value="none">none</MenuItem>
                        <MenuItem value="balanced">balanced</MenuItem>
                    </TextField>
                    {isLinear && !isNb ? (
                        <TextField
                            size="small"
                            type="number"
                            label="C (regularization)"
                            value={config.regularizationC}
                            onChange={(e) =>
                                patch({ regularizationC: Number(e.target.value) || 1 })
                            }
                            inputProps={{ min: 0.001, step: 0.1 }}
                            sx={{ width: 150 }}
                        />
                    ) : null}
                    {isNb ? (
                        <TextField
                            size="small"
                            type="number"
                            label="NB alpha"
                            value={config.nbAlpha}
                            onChange={(e) => patch({ nbAlpha: Number(e.target.value) || 1 })}
                            inputProps={{ min: 0.0001, step: 0.1 }}
                            sx={{ width: 130 }}
                        />
                    ) : null}
                    {isSgd ? (
                        <TextField
                            select
                            size="small"
                            label="SGD loss"
                            value={config.sgdLoss}
                            onChange={(e) => patch({ sgdLoss: e.target.value })}
                            sx={{ minWidth: 160 }}
                        >
                            <MenuItem value="log_loss">log_loss</MenuItem>
                            <MenuItem value="hinge">hinge</MenuItem>
                            <MenuItem value="modified_huber">modified_huber</MenuItem>
                        </TextField>
                    ) : null}
                </Stack>
            </Section>

            <Section
                title="Validation"
                description="Document-grouped holdout or nested grouped CV."
                defaultExpanded
            >
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                    <TextField
                        select
                        size="small"
                        label="Validation strategy"
                        value={config.validationStrategy}
                        onChange={(e) =>
                            patch({
                                validationStrategy: e.target
                                    .value as ClassificationTrainConfig["validationStrategy"],
                            })
                        }
                        sx={{ minWidth: 200 }}
                    >
                        <MenuItem value="holdout">Grouped holdout</MenuItem>
                        <MenuItem value="nested_grouped_cv">Nested grouped CV</MenuItem>
                    </TextField>
                    <TextField
                        size="small"
                        type="number"
                        label="test_size"
                        value={config.testSize}
                        onChange={(e) => patch({ testSize: Number(e.target.value) || 0.25 })}
                        inputProps={{ min: 0.05, max: 0.5, step: 0.05 }}
                        sx={{ width: 120 }}
                        disabled={config.validationStrategy === "nested_grouped_cv"}
                    />
                    <TextField
                        size="small"
                        type="number"
                        label="val_size"
                        value={config.valSize}
                        onChange={(e) => patch({ valSize: Number(e.target.value) || 0 })}
                        inputProps={{ min: 0, max: 0.5, step: 0.05 }}
                        sx={{ width: 120 }}
                        helperText="0 disables val fold"
                        disabled={config.validationStrategy === "nested_grouped_cv"}
                    />
                    {config.validationStrategy === "nested_grouped_cv" ? (
                        <>
                            <TextField
                                size="small"
                                type="number"
                                label="Outer splits"
                                value={config.nestedOuter}
                                onChange={(e) =>
                                    patch({ nestedOuter: Number(e.target.value) || 5 })
                                }
                                inputProps={{ min: 2, max: 10, step: 1 }}
                                sx={{ width: 130 }}
                            />
                            <TextField
                                size="small"
                                type="number"
                                label="Inner splits"
                                value={config.nestedInner}
                                onChange={(e) =>
                                    patch({ nestedInner: Number(e.target.value) || 3 })
                                }
                                inputProps={{ min: 2, max: 10, step: 1 }}
                                sx={{ width: 130 }}
                            />
                        </>
                    ) : null}
                    <TextField
                        size="small"
                        type="number"
                        label="random_seed"
                        value={config.randomSeed}
                        onChange={(e) => patch({ randomSeed: Number(e.target.value) || 0 })}
                        inputProps={{ step: 1 }}
                        sx={{ width: 130 }}
                    />
                </Stack>
            </Section>

            <Section title="Calibration" description="Probability recalibration method.">
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                    <TextField
                        select
                        size="small"
                        label="Calibration method"
                        value={config.calibrationMethod}
                        onChange={(e) =>
                            patch({
                                calibrationMethod: e.target.value as "sigmoid" | "isotonic",
                            })
                        }
                        sx={{ minWidth: 180 }}
                    >
                        <MenuItem value="sigmoid">Sigmoid (Platt)</MenuItem>
                        <MenuItem value="isotonic">Isotonic</MenuItem>
                    </TextField>
                </Stack>
            </Section>

            <Section
                title="Uncertainty"
                description="Threshold tuning and bootstrap confidence intervals."
            >
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={config.tuneThresholds}
                                onChange={(_, checked) => patch({ tuneThresholds: checked })}
                            />
                        }
                        label="Tune decision thresholds on validation"
                    />
                    <TextField
                        select
                        size="small"
                        label="Threshold objective"
                        value={config.thresholdObjective}
                        onChange={(e) => patch({ thresholdObjective: e.target.value })}
                        disabled={!config.tuneThresholds}
                        sx={{ minWidth: 200 }}
                    >
                        <MenuItem value="f1">F1</MenuItem>
                        <MenuItem value="precision">Precision</MenuItem>
                        <MenuItem value="recall">Recall</MenuItem>
                        <MenuItem value="balanced_accuracy">Balanced accuracy</MenuItem>
                        <MenuItem value="youden_j">Youden J</MenuItem>
                        <MenuItem value="expected_cost">Expected cost</MenuItem>
                        <MenuItem value="custom_utility">Custom utility</MenuItem>
                    </TextField>
                    <TextField
                        size="small"
                        type="number"
                        label="Bootstrap samples"
                        value={config.bootstrapSamples}
                        onChange={(e) =>
                            patch({ bootstrapSamples: Number(e.target.value) || 0 })
                        }
                        inputProps={{ min: 0, max: 2000, step: 50 }}
                        sx={{ width: 160 }}
                    />
                    <TextField
                        size="small"
                        type="number"
                        label="CI level"
                        value={config.ciLevel}
                        onChange={(e) => patch({ ciLevel: Number(e.target.value) || 0.95 })}
                        inputProps={{ min: 0.5, max: 0.99, step: 0.01 }}
                        sx={{ width: 120 }}
                    />
                </Stack>
            </Section>

            <Section title="Advanced" description="Hyperparameter search grid / random search.">
                <Stack spacing={2}>
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={config.tuneHyperparameters}
                                onChange={(_, checked) => patch({ tuneHyperparameters: checked })}
                            />
                        }
                        label="Enable hyperparameter search (validation only)"
                    />
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                        <TextField
                            select
                            size="small"
                            label="Search type"
                            value={config.searchType}
                            onChange={(e) =>
                                patch({
                                    searchType: e.target.value as "grid" | "random",
                                })
                            }
                            disabled={!config.tuneHyperparameters}
                            sx={{ minWidth: 140 }}
                        >
                            <MenuItem value="grid">Grid</MenuItem>
                            <MenuItem value="random">Random</MenuItem>
                        </TextField>
                        <TextField
                            select
                            size="small"
                            label="Scoring metric"
                            value={config.searchScoring}
                            onChange={(e) => patch({ searchScoring: e.target.value })}
                            disabled={!config.tuneHyperparameters}
                            sx={{ minWidth: 160 }}
                        >
                            <MenuItem value="f1_macro">f1_macro</MenuItem>
                            <MenuItem value="f1_micro">f1_micro</MenuItem>
                            <MenuItem value="f1_weighted">f1_weighted</MenuItem>
                            <MenuItem value="accuracy">accuracy</MenuItem>
                            <MenuItem value="roc_auc">roc_auc</MenuItem>
                        </TextField>
                        <TextField
                            size="small"
                            type="number"
                            label="Random n_iter"
                            value={config.hyperparameterNIter}
                            onChange={(e) =>
                                patch({ hyperparameterNIter: Number(e.target.value) || 10 })
                            }
                            disabled={!config.tuneHyperparameters || config.searchType !== "random"}
                            inputProps={{ min: 1, max: 100, step: 1 }}
                            sx={{ width: 130 }}
                        />
                    </Stack>
                    <TextField
                        size="small"
                        label="Parameter grid (JSON object)"
                        value={config.paramGridText}
                        onChange={(e) => patch({ paramGridText: e.target.value })}
                        disabled={!config.tuneHyperparameters}
                        multiline
                        minRows={3}
                        placeholder='{"C":[0.1,1,10],"max_features":[5000,null]}'
                        helperText="Optional. Keys must match tunable hyperparameters."
                        fullWidth
                    />
                </Stack>
            </Section>
        </Stack>
    );
}
