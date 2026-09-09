export type ClassifierAlgorithm =
    | "logistic_regression"
    | "linear_svm"
    | "multinomial_nb"
    | "complement_nb"
    | "sgd_classifier"
    | "embedding_logistic"
    | "embedding_svm";

export type ClassificationTrainConfig = {
    algorithm: ClassifierAlgorithm;
    taskType: "" | "binary" | "multiclass" | "multilabel";
    profileId: string;
    modelName: string;
    vectorizer: "tfidf" | "count";
    useWordNgrams: boolean;
    ngramMin: number;
    ngramMax: number;
    useCharNgrams: boolean;
    charNgramMin: number;
    charNgramMax: number;
    minDf: number;
    maxDf: number;
    maxFeatures: string;
    featureSelectionMethod: "none" | "chi2" | "mutual_info" | "l1";
    featureSelectionK: string;
    featureSelectionPercentile: string;
    classWeight: "none" | "balanced";
    regularizationC: number;
    nbAlpha: number;
    sgdLoss: string;
    testSize: number;
    valSize: number;
    randomSeed: number;
    validationStrategy: "holdout" | "nested_grouped_cv";
    nestedOuter: number;
    nestedInner: number;
    tuneHyperparameters: boolean;
    searchType: "grid" | "random";
    paramGridText: string;
    searchScoring: string;
    hyperparameterNIter: number;
    tuneThresholds: boolean;
    thresholdObjective: string;
    bootstrapSamples: number;
    ciLevel: number;
    calibrationMethod: "sigmoid" | "isotonic";
    embeddingProvider: string;
};

export const DEFAULT_CLASSIFICATION_TRAIN_CONFIG: ClassificationTrainConfig = {
    algorithm: "logistic_regression",
    taskType: "",
    profileId: "",
    modelName: "",
    vectorizer: "tfidf",
    useWordNgrams: true,
    ngramMin: 1,
    ngramMax: 2,
    useCharNgrams: false,
    charNgramMin: 3,
    charNgramMax: 5,
    minDf: 1,
    maxDf: 1,
    maxFeatures: "",
    featureSelectionMethod: "none",
    featureSelectionK: "5000",
    featureSelectionPercentile: "",
    classWeight: "balanced",
    regularizationC: 1,
    nbAlpha: 1,
    sgdLoss: "log_loss",
    testSize: 0.25,
    valSize: 0.2,
    randomSeed: 42,
    validationStrategy: "holdout",
    nestedOuter: 5,
    nestedInner: 3,
    tuneHyperparameters: false,
    searchType: "grid",
    paramGridText: "",
    searchScoring: "f1_macro",
    hyperparameterNIter: 10,
    tuneThresholds: true,
    thresholdObjective: "f1",
    bootstrapSamples: 200,
    ciLevel: 0.95,
    calibrationMethod: "sigmoid",
    embeddingProvider: "hashing",
};
