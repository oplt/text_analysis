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
