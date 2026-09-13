/**
 * Method notes and chart chrome for the Analysis research laboratory (Phase 12).
 * Keep notes interpretive and non-prescriptive.
 */

export type AnalysisMethodId =
    | "overview"
    | "frequencies"
    | "ngrams"
    | "kwic"
    | "dfm"
    | "keyness"
    | "dictionaries"
    | "cooccurrence"
    | "similarity"
    | "duplicates"
    | "clustering"
    | "dimensionality"
    | "readability"
    | "statistical"
    | "measurement";

export type AnalysisChartMeta = {
    title: string;
    subtitle: string;
    /** Series / legend label when a single measure is plotted. */
    seriesLabel?: string;
    xAxisLabel?: string;
    yAxisLabel?: string;
    methodologicalNote: string;
};

export type AnalysisMethodMeta = {
    id: AnalysisMethodId;
    label: string;
    /** Short workspace description shown beside configuration. */
    configurationHint: string;
    /** Side-panel method note (always visible when possible). */
    methodNote: string;
    chart?: AnalysisChartMeta;
};

const META: Record<AnalysisMethodId, AnalysisMethodMeta> = {
    overview: {
        id: "overview",
        label: "Corpus overview",
        configurationHint: "Summarize document and unit coverage for the current selection.",
        methodNote:
            "Corpus overview reports descriptive counts and length distributions for the filtered unit set. It does not estimate causal effects or topical structure.",
        chart: {
            title: "Units by organization",
            subtitle: "Document/unit counts stratified by organization metadata.",
            seriesLabel: "Units",
            xAxisLabel: "Count",
            yAxisLabel: "Organization",
            methodologicalNote:
                "Counts reflect the current metadata filters and unit type. Missing organization values are omitted from the chart.",
        },
    },
    frequencies: {
        id: "frequencies",
        label: "Term frequencies",
        configurationHint: "Rank tokens by raw or relative frequency after preprocessing.",
        methodNote:
            "Term frequencies are bag-of-words counts after the selected preprocessing profile. High-frequency function words may dominate unless the profile removes them.",
        chart: {
            title: "Top terms",
            subtitle: "Highest-ranked tokens in the selected corpus slice.",
            seriesLabel: "Frequency",
            xAxisLabel: "Count / share",
            yAxisLabel: "Term",
            methodologicalNote:
                "Ranks depend on tokenization, stopwording, and Top N. Relative share is corpus-slice specific — compare carefully across unequal subsets.",
        },
    },
    ngrams: {
        id: "ngrams",
        label: "N-grams",
        configurationHint: "Count contiguous token sequences of length N.",
        methodNote:
            "N-grams preserve local word order within a fixed window. They remain sensitive to punctuation handling and lemmatization choices in the preprocessing profile.",
        chart: {
            title: "Top n-grams",
            subtitle: "Most frequent contiguous token sequences.",
            seriesLabel: "Frequency",
            xAxisLabel: "Count / share",
            yAxisLabel: "N-gram",
            methodologicalNote:
                "Phrase ranks shift with N, corpus size, and cleaning. Treat charted n-grams as exploratory evidence, not latent topics.",
        },
    },
    kwic: {
        id: "kwic",
        label: "KWIC",
        configurationHint: "Inspect concordance lines around a query term or passage.",
        methodNote:
            "Lexical KWIC is deterministic concordance over TextUnits. Semantic/hybrid modes retrieve corpus-scoped passages and should be labeled as retrieval-assisted inspection.",
    },
    dfm: {
        id: "dfm",
        label: "Document-feature matrix",
        configurationHint: "Build a sparse document × feature matrix with weighting and trimming.",
        methodNote:
            "A DFM is a sparse bag-of-words representation. Weighting (count/tf-idf) and trim thresholds change sparsity and downstream modeling behavior.",
    },
    keyness: {
        id: "keyness",
        label: "Keyness",
        configurationHint: "Compare distinctive features between two metadata groups.",
        methodNote:
            "Keyness contrasts feature rates between two groups under a chosen statistic and multiple-testing correction. Direction and magnitude are descriptive, not causal.",
        chart: {
            title: "Keyness contrast",
            subtitle: "Signed keyness scores (Group B positive, Group A negative).",
            seriesLabel: "Keyness statistic",
            xAxisLabel: "Signed score",
            yAxisLabel: "Feature",
            methodologicalNote:
                "p-values and BH-adjusted p depend on the chosen test and feature inventory. Always report group sizes, method, and correction alongside the plot.",
        },
    },
    dictionaries: {
        id: "dictionaries",
        label: "Dictionaries",
        configurationHint: "Apply a dictionary or custom term list and summarize hits.",
        methodNote:
            "Dictionary hits count lexical matches only. Conceptual validity depends on dictionary coverage and whether terms are used in the intended sense.",
        chart: {
            title: "Dictionary hits by group",
            subtitle: "Match counts stratified by the selected group-by field.",
            seriesLabel: "Hits",
            xAxisLabel: "Hits",
            yAxisLabel: "Group",
            methodologicalNote:
                "Unequal group sizes affect raw hits. Prefer normalized rates (e.g. hits per 1,000 tokens) when comparing strata.",
        },
    },
    cooccurrence: {
        id: "cooccurrence",
        label: "Co-occurrence",
        configurationHint: "Estimate term association within a sliding window.",
        methodNote:
            "Co-occurrence associations (e.g. PMI) measure local co-presence, not semantic synonymy. Window size and minimum frequency strongly affect edges.",
        chart: {
            title: "Co-occurrence network",
            subtitle: "Edges filtered by count threshold and Top-N display limit.",
            methodologicalNote:
                "Network layout is aesthetic, not a distance model. Edge thickness encodes co-occurrence count; association scores appear in the table.",
        },
    },
    similarity: {
        id: "similarity",
        label: "Similarity",
        configurationHint: "Compare units or documents with an embedding or lexical similarity measure.",
        methodNote:
            "Similarity scores depend on the embedding/model and preprocessing. Near-duplicates and paraphrases can score high without shared topical meaning.",
    },
    duplicates: {
        id: "duplicates",
        label: "Duplicates",
        configurationHint: "Detect near-duplicate or overlapping text units.",
        methodNote:
            "Duplicate detection thresholds trade precision vs recall. Review flagged pairs before dropping units from a scientific sample.",
    },
    clustering: {
        id: "clustering",
        label: "Clustering",
        configurationHint: "Group units by unsupervised structure in feature space.",
        methodNote:
            "Clusters are exploratory partitions. Stability can change with seed, feature space, and k; do not treat cluster IDs as ground-truth categories.",
    },
    dimensionality: {
        id: "dimensionality",
        label: "Dimensionality reduction",
        configurationHint: "Project high-dimensional text features into a low-dimensional view.",
        methodNote:
            "2D/3D projections compress variance and can distort distances. Use them for exploration; confirm patterns with explicit metrics when possible.",
    },
    readability: {
        id: "readability",
        label: "Readability",
        configurationHint: "Compute classical readability indices on text units.",
        methodNote:
            "Readability formulas are language- and genre-sensitive heuristics. Report the index name and do not equate score differences with comprehension outcomes.",
    },
    statistical: {
        id: "statistical",
        label: "Statistical models",
        configurationHint: "Fit supervised or explanatory statistical models on text-derived features.",
        methodNote:
            "Model coefficients and fit statistics are only as valid as the measurement and sampling design. Check diagnostics, identification assumptions, and overfitting risk.",
    },
    measurement: {
        id: "measurement",
        label: "Measurement comparison",
        configurationHint: "Compare alternative operationalizations of the same construct.",
        methodNote:
            "Measurement comparison assesses agreement/divergence across indicators. Prefer reporting uncertainty and construct coverage over a single preferred score.",
    },
};

export function analysisMethodMeta(id: AnalysisMethodId): AnalysisMethodMeta {
    return META[id];
}
