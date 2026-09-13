/**
 * Central help-text registry for academic / NLP / ML / RAG terminology (Phase 5).
 * Short `definition` → tooltip. Optional `detail` → popover "Learn more".
 * Never put required workflow steps only in these strings.
 */

export type HelpTermId =
    | "cohens_kappa"
    | "fleiss_kappa"
    | "krippendorff_alpha"
    | "precision"
    | "recall"
    | "f1"
    | "macro_f1"
    | "micro_f1"
    | "tfidf"
    | "vocabulary_size"
    | "min_df"
    | "max_df"
    | "dimensionality"
    | "confidence"
    | "annotation_agreement"
    | "blind_reliability"
    | "ai_assisted_coding"
    | "agreement_benchmarks"
    | "chunk_size"
    | "chunk_overlap"
    | "embeddings"
    | "similarity_threshold"
    | "top_k_retrieval"
    | "reranking"
    | "temperature"
    | "model_lifecycle"
    | "active_learning"
    | "drift"
    | "robustness"
    | "provenance"
    | "observed_agreement"
    | "expected_agreement"
    | "missingness"
    | "n_coders"
    | "pairable_units"
    | "max_features"
    | "class_imbalance"
    | "class_weight"
    | "train_test_split"
    | "validation_set"
    | "model_family"
    | "citation_coverage"
    | "grounding"
    | "parsing_state"
    | "indexing_state"
    | "agent_tools"
    | "agent_trace";

export type HelpTerm = {
    id: HelpTermId;
    /** Short display title in popovers. */
    title: string;
    /** One–two sentence definition for tooltips. */
    definition: string;
    /** Longer explanation for popover / help drawer. */
    detail?: string;
};

export const HELP_TERMS: Record<HelpTermId, HelpTerm> = {
    cohens_kappa: {
        id: "cohens_kappa",
        title: "Cohen's κ",
        definition:
            "Chance-corrected agreement between exactly two coders on the same units.",
        detail:
            "Prefer Cohen's κ for two-rater designs with complete overlap. Values near 1 mean strong agreement above chance; near 0 means agreement no better than chance. Always report the overlapping sample size.",
    },
    fleiss_kappa: {
        id: "fleiss_kappa",
        title: "Fleiss' κ",
        definition:
            "Chance-corrected agreement among three or more coders on the same units.",
        detail:
            "Use Fleiss' κ for multi-rater designs with complete overlap. It assumes a fixed number of ratings per unit and is less appropriate when many units have missing raters—prefer Krippendorff's α then.",
    },
    krippendorff_alpha: {
        id: "krippendorff_alpha",
        title: "Krippendorff's α",
        definition:
            "Agreement coefficient that supports any number of coders and missing values.",
        detail:
            "Prefer α when there are more than two coders or incomplete annotation overlap. It generalizes several classical reliability measures and remains defined under missingness.",
    },
    precision: {
        id: "precision",
        title: "Precision",
        definition:
            "Of items predicted as a class, the share that truly belong to that class.",
        detail:
            "High precision means few false positives. Useful when false alarms are costly. Report per-class and aggregate (macro/micro) precision for multi-class tasks.",
    },
    recall: {
        id: "recall",
        title: "Recall",
        definition:
            "Of items that truly belong to a class, the share the model correctly finds.",
        detail:
            "High recall means few false negatives. Useful when missing positives is costly. Trade off against precision; F1 summarizes the balance.",
    },
    f1: {
        id: "f1",
        title: "F1",
        definition:
            "Harmonic mean of precision and recall for a single class or binary decision.",
        detail:
            "F1 penalizes extremes: strong precision with weak recall (or vice versa) lowers the score. Prefer macro/micro F1 when comparing multi-class models.",
    },
    macro_f1: {
        id: "macro_f1",
        title: "Macro F1",
        definition:
            "Unweighted average of per-class F1 scores—each class counts equally.",
        detail:
            "Macro F1 highlights performance on rare classes. Prefer it when class balance matters scientifically, not just overall accuracy.",
    },
    micro_f1: {
        id: "micro_f1",
        title: "Micro F1",
        definition:
            "F1 computed from pooled true/false positives and negatives across classes.",
        detail:
            "Micro F1 weights classes by support and often tracks overall accuracy in single-label multi-class settings. Dominant classes influence the score more.",
    },
    tfidf: {
        id: "tfidf",
        title: "TF-IDF",
        definition:
            "Term frequency × inverse document frequency—boosts distinctive terms, down-weights ubiquitous ones.",
        detail:
            "TF-IDF is a sparse bag-of-words representation. It ignores word order but is strong for classical classification and keyness-style analyses. Tune min/max document frequency to control vocabulary noise.",
    },
    vocabulary_size: {
        id: "vocabulary_size",
        title: "Vocabulary size",
        definition:
            "Number of unique terms kept after filtering (and optional max_features).",
        detail:
            "Larger vocabularies capture more detail but increase sparsity and overfitting risk. Document frequency filters and feature selection shrink the vocabulary deliberately.",
    },
    min_df: {
        id: "min_df",
        title: "Minimum document frequency",
        definition:
            "Ignore terms that appear in fewer than this many documents (or this fraction).",
        detail:
            "Raising min_df removes rare, often noisy terms and shrinks the vocabulary. Integer values are document counts; fractions (0–1) are proportions of the corpus.",
    },
    max_df: {
        id: "max_df",
        title: "Maximum document frequency",
        definition:
            "Ignore terms that appear in more than this many documents (or this fraction).",
        detail:
            "Lowering max_df removes overly common terms (near-stopwords in your corpus). Fractions are usually easier to reason about than raw counts on large corpora.",
    },
    dimensionality: {
        id: "dimensionality",
        title: "Dimensionality",
        definition:
            "Number of axes in a feature or embedding space after reduction or model design.",
        detail:
            "Lower dimensionality can reduce noise and compute cost but may discard signal. Report the method (e.g. PCA, SVD) and retained components when interpreting plots.",
    },
    confidence: {
        id: "confidence",
        title: "Confidence",
        definition:
            "Model-estimated probability or score that a prediction is correct—not a guarantee.",
        detail:
            "Confidence is useful for triage (active learning, review queues) but can be poorly calibrated. Prefer reliability diagrams or temperature scaling when decisions depend on probabilities.",
    },
    annotation_agreement: {
        id: "annotation_agreement",
        title: "Annotation agreement",
        definition:
            "How consistently multiple annotators code the same units under a shared codebook.",
        detail:
            "Report chance-corrected metrics (κ / α), sample size, missingness, and codebook version. Low agreement usually means refine the codebook before training classifiers.",
    },
    blind_reliability: {
        id: "blind_reliability",
        title: "Blind reliability coding",
        definition:
            "Independent coding with peer decisions and model suggestions hidden.",
        detail:
            "Use blind reliability when measuring inter-annotator agreement. Hiding peers and AI suggestions prevents contamination of independent judgments. Release the campaign later if you need adjudication or AI-assisted review.",
    },
    ai_assisted_coding: {
        id: "ai_assisted_coding",
        title: "AI-assisted coding",
        definition:
            "Annotators may see model suggestions, but decisions remain human-applied.",
        detail:
            "Suggestions are never auto-written into annotations. Treat AI output as optional evidence beside the codebook. Prefer blind reliability when estimating agreement for scientific reporting.",
    },
    agreement_benchmarks: {
        id: "agreement_benchmarks",
        title: "Agreement benchmarks",
        definition:
            "Verbal labels like “substantial” are illustrative heuristics, not universal scientific cutoffs.",
        detail:
            "Commonly cited bands (e.g. Landis & Koch for κ, or Krippendorff-style guidance for α) were developed in specific contexts. Acceptable reliability depends on codebook complexity, prevalence, missingness, discipline norms, and how the codes will be used. Always report the coefficient, CI, sample size, and codebook version — and treat verbal labels as optional orientation only.",
    },
    chunk_size: {
        id: "chunk_size",
        title: "Chunk size",
        definition:
            "Target length of each retrieved text segment during RAG ingestion (often tokens or characters).",
        detail:
            "Larger chunks keep more local context but dilute relevance; smaller chunks retrieve more precisely but may lose surrounding meaning. Pair with overlap and evaluate retrieval quality.",
    },
    chunk_overlap: {
        id: "chunk_overlap",
        title: "Chunk overlap",
        definition:
            "Shared text between consecutive chunks so sentences are not split without context.",
        detail:
            "Modest overlap reduces boundary artifacts. Excessive overlap duplicates content in the index and can bias retrieval toward repeated passages.",
    },
    embeddings: {
        id: "embeddings",
        title: "Embeddings",
        definition:
            "Dense vector representations of text used for semantic similarity and retrieval.",
        detail:
            "Embedding identity (provider, model, revision, dimension) belongs in provenance. Changing the embedding model invalidates stored vectors and comparisons.",
    },
    similarity_threshold: {
        id: "similarity_threshold",
        title: "Similarity threshold",
        definition:
            "Minimum similarity score required to keep a retrieved or matched item.",
        detail:
            "Raising the threshold increases precision of matches but may drop useful near-neighbors. Thresholds are scale-dependent on the similarity metric (cosine, etc.).",
    },
    top_k_retrieval: {
        id: "top_k_retrieval",
        title: "Top-k retrieval",
        definition:
            "Number of highest-scoring passages returned before optional reranking or context packing.",
        detail:
            "Larger k improves recall into the candidate set but increases noise and context cost. Evaluate answer quality and citation validity when changing k.",
    },
    reranking: {
        id: "reranking",
        title: "Reranking",
        definition:
            "Second-stage scoring that reorders an initial retrieval list for better relevance.",
        detail:
            "Rerankers are slower but can improve precision@k. Document when answers used hybrid retrieval + rerank versus dense-only retrieval.",
    },
    temperature: {
        id: "temperature",
        title: "Temperature",
        definition:
            "Sampling randomness for generative models—higher values yield more varied text.",
        detail:
            "For research answers and coding assistance, lower temperature usually improves consistency. Temperature does not fix weak retrieval; it only changes generation style.",
    },
    model_lifecycle: {
        id: "model_lifecycle",
        title: "Model lifecycle",
        definition:
            "Governed stages for a trained model: candidate → staging → production → deprecated/archived.",
        detail:
            "Only promote models with documented metrics, dataset snapshots, and review. Production models should not be silently overwritten; use versioned artifacts and lifecycle events.",
    },
    active_learning: {
        id: "active_learning",
        title: "Active learning",
        definition:
            "Selects uncertain model predictions for human annotation to improve the next training round.",
        detail:
            "Uncertainty sampling prioritizes units the model is least sure about. Keep selection tied to a frozen prediction set and codebook version for reproducibility.",
    },
    drift: {
        id: "drift",
        title: "Drift",
        definition:
            "Change in data, features, or prediction patterns relative to a reference window.",
        detail:
            "Prediction drift and feature/coefficient drift answer different questions. Investigate before retracting a production model; confirm sampling differences and label shift.",
    },
    robustness: {
        id: "robustness",
        title: "Robustness",
        definition:
            "How stable metrics stay under controlled perturbations (e.g. leave-one-group-out).",
        detail:
            "Robustness checks are not a substitute for held-out evaluation. Report which groups or seeds were ablated and whether folds remained evaluable.",
    },
    provenance: {
        id: "provenance",
        title: "Provenance",
        definition:
            "Record of inputs, parameters, code/model identities, and artifacts that produced a result.",
        detail:
            "Use provenance to replay or exactly reproduce runs. Prefer structured identity fields over informal notes; never treat mutable “replay” as exact reproduction when inputs changed.",
    },
    observed_agreement: {
        id: "observed_agreement",
        title: "Observed agreement",
        definition:
            "Raw share of units where coders assign the same value, before chance correction.",
    },
    expected_agreement: {
        id: "expected_agreement",
        title: "Expected agreement",
        definition:
            "Agreement rate expected by chance given category marginals—used to compute κ.",
    },
    missingness: {
        id: "missingness",
        title: "Missingness",
        definition:
            "Share of coder×unit cells without a value; high missingness shrinks the reliability sample.",
    },
    n_coders: {
        id: "n_coders",
        title: "Number of coders",
        definition:
            "Distinct annotators who contributed values for this label in the selected codebook version.",
    },
    pairable_units: {
        id: "pairable_units",
        title: "Pairable units",
        definition:
            "Units annotated by at least two coders and included in reliability calculations.",
    },
    max_features: {
        id: "max_features",
        title: "Max features",
        definition:
            "Optional cap on vocabulary size after document-frequency filtering.",
        detail:
            "When set, only the top-scoring terms (by term frequency across the corpus) are kept. Leave blank for no hard cap beyond min_df/max_df.",
    },
    class_imbalance: {
        id: "class_imbalance",
        title: "Class imbalance",
        definition:
            "When one label (or value) dominates the labeled sample, overall accuracy can look strong while rare classes fail.",
        detail:
            "Prefer macro F1 and per-class precision/recall, inspect confusion matrices, and consider class_weight=balanced or resampling. Imbalance warnings are review signals — not automatic blockers.",
    },
    class_weight: {
        id: "class_weight",
        title: "Class weight",
        definition:
            "Optional reweighting of training loss so rarer classes contribute more strongly.",
        detail:
            "balanced adjusts weights inversely to class frequency. It can improve recall on minority classes but may reduce precision — always check per-class metrics on a held-out test set.",
    },
    train_test_split: {
        id: "train_test_split",
        title: "Train / test split",
        definition:
            "Partition of labeled units into training data (fit the model) and a held-out test set (final evaluation).",
        detail:
            "This workspace groups splits by source document to reduce leakage across units from the same document. Never tune thresholds or hyperparameters on the test set.",
    },
    validation_set: {
        id: "validation_set",
        title: "Validation set",
        definition:
            "A held-out fold used for model selection (thresholds, hyperparameters) — distinct from the final test set.",
        detail:
            "Keep validation separate from test so reported test metrics stay honest. Nested cross-validation further reduces selection bias when tuning.",
    },
    model_family: {
        id: "model_family",
        title: "Model family",
        definition:
            "The algorithm class (e.g. logistic regression, linear SVM, naïve Bayes, embedding + linear head).",
        detail:
            "Family choice affects inductive bias, calibration quality, and which hyperparameters matter. Report family, feature space, seed, and snapshot version with results.",
    },
    citation_coverage: {
        id: "citation_coverage",
        title: "Citation coverage",
        definition:
            "Share of retrieved chunks that appear as citations in the grounded answer.",
        detail:
            "High coverage means the answer leans on the retrieved evidence set. Low coverage can mean unused evidence, over-generation, or sparse citation formatting — inspect snippets before trusting claims.",
    },
    grounding: {
        id: "grounding",
        title: "Grounding",
        definition:
            "Whether generated claims are supported by retrieved passages (citations) rather than model prior knowledge alone.",
        detail:
            "Grounding is approximate: citations show evidence the system associated with the answer, not a formal proof. Prefer answers with non-degraded retrieval and inspect citation snippets.",
    },
    parsing_state: {
        id: "parsing_state",
        title: "Parsing state",
        definition:
            "Whether the uploaded file has been decoded into text (PDF/DOCX/etc.) ready for chunking.",
        detail:
            "Parsing failures (corrupt PDFs, scanned pages without OCR) block indexing. Re-upload or re-index after fixing the source file.",
    },
    indexing_state: {
        id: "indexing_state",
        title: "Indexing state",
        definition:
            "Whether document chunks have been embedded and written to the vector index for retrieval.",
        detail:
            "Only indexed documents participate in semantic retrieval. Changing embedding models or chunk policy requires re-indexing for consistency.",
    },
    agent_tools: {
        id: "agent_tools",
        title: "Agent tools",
        definition:
            "Capabilities the agent may invoke during a run (e.g. retrieve evidence, generate text, link memory).",
        detail:
            "Tools are not free-form chat. Each tool call should appear as a structured trace step with inputs, results, and errors when available.",
    },
    agent_trace: {
        id: "agent_trace",
        title: "Agent trace",
        definition:
            "Ordered record of plan, tool calls, results, and errors for one agent run.",
        detail:
            "Prefer structured steps over raw logs. Separate the final Output artifact from Trace so researchers can audit process without drowning in text.",
    },
};

/** Resolve a term; returns null for unknown ids (callers should guard). */
export function getHelpTerm(id: HelpTermId | string): HelpTerm | null {
    return (HELP_TERMS as Record<string, HelpTerm>)[id] ?? null;
}

/** Map common metric / parameter key names onto help term ids. */
export function helpTermIdForKey(key: string): HelpTermId | null {
    const normalized = key.trim().toLowerCase().replace(/\s+/g, "_");
    const aliases: Record<string, HelpTermId> = {
        kappa: "cohens_kappa",
        cohens_kappa: "cohens_kappa",
        mean_cohens_kappa: "cohens_kappa",
        fleiss: "fleiss_kappa",
        fleiss_kappa: "fleiss_kappa",
        alpha: "krippendorff_alpha",
        krippendorff_alpha: "krippendorff_alpha",
        observed: "observed_agreement",
        observed_agreement: "observed_agreement",
        expected: "expected_agreement",
        expected_agreement: "expected_agreement",
        missingness: "missingness",
        n_coders: "n_coders",
        pairable: "pairable_units",
        precision: "precision",
        recall: "recall",
        f1: "f1",
        f1_macro: "macro_f1",
        macro_f1: "macro_f1",
        f1_micro: "micro_f1",
        micro_f1: "micro_f1",
        tfidf: "tfidf",
        min_df: "min_df",
        max_df: "max_df",
        max_features: "max_features",
        vocabulary_size: "vocabulary_size",
        vocab_size: "vocabulary_size",
        dimensionality: "dimensionality",
        n_components: "dimensionality",
        confidence: "confidence",
        temperature: "temperature",
        top_k: "top_k_retrieval",
        chunk_size: "chunk_size",
        chunk_overlap: "chunk_overlap",
        embeddings: "embeddings",
        embedding: "embeddings",
        similarity_threshold: "similarity_threshold",
        rerank: "reranking",
        reranking: "reranking",
        lifecycle: "model_lifecycle",
        model_lifecycle: "model_lifecycle",
        active_learning: "active_learning",
        drift: "drift",
        robustness: "robustness",
        provenance: "provenance",
        annotation_agreement: "annotation_agreement",
        blind_reliability: "blind_reliability",
        blind_mode: "blind_reliability",
        ai_assisted: "ai_assisted_coding",
        ai_assisted_coding: "ai_assisted_coding",
        agreement_benchmarks: "agreement_benchmarks",
        interpretation: "agreement_benchmarks",
        class_imbalance: "class_imbalance",
        imbalance: "class_imbalance",
        class_weight: "class_weight",
        train_test_split: "train_test_split",
        test_size: "train_test_split",
        validation_set: "validation_set",
        val_size: "validation_set",
        model_family: "model_family",
        algorithm: "model_family",
        citation_coverage: "citation_coverage",
        grounding: "grounding",
        parsing_state: "parsing_state",
        indexing_state: "indexing_state",
        agent_tools: "agent_tools",
        tools: "agent_tools",
        agent_trace: "agent_trace",
        trace: "agent_trace",
    };
    return aliases[normalized] ?? null;
}
