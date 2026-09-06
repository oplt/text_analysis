import type { PreprocessingConfigPayload } from "../../../api/textResearch";

export const DEFAULT_PREPROCESSING_CONFIG: PreprocessingConfigPayload = {
    lowercase: true,
    remove_punctuation: true,
    remove_numbers: false,
    remove_stopwords: false,
    preserve_negation: true,
    stemming: false,
    lemmatization: false,
    ngram_min: 1,
    ngram_max: 1,
    min_df: 1,
    max_df: 1.0,
    max_features: null,
    custom_stopwords: [],
};
