import type { PreprocessingConfigPayload } from "../../../api/textResearch";

export const DEFAULT_PREPROCESSING_CONFIG: PreprocessingConfigPayload = {
    language: "en",
    language_mode: "manual",
    auto_detect_language: false,
    multilingual: false,
    unicode_normalization: "NFC",
    fix_encoding: false,
    lowercase: true,
    remove_punctuation: true,
    remove_numbers: false,
    remove_stopwords: false,
    preserve_negation: true,
    stemming: false,
    lemmatization: false,
    pos_lemmatization: false,
    spacy_model: "en_core_web_sm",
    enable_ner: false,
    entity_masking: false,
    phrase_detection: false,
    ngram_min: 1,
    ngram_max: 1,
    min_df: 1,
    max_df: 1.0,
    max_features: null,
    custom_stopwords: [],
};

export const LANGUAGE_OPTIONS = [
    { value: "en", label: "English" },
    { value: "de", label: "German" },
    { value: "fr", label: "French" },
    { value: "tr", label: "Turkish" },
] as const;

export const SPACY_MODEL_OPTIONS = [
    { value: "en_core_web_sm", label: "en_core_web_sm" },
    { value: "en_core_web_md", label: "en_core_web_md" },
    { value: "de_core_news_sm", label: "de_core_news_sm" },
    { value: "fr_core_news_sm", label: "fr_core_news_sm" },
] as const;
