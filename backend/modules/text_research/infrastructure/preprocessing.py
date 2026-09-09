"""Deterministic Unicode-aware text preprocessing for text research.

Pipeline stages (all optional via config): encoding fix (ftfy) → Unicode
normalization → whitespace normalize → lowercase → tokenization → number /
stopword filters → stemming or lemmatization.

Original text is never mutated. Stemming / lemmatization are never claimed
unless the required language resources are actually available; requesting an
unavailable transform raises ``ValueError``.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import ftfy
import regex
import simplemma
import snowballstemmer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer

from backend.modules.text_research.infrastructure import language_processing as lang
from backend.modules.text_research.infrastructure.language_processing import (
    UNICODE_TOKEN_RE as _UNICODE_TOKEN_RE,
)
from backend.modules.text_research.infrastructure.language_processing import (
    lemmatization_available,
    negation_words_for,
    resolve_language,
    snowball_algorithm_for_language,
    stemming_available,
    stopwords_for,
)

PREPROCESSING_IMPLEMENTATION = "text_research.preprocessing"
PREPROCESSING_IMPLEMENTATION_VERSION = "3"

DEFAULT_PREPROCESSING_CONFIG: dict[str, Any] = {
    "language": "en",
    "language_mode": "manual",  # manual | auto | per_unit
    "auto_detect_language": False,
    "multilingual": False,
    "unicode_normalization": "NFC",  # None | NFC | NFKC | NFD | NFKD
    "fix_encoding": False,
    "lowercase": True,
    "remove_punctuation": True,
    "remove_numbers": False,
    "remove_stopwords": False,
    "preserve_negation": True,
    "stemming": False,
    "lemmatization": False,
    "pos_lemmatization": False,
    "spacy_model": "en_core_web_sm",
    "enable_ner": False,
    "entity_masking": False,
    "phrase_detection": False,
    "ngram_min": 1,
    "ngram_max": 1,
    "min_df": 1,
    "max_df": 1.0,
    "max_features": None,
    "custom_stopwords": [],
}

_VALID_UNICODE_FORMS = frozenset({"NFC", "NFKC", "NFD", "NFKD"})

_WHITESPACE_RE = regex.compile(r"\s+")
_NUMBER_RE = regex.compile(r"^[\p{N}]+([.,][\p{N}]+)*$", regex.VERSION1)

# Backward-compatible English defaults (language-aware helpers prefer stopwords_for).
NEGATION_WORDS: frozenset[str] = negation_words_for("en")
STOPWORDS: frozenset[str] = stopwords_for("en")

assert STOPWORDS.isdisjoint(NEGATION_WORDS)

_stemmer_cache: dict[str, Any] = {}


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def normalize_language_code(language: str | None) -> str:
    """Normalize to a primary subtag; blank/missing defaults to ``en`` for profiles."""
    code = lang.normalize_language_code(language)
    return code or "en"


@dataclass
class PreprocessingConfig:
    """Serializable preprocessing profile used by analysis services."""

    language: str = "en"
    language_mode: str = "manual"
    auto_detect_language: bool = False
    multilingual: bool = False
    unicode_normalization: str | None = "NFC"
    fix_encoding: bool = False
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = False
    preserve_negation: bool = True
    stemming: bool = False
    lemmatization: bool = False
    pos_lemmatization: bool = False
    spacy_model: str = "en_core_web_sm"
    enable_ner: bool = False
    entity_masking: bool = False
    phrase_detection: bool = False
    ngram_min: int = 1
    ngram_max: int = 1
    min_df: int | float = 1
    max_df: float = 1.0
    max_features: int | None = None
    custom_stopwords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PreprocessingConfig:
        merged = dict(DEFAULT_PREPROCESSING_CONFIG)
        if data:
            merged.update(data)
        form = merged.get("unicode_normalization")
        if form is not None:
            form = str(form).strip().upper()
            if form not in _VALID_UNICODE_FORMS:
                raise ValueError(
                    f"unicode_normalization must be one of "
                    f"{sorted(_VALID_UNICODE_FORMS)} or null, got {form!r}"
                )
            merged["unicode_normalization"] = form
        merged["language"] = normalize_language_code(merged.get("language") or "en")
        mode = str(merged.get("language_mode") or "manual").strip().lower()
        if mode not in {"manual", "auto", "per_unit"}:
            raise ValueError("language_mode must be one of: manual, auto, per_unit")
        merged["language_mode"] = mode
        if merged.get("auto_detect_language") and mode == "manual":
            merged["language_mode"] = "auto"
        if merged.get("multilingual"):
            merged["language_mode"] = "per_unit"
        merged["custom_stopwords"] = list(merged.get("custom_stopwords") or [])
        merged["spacy_model"] = str(merged.get("spacy_model") or "en_core_web_sm").strip()
        if merged.get("stemming") and merged.get("lemmatization"):
            raise ValueError("Enable either stemming or lemmatization, not both.")
        if merged.get("pos_lemmatization") and (
            merged.get("stemming") or merged.get("lemmatization")
        ):
            raise ValueError(
                "pos_lemmatization uses spaCy lemmas; disable stemming/lemmatization first."
            )
        if merged.get("stemming") and not stemming_available(merged["language"]):
            raise ValueError(
                f"Stemming requested but no Snowball stemmer is available for "
                f"language={merged['language']!r}."
            )
        if merged.get("lemmatization") and not lemmatization_available(merged["language"]):
            raise ValueError(
                f"Lemmatization requested but no lemmatizer resources are available for "
                f"language={merged['language']!r}."
            )
        # spaCy-backed options are validated at tokenize/preview time so profiles
        # can be authored before optional NLP extras are installed.
        field_names = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: merged[key] for key in field_names if key in merged})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_implementation(
    config: PreprocessingConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recordable preprocessing provenance (never claims unused transforms)."""
    cfg = (
        config if isinstance(config, PreprocessingConfig) else PreprocessingConfig.from_dict(config)
    )
    lang_code = normalize_language_code(cfg.language)
    profile = resolve_language(lang_code)
    stem_algo = snowball_algorithm_for_language(lang_code) if cfg.stemming else None
    lemma_name = "simplemma" if cfg.lemmatization else None
    if cfg.pos_lemmatization:
        lemma_name = f"spacy:{cfg.spacy_model}"
    from backend.modules.text_research.infrastructure.nlp_preprocessing import (
        describe_spacy_provenance,
        spacy_available,
    )

    spacy_meta = describe_spacy_provenance(cfg.spacy_model)
    return {
        "preprocessing_implementation": PREPROCESSING_IMPLEMENTATION,
        "preprocessing_implementation_version": PREPROCESSING_IMPLEMENTATION_VERSION,
        "language": lang_code,
        "language_mode": cfg.language_mode,
        "auto_detect_language": bool(cfg.auto_detect_language),
        "multilingual": bool(cfg.multilingual),
        "language_profile": profile.to_dict(),
        "tokenizer": "spacy_pos" if cfg.pos_lemmatization else "unicode_regex",
        "tokenizer_pattern": None if cfg.pos_lemmatization else _UNICODE_TOKEN_RE.pattern,
        "unicode_normalization": cfg.unicode_normalization,
        "fix_encoding": bool(cfg.fix_encoding),
        "stemmer": f"snowball_{stem_algo}" if stem_algo else None,
        "stemmer_package": "snowballstemmer" if cfg.stemming else None,
        "stemmer_package_version": _pkg_version("snowballstemmer") if cfg.stemming else None,
        "lemmatizer": lemma_name,
        "lemmatizer_package": (
            "spacy" if cfg.pos_lemmatization else ("simplemma" if cfg.lemmatization else None)
        ),
        "lemmatizer_package_version": (
            spacy_meta.get("package_version")
            if cfg.pos_lemmatization
            else (_pkg_version("simplemma") if cfg.lemmatization else None)
        ),
        "model_name": cfg.spacy_model
        if (cfg.pos_lemmatization or cfg.entity_masking or cfg.phrase_detection or cfg.enable_ner)
        else None,
        "model_version": spacy_meta.get("model_version")
        if (cfg.pos_lemmatization or cfg.entity_masking or cfg.phrase_detection or cfg.enable_ner)
        else None,
        "spacy": spacy_meta,
        "entity_masking": bool(cfg.entity_masking),
        "phrase_detection": bool(cfg.phrase_detection),
        "enable_ner": bool(cfg.enable_ner),
        "pos_lemmatization": bool(cfg.pos_lemmatization),
        "package_versions": {
            "regex": _pkg_version("regex"),
            "ftfy": _pkg_version("ftfy"),
            "snowballstemmer": _pkg_version("snowballstemmer"),
            "simplemma": _pkg_version("simplemma"),
            "spacy": spacy_meta.get("package_version"),
        },
        "stemming_available": stemming_available(lang_code),
        "lemmatization_available": lemmatization_available(lang_code),
        "spacy_available": spacy_available(cfg.spacy_model),
        "stemming_requested": bool(cfg.stemming),
        "lemmatization_requested": bool(cfg.lemmatization),
        "degraded": profile.degraded,
    }


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to a single space and strip ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def apply_unicode_normalization(text: str, form: str | None) -> str:
    if not form:
        return text
    return unicodedata.normalize(form, text)


def _is_number_token(token: str) -> bool:
    return bool(_NUMBER_RE.match(token))


def _get_snowball_stemmer(language: str | None):
    algo = snowball_algorithm_for_language(language)
    if algo is None:
        raise ValueError(f"No Snowball stemmer for language={language!r}")
    if algo not in _stemmer_cache:
        _stemmer_cache[algo] = snowballstemmer.stemmer(algo)
    return _stemmer_cache[algo]


def snowball_stem(token: str, language: str | None = "en") -> str:
    """Stem ``token`` with the Snowball stemmer for ``language``."""
    return _get_snowball_stemmer(language).stemWord(token)


def simple_stem(token: str) -> str:
    """Compatibility alias for English Snowball stemming."""
    return snowball_stem(token, "en")


def lemmatize_token(token: str, language: str | None = "en") -> str:
    lang = normalize_language_code(language)
    if not lemmatization_available(lang):
        raise ValueError(f"Lemmatization unavailable for language={lang!r}")
    return simplemma.lemmatize(token, lang=lang)


def _merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return PreprocessingConfig.from_dict(config).to_dict()


def _stopword_set(config: dict[str, Any]) -> frozenset[str]:
    """Build the effective stopword set for the given config.

    Uses language-specific stopwords. Unknown languages get an empty lexicon
    (never silently fall back to English). ``preserve_negation`` protects
    language-specific negation tokens even if listed in custom_stopwords.
    """
    language = config.get("language")
    stopwords = set(stopwords_for(language))
    stopwords.update(config.get("custom_stopwords") or [])
    if config.get("preserve_negation", True):
        stopwords -= negation_words_for(language)
    return frozenset(stopwords)


def tokenize(text: str, config: dict[str, Any] | None = None) -> list[str]:
    """Tokenize ``text`` according to ``config``.

    Original ``text`` is never mutated.
    """
    cfg = _merge_config(config)
    language = cfg.get("language") or "en"
    negation_words = (
        negation_words_for(language) if cfg.get("preserve_negation", True) else frozenset()
    )
    model_name = str(cfg.get("spacy_model") or "en_core_web_sm")

    working = text
    if cfg.get("fix_encoding"):
        working = ftfy.fix_text(working)
    working = apply_unicode_normalization(working, cfg.get("unicode_normalization"))
    working = normalize_whitespace(working)

    if cfg.get("entity_masking"):
        from backend.modules.text_research.infrastructure.nlp_preprocessing import (
            mask_named_entities,
        )

        working = mask_named_entities(working, model_name=model_name)

    if cfg.get("pos_lemmatization"):
        from backend.modules.text_research.infrastructure.nlp_preprocessing import (
            pos_aware_lemmas,
        )

        tokens = pos_aware_lemmas(working, model_name=model_name)
        if cfg.get("lowercase", True):
            tokens = [t.lower() for t in tokens]
    else:
        if cfg.get("lowercase", True):
            # Use lower() (not casefold) for stable research reproducibility across profiles.
            working = working.lower()

        if cfg.get("remove_punctuation", True):
            tokens = _UNICODE_TOKEN_RE.findall(working)
        else:
            tokens = [t for t in working.split(" ") if t]

    if cfg.get("remove_numbers", False):
        tokens = [t for t in tokens if t in negation_words or not _is_number_token(t)]

    if cfg.get("remove_stopwords", False):
        stop_set = _stopword_set(cfg)
        tokens = [t for t in tokens if t in negation_words or t not in stop_set]

    if not cfg.get("pos_lemmatization"):
        if cfg.get("lemmatization"):
            tokens = [t if t in negation_words else lemmatize_token(t, language) for t in tokens]
        elif cfg.get("stemming"):
            tokens = [t if t in negation_words else snowball_stem(t, language) for t in tokens]

    if cfg.get("phrase_detection"):
        from backend.modules.text_research.infrastructure.nlp_preprocessing import (
            merge_phrase_tokens,
            noun_chunk_phrases,
        )

        phrases = noun_chunk_phrases(text, model_name=model_name)
        tokens = merge_phrase_tokens(tokens, phrases)

    return tokens


def preprocess_text(text: str, config: dict[str, Any] | None = None) -> str:
    """Return the space-joined token sequence for ``text``.

    The original ``text`` is never mutated; this returns a new string.
    """
    return " ".join(tokenize(text, config))


def _baseline_tokens(text: str, config: dict[str, Any]) -> list[str]:
    """Tokens before stopword/number removal and stem/lemma (for preview diffs)."""
    baseline = {
        **config,
        "remove_numbers": False,
        "remove_stopwords": False,
        "stemming": False,
        "lemmatization": False,
        "custom_stopwords": [],
    }
    return tokenize(text, baseline)


def preview_preprocessing(
    texts: list[str],
    config: dict[str, Any] | None = None,
    *,
    removed_top_n: int = 20,
) -> dict[str, Any]:
    """Build a live preprocessing preview for sample texts."""
    cfg = _merge_config(config)
    impl = describe_implementation(cfg)
    rows: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    before_total = 0
    after_total = 0
    vocabulary: set[str] = set()

    for text in texts:
        before = _baseline_tokens(text, cfg)
        after = tokenize(text, cfg)
        before_total += len(before)
        after_total += len(after)
        vocabulary.update(after)
        removed.update(Counter(before) - Counter(after))
        rows.append(
            {
                "original": text,
                "processed": " ".join(after),
                "token_count_before": len(before),
                "token_count_after": len(after),
            }
        )

    return {
        "rows": rows,
        "token_count_before": before_total,
        "token_count_after": after_total,
        "vocabulary_size": len(vocabulary),
        "most_frequently_removed_terms": [
            {"term": term, "count": count} for term, count in removed.most_common(removed_top_n)
        ],
        "config": cfg,
        "stemmer": impl["stemmer"] or ("snowball_english" if stemming_available("en") else None),
        "lemmatization_supported": impl["lemmatization_available"],
        "spacy_available": impl.get("spacy_available", False),
        "implementation": impl,
    }


def _vectorizer_kwargs(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = _merge_config(config)
    return {
        "tokenizer": lambda doc: tokenize(doc, cfg),
        "preprocessor": lambda doc: doc,
        "lowercase": False,
        "token_pattern": None,
        "ngram_range": (int(cfg.get("ngram_min", 1)), int(cfg.get("ngram_max", 1))),
        "min_df": cfg.get("min_df", 1),
        "max_df": cfg.get("max_df", 1.0),
        "max_features": cfg.get("max_features"),
    }


def build_count_vectorizer(
    config: dict[str, Any] | None = None, **overrides: Any
) -> CountVectorizer:
    """Build a ``CountVectorizer`` whose tokenizer is :func:`tokenize`."""
    kwargs = _vectorizer_kwargs(config)
    kwargs.update(overrides)
    return CountVectorizer(**kwargs)


def build_tfidf_vectorizer(
    config: dict[str, Any] | None = None, **overrides: Any
) -> TfidfVectorizer:
    """Build a ``TfidfVectorizer`` whose tokenizer is :func:`tokenize`."""
    kwargs = _vectorizer_kwargs(config)
    kwargs.update(overrides)
    return TfidfVectorizer(**kwargs)


def build_hashing_vectorizer(
    config: dict[str, Any] | None = None,
    *,
    n_features: int = 2**18,
    alternate_sign: bool = True,
    norm: str | None = None,
    **overrides: Any,
) -> HashingVectorizer:
    """Build a ``HashingVectorizer`` for huge exploratory corpora (no vocabulary).

    HashingVectorizer does not support ``min_df`` / ``max_df`` / ``max_features``
    the same way CountVectorizer does — those keys are dropped deliberately.
    """
    kwargs = _vectorizer_kwargs(config)
    for key in ("min_df", "max_df", "max_features"):
        kwargs.pop(key, None)
    kwargs.update(
        {
            "n_features": int(n_features),
            "alternate_sign": alternate_sign,
            "norm": norm,
            "dtype": "float64",
        }
    )
    kwargs.update(overrides)
    return HashingVectorizer(**kwargs)


def _char_vectorizer_kwargs(
    config: dict[str, Any] | None,
    *,
    ngram_min: int,
    ngram_max: int,
    min_df: float | int,
    max_df: float | int,
    max_features: int | None,
) -> dict[str, Any]:
    """Build kwargs for a character-n-gram vectorizer.

    Character n-grams bypass word tokenization (``analyzer="char_wb"``), but
    still benefit from the same normalization pipeline (encoding fix, Unicode
    normalization, whitespace normalization, lowercasing) applied by
    :func:`tokenize`. Word boundaries are preserved by re-joining tokens with
    a single space before character n-grams are sliced, so ``char_wb`` pads
    n-grams at word edges rather than spanning arbitrary whitespace runs.
    """
    cfg = _merge_config(config)

    def _normalize(doc: str) -> str:
        return " ".join(tokenize(doc, cfg))

    return {
        "preprocessor": _normalize,
        "lowercase": False,
        "analyzer": "char_wb",
        "ngram_range": (int(ngram_min), int(ngram_max)),
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
    }


def build_char_count_vectorizer(
    config: dict[str, Any] | None = None,
    *,
    ngram_min: int = 3,
    ngram_max: int = 5,
    min_df: float | int = 1,
    max_df: float | int = 1.0,
    max_features: int | None = None,
    **overrides: Any,
) -> CountVectorizer:
    """Build a character-n-gram ``CountVectorizer`` (word-boundary aware)."""
    kwargs = _char_vectorizer_kwargs(
        config,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    kwargs.update(overrides)
    return CountVectorizer(**kwargs)


def build_char_tfidf_vectorizer(
    config: dict[str, Any] | None = None,
    *,
    ngram_min: int = 3,
    ngram_max: int = 5,
    min_df: float | int = 1,
    max_df: float | int = 1.0,
    max_features: int | None = None,
    **overrides: Any,
) -> TfidfVectorizer:
    """Build a character-n-gram ``TfidfVectorizer`` (word-boundary aware)."""
    kwargs = _char_vectorizer_kwargs(
        config,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    kwargs.update(overrides)
    return TfidfVectorizer(**kwargs)
