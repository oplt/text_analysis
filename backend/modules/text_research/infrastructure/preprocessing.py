"""Deterministic text preprocessing for Policy Text Lab.

Implements whitespace normalization, tokenization, stopword removal,
Snowball English stemming, and negation preservation, plus factory
functions that build scikit-learn ``CountVectorizer`` /
``TfidfVectorizer`` instances that route through the same tokenizer.

Original text is never mutated; all transformations operate on copies /
derived token lists.

Lemmatization is not implemented. Configurations that request it are
rejected so profiles never claim lemmatization occurred when it did not.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import snowballstemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#: Default preprocessing configuration, matching the PreprocessingProfile
#: example in the Policy Text Lab specification.
DEFAULT_PREPROCESSING_CONFIG: dict[str, Any] = {
    "lowercase": True,
    "remove_punctuation": True,
    "remove_numbers": False,
    "remove_stopwords": False,
    "preserve_negation": True,
    "stemming": False,
    "lemmatization": False,
    "ngram_min": 1,
    "ngram_max": 1,
    "min_df": 1,
    "max_df": 1.0,
    "max_features": None,
}

LEMMATIZATION_UNSUPPORTED_MESSAGE = (
    "Lemmatization is not implemented. Set lemmatization to false "
    "(or omit it) until a real lemmatizer is available."
)

_ENGLISH_STEMMER = snowballstemmer.stemmer("english")


@dataclass
class PreprocessingConfig:
    """Serializable preprocessing profile used by analysis services."""

    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = False
    preserve_negation: bool = True
    stemming: bool = False
    lemmatization: bool = False
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
        if merged.get("lemmatization"):
            raise ValueError(LEMMATIZATION_UNSUPPORTED_MESSAGE)
        merged["lemmatization"] = False
        field_names = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: merged[key] for key in field_names if key in merged})

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lemmatization"] = False
        return payload


#: Negation words that must never be silently dropped when preserve_negation
#: is enabled (the default). These are deliberately excluded from
#: STOPWORDS below.
NEGATION_WORDS: frozenset[str] = frozenset({"not", "no", "never"})

#: Minimal English stopword list (~50 common function words), excluding
#: negation words on purpose. This is not meant to be exhaustive (e.g.
#: NLTK's list); it is a small, auditable, dependency-free set suitable for
#: a research demo.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "as",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "we",
        "our",
        "you",
        "your",
        "i",
        "my",
        "me",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "shall",
        "may",
        "might",
        "must",
        "so",
        "than",
        "too",
        "very",
        "just",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
    }
)

assert STOPWORDS.isdisjoint(NEGATION_WORDS)

# Word tokens: letters/digits with optional internal apostrophe (contractions).
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"^[0-9]+([.,][0-9]+)*$")


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to a single space and strip ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _is_number_token(token: str) -> bool:
    return bool(_NUMBER_RE.match(token))


def snowball_stem(token: str) -> str:
    """Stem ``token`` with the Snowball English stemmer."""
    return _ENGLISH_STEMMER.stemWord(token)


def simple_stem(token: str) -> str:
    """Compatibility alias for :func:`snowball_stem`."""
    return snowball_stem(token)


def _merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_PREPROCESSING_CONFIG)
    if config:
        merged.update(config)
    if merged.get("lemmatization"):
        raise ValueError(LEMMATIZATION_UNSUPPORTED_MESSAGE)
    merged["lemmatization"] = False
    return merged


def _stopword_set(config: dict[str, Any]) -> frozenset[str]:
    """Build the effective stopword set for the given config.

    ``preserve_negation`` acts as a hard guard: even if custom stopwords or
    future stopword-list extensions include "not"/"no"/"never", they are
    never removed while preserve_negation is True.
    """
    stopwords = set(STOPWORDS)
    stopwords.update(config.get("custom_stopwords") or [])
    if config.get("preserve_negation", True):
        stopwords -= NEGATION_WORDS
    return frozenset(stopwords)


def tokenize(text: str, config: dict[str, Any] | None = None) -> list[str]:
    """Tokenize ``text`` according to ``config``.

    Pipeline: whitespace normalization -> optional lowercasing -> word
    extraction (optionally punctuation-preserving) -> optional number
    removal -> optional stopword removal (negation-safe) -> optional
    Snowball stemming (negation-safe).
    """
    cfg = _merge_config(config)
    negation_words = NEGATION_WORDS if cfg.get("preserve_negation", True) else frozenset()

    working = normalize_whitespace(text)
    if cfg.get("lowercase", True):
        working = working.lower()

    if cfg.get("remove_punctuation", True):
        tokens = _TOKEN_RE.findall(working)
    else:
        tokens = working.split(" ")
        tokens = [t for t in tokens if t]

    if cfg.get("remove_numbers", False):
        tokens = [t for t in tokens if t in negation_words or not _is_number_token(t)]

    if cfg.get("remove_stopwords", False):
        stop_set = _stopword_set(cfg)
        tokens = [t for t in tokens if t in negation_words or t not in stop_set]

    if cfg.get("stemming", False):
        tokens = [t if t in negation_words else snowball_stem(t) for t in tokens]

    return tokens


def preprocess_text(text: str, config: dict[str, Any] | None = None) -> str:
    """Return the space-joined token sequence for ``text``.

    The original ``text`` is never mutated; this returns a new string.
    """
    return " ".join(tokenize(text, config))


def _baseline_tokens(text: str, config: dict[str, Any]) -> list[str]:
    """Tokens before stopword/number removal and stemming (for preview diffs)."""
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
    """Build a live preprocessing preview for sample texts.

    Returns original/processed pairs, token counts before/after, vocabulary
    size after processing, and the most frequently removed baseline terms.
    """
    cfg = _merge_config(config)
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
        "stemmer": "snowball_english",
        "lemmatization_supported": False,
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
    """Build a ``CountVectorizer`` whose tokenizer is :func:`tokenize`.

    ``ngram_range``/``min_df``/``max_df``/``max_features`` come from
    ``config`` (falling back to :data:`DEFAULT_PREPROCESSING_CONFIG`).
    """
    kwargs = _vectorizer_kwargs(config)
    kwargs.update(overrides)
    return CountVectorizer(**kwargs)


def build_tfidf_vectorizer(
    config: dict[str, Any] | None = None, **overrides: Any
) -> TfidfVectorizer:
    """Build a ``TfidfVectorizer`` whose tokenizer is :func:`tokenize`.

    ``ngram_range``/``min_df``/``max_df``/``max_features`` come from
    ``config`` (falling back to :data:`DEFAULT_PREPROCESSING_CONFIG`).
    """
    kwargs = _vectorizer_kwargs(config)
    kwargs.update(overrides)
    return TfidfVectorizer(**kwargs)
