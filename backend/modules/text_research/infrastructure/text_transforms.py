"""Composable string- and token-level text transforms for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import ftfy

from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    apply_unicode_normalization,
    lemmatize_token,
    normalize_whitespace,
    snowball_stem,
)
from backend.modules.text_research.infrastructure.preprocessing import (
    tokenize as full_tokenize,
)

TRANSFORM_VERSION = "1"


@runtime_checkable
class TextTransform(Protocol):
    """String- or token-stage preprocessing transform."""

    name: str
    version: str

    def transform(self, text: str, context: dict[str, Any]) -> str: ...

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]: ...


def _cfg(context: dict[str, Any]) -> dict[str, Any]:
    config = context.get("config")
    if isinstance(config, PreprocessingConfig):
        return config.to_dict()
    if isinstance(config, dict):
        return config
    return context


@dataclass
class UnicodeNormalize:
    name: str = "unicode_normalize"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any]) -> str:
        cfg = _cfg(context)
        return apply_unicode_normalization(text, cfg.get("unicode_normalization"))

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]:
        return tokens


@dataclass
class FixEncoding:
    name: str = "fix_encoding"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any]) -> str:
        return ftfy.fix_text(text)

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]:
        return tokens


@dataclass
class NormalizeWhitespace:
    name: str = "normalize_whitespace"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any]) -> str:
        return normalize_whitespace(text)

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]:
        return tokens


@dataclass
class Lowercase:
    name: str = "lowercase"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any]) -> str:
        return text.lower()

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]:
        return [t.lower() for t in tokens]


@dataclass
class TokenizeTransform:
    """Split cleaned text into tokens (no stopword/stem/lemma filters)."""

    name: str = "tokenize"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any]) -> str:
        return " ".join(self.tokenize_text(text, context))

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]:
        return tokens

    def tokenize_text(self, text: str, context: dict[str, Any]) -> list[str]:
        cfg = dict(_cfg(context))
        cfg.update(
            {
                "remove_stopwords": False,
                "stemming": False,
                "lemmatization": False,
            }
        )
        return full_tokenize(text, cfg)


@dataclass
class RemoveStopwordsTransform:
    name: str = "remove_stopwords"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any]) -> str:
        tokens = text.split()
        return " ".join(self.transform_tokens(tokens, context))

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]:
        cfg = dict(_cfg(context))
        cfg["remove_stopwords"] = True
        cfg["stemming"] = False
        cfg["lemmatization"] = False
        joined = " ".join(tokens)
        tokenized = full_tokenize(joined, cfg)
        return tokenized


@dataclass
class StemTransform:
    name: str = "stem"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any]) -> str:
        return " ".join(self.transform_tokens(text.split(), context))

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]:
        cfg = _cfg(context)
        language = cfg.get("language") or "en"
        from backend.modules.text_research.infrastructure.language_processing import (
            negation_words_for,
        )

        negation_words = (
            negation_words_for(language) if cfg.get("preserve_negation", True) else frozenset()
        )
        return [t if t in negation_words else snowball_stem(t, language) for t in tokens]


@dataclass
class LemmatizeTransform:
    name: str = "lemmatize"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any]) -> str:
        return " ".join(self.transform_tokens(text.split(), context))

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]:
        cfg = _cfg(context)
        language = cfg.get("language") or "en"
        from backend.modules.text_research.infrastructure.language_processing import (
            negation_words_for,
        )

        negation_words = (
            negation_words_for(language) if cfg.get("preserve_negation", True) else frozenset()
        )
        return [t if t in negation_words else lemmatize_token(t, language) for t in tokens]


@dataclass
class Compose:
    """Apply a sequence of transforms in order."""

    steps: list[TextTransform] = field(default_factory=list)
    name: str = "compose"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        out = text
        for step in self.steps:
            if step.name == "tokenize":
                break
            out = step.transform(out, ctx)
        return out

    def transform_tokens(
        self, tokens: list[str], context: dict[str, Any] | None = None
    ) -> list[str]:
        ctx = context or {}
        out = tokens
        token_stage = False
        for step in self.steps:
            if step.name == "tokenize":
                token_stage = True
                continue
            if token_stage:
                out = step.transform_tokens(out, ctx)
        return out

    def describe(self) -> list[dict[str, str]]:
        return [{"name": step.name, "version": step.version} for step in self.steps]


def _string_steps_from_config(cfg: dict[str, Any]) -> list[TextTransform]:
    steps: list[TextTransform] = []
    if cfg.get("fix_encoding"):
        steps.append(FixEncoding())
    if cfg.get("unicode_normalization"):
        steps.append(UnicodeNormalize())
    steps.append(NormalizeWhitespace())
    if cfg.get("entity_masking"):
        steps.append(EntityMaskTransform())
    if cfg.get("lowercase", True) and not cfg.get("pos_lemmatization"):
        steps.append(Lowercase())
    return steps


@dataclass
class EntityMaskTransform:
    name: str = "entity_mask"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any]) -> str:
        cfg = _cfg(context)
        if not cfg.get("entity_masking"):
            return text
        from backend.modules.text_research.infrastructure.nlp_preprocessing import (
            mask_named_entities,
        )

        return mask_named_entities(text, model_name=str(cfg.get("spacy_model") or "en_core_web_sm"))

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]:
        return tokens


@dataclass
class PosLemmatizeTransform:
    """spaCy POS-aware lemmatization used in place of classical tokenization."""

    name: str = "tokenize"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any]) -> str:
        return " ".join(self.tokenize_text(text, context))

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]:
        return tokens

    def tokenize_text(self, text: str, context: dict[str, Any]) -> list[str]:
        cfg = _cfg(context)
        from backend.modules.text_research.infrastructure.nlp_preprocessing import (
            pos_aware_lemmas,
        )

        lemmas = pos_aware_lemmas(text, model_name=str(cfg.get("spacy_model") or "en_core_web_sm"))
        if cfg.get("lowercase", True):
            lemmas = [t.lower() for t in lemmas]
        return lemmas


@dataclass
class PhraseDetectTransform:
    name: str = "phrase_detect"
    version: str = TRANSFORM_VERSION

    def transform(self, text: str, context: dict[str, Any]) -> str:
        return " ".join(self.transform_tokens(text.split(), context))

    def transform_tokens(self, tokens: list[str], context: dict[str, Any]) -> list[str]:
        cfg = _cfg(context)
        from backend.modules.text_research.infrastructure.nlp_preprocessing import (
            merge_phrase_tokens,
            noun_chunk_phrases,
        )

        phrases = noun_chunk_phrases(
            " ".join(tokens), model_name=str(cfg.get("spacy_model") or "en_core_web_sm")
        )
        return merge_phrase_tokens(tokens, phrases)


def build_default_text_pipeline(config: dict[str, Any] | PreprocessingConfig | None) -> Compose:
    """Build the default string-level pipeline mirroring ``PreprocessingConfig`` booleans."""
    if isinstance(config, PreprocessingConfig):
        cfg = config.to_dict()
    elif config:
        cfg = PreprocessingConfig.from_dict(config).to_dict()
    else:
        cfg = PreprocessingConfig().to_dict()
    return Compose(steps=_string_steps_from_config(cfg))


def build_full_pipeline(config: dict[str, Any] | PreprocessingConfig | None) -> Compose:
    """Build string stages, tokenization, then optional token filters."""
    if isinstance(config, PreprocessingConfig):
        cfg = config.to_dict()
    elif config:
        cfg = PreprocessingConfig.from_dict(config).to_dict()
    else:
        cfg = PreprocessingConfig().to_dict()

    steps = _string_steps_from_config(cfg)
    if cfg.get("pos_lemmatization"):
        steps.append(PosLemmatizeTransform())
    else:
        steps.append(TokenizeTransform())
        if cfg.get("remove_stopwords"):
            steps.append(RemoveStopwordsTransform())
        if cfg.get("lemmatization"):
            steps.append(LemmatizeTransform())
        elif cfg.get("stemming"):
            steps.append(StemTransform())
    if cfg.get("phrase_detection"):
        steps.append(PhraseDetectTransform())
    return Compose(steps=steps)


def apply_pipeline(
    texts: list[str],
    pipeline: Compose,
    *,
    context: dict[str, Any] | None = None,
) -> list[list[str]]:
    """Apply a full pipeline and return per-text token sequences."""
    ctx = dict(context or {})
    if "config" not in ctx and ctx.get("config") is None:
        pass

    tokenize_idx = next(
        (index for index, step in enumerate(pipeline.steps) if step.name == "tokenize"),
        len(pipeline.steps),
    )
    string_steps = pipeline.steps[:tokenize_idx]
    tokenize_step = pipeline.steps[tokenize_idx] if tokenize_idx < len(pipeline.steps) else None
    token_steps = pipeline.steps[tokenize_idx + 1 :]

    string_compose = Compose(steps=string_steps)
    results: list[list[str]] = []
    for text in texts:
        cleaned = string_compose.transform(text, ctx)
        if tokenize_step is None:
            results.append(cleaned.split())
            continue
        tokens = tokenize_step.tokenize_text(cleaned, ctx)  # type: ignore[attr-defined]
        for step in token_steps:
            tokens = step.transform_tokens(tokens, ctx)
        results.append(tokens)
    return results
