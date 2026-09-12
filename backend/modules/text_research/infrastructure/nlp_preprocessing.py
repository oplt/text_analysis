"""Optional spaCy-backed NLP preprocessing helpers (§19).

Classical Unicode preprocessing remains the default. These helpers only run
when explicitly enabled in :class:`PreprocessingConfig` and require optional
spaCy + a loaded language model. Missing dependencies raise ``ValueError``
rather than silently claiming NER/lemma/phrase output.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

DEFAULT_SPACY_MODEL = "en_core_web_sm"


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def spacy_available(model_name: str = DEFAULT_SPACY_MODEL) -> bool:
    """Return True when spaCy + ``model_name`` can be loaded via the process cache.

    Always routes through :func:`_load_nlp` so availability checks share the same
    cached pipeline as tokenization/provenance (TASK-011).
    """
    try:
        _load_nlp(model_name)
        return True
    except Exception:  # noqa: BLE001 — missing dep / model must stay False
        return False


@lru_cache(maxsize=8)
def _load_nlp(model_name: str) -> Any:
    try:
        import spacy
    except ImportError as exc:
        raise ValueError(
            "spaCy is required for this preprocessing option; install with: pip install '.[nlp]'"
        ) from exc
    try:
        return spacy.load(model_name)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"spaCy model {model_name!r} is unavailable: {exc}. "
            f"Install it explicitly (e.g. python -m spacy download {model_name})."
        ) from exc


def require_spacy(model_name: str = DEFAULT_SPACY_MODEL) -> Any:
    """Load spaCy or raise a clear ValueError."""
    return _load_nlp(model_name)


def describe_spacy_provenance(model_name: str = DEFAULT_SPACY_MODEL) -> dict[str, Any]:
    available = spacy_available(model_name)
    meta: dict[str, Any] = {
        "available": available,
        "model_name": model_name,
        "package": "spacy",
        "package_version": _pkg_version("spacy"),
        "model_version": None,
        "model_lang": None,
    }
    if not available:
        return meta
    try:
        nlp = require_spacy(model_name)
        meta["model_version"] = (
            getattr(nlp.meta, "get", lambda *_: None)("version") if hasattr(nlp, "meta") else None
        )
        if isinstance(getattr(nlp, "meta", None), dict):
            meta["model_version"] = nlp.meta.get("version")
            meta["model_lang"] = nlp.meta.get("lang")
            meta["pipeline"] = list(nlp.pipe_names)
    except ValueError:
        meta["available"] = False
    return meta


def mask_named_entities(text: str, *, model_name: str = DEFAULT_SPACY_MODEL) -> str:
    """Replace named entities with ``ENT_<LABEL>`` placeholders (order preserved)."""
    nlp = require_spacy(model_name)
    doc = nlp(text or "")
    if not doc.ents:
        return text
    pieces: list[str] = []
    last = 0
    for ent in doc.ents:
        pieces.append(text[last : ent.start_char])
        pieces.append(f"ENT_{ent.label_}")
        last = ent.end_char
    pieces.append(text[last:])
    return "".join(pieces)


def pos_aware_lemmas(text: str, *, model_name: str = DEFAULT_SPACY_MODEL) -> list[str]:
    """Tokenize with spaCy and return POS-informed lemmas (non-space, non-punct)."""
    nlp = require_spacy(model_name)
    doc = nlp(text or "")
    lemmas: list[str] = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.strip()
        if lemma:
            lemmas.append(lemma)
    return lemmas


def noun_chunk_phrases(
    text: str,
    *,
    model_name: str = DEFAULT_SPACY_MODEL,
    use_lemmas: bool = False,
) -> list[str]:
    """Return spaCy noun-chunk phrases (multi-token only), lowercased.

    When ``use_lemmas`` is True, phrase parts are spaCy lemmas so they align with
    POS-lemmatized token streams (TASK-012).
    """
    nlp = require_spacy(model_name)
    doc = nlp(text or "")
    phrases: list[str] = []
    for chunk in doc.noun_chunks:
        parts: list[str] = []
        for token in chunk:
            if token.is_space or token.is_punct:
                continue
            part = (token.lemma_ if use_lemmas else token.text).strip()
            if part:
                parts.append(part.lower())
        if len(parts) >= 2:
            phrases.append("_".join(parts))
    return phrases


def morph_normalize_phrase_parts(
    phrases: list[str],
    *,
    normalize_part,
) -> list[str]:
    """Apply ``normalize_part`` to each underscore-separated phrase component."""
    if not phrases:
        return phrases
    out: list[str] = []
    for phrase in phrases:
        parts = [normalize_part(part) for part in phrase.split("_")]
        if len(parts) >= 2:
            out.append("_".join(parts))
    return out


def merge_phrase_tokens(
    tokens: list[str],
    phrases: list[str],
) -> list[str]:
    """Greedily replace contiguous token sequences matching known phrases."""
    if not phrases or not tokens:
        return tokens
    lowered = [t.lower() for t in tokens]
    phrase_parts = sorted(
        (p.split("_") for p in phrases if "_" in p),
        key=lambda parts: len(parts),
        reverse=True,
    )
    if not phrase_parts:
        return tokens
    out: list[str] = []
    i = 0
    n = len(lowered)
    while i < n:
        matched = False
        for parts in phrase_parts:
            length = len(parts)
            if i + length <= n and lowered[i : i + length] == parts:
                out.append("_".join(tokens[i : i + length]).lower())
                i += length
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1
    return out


def extract_entities_for_provenance(
    texts: list[str],
    *,
    model_name: str = DEFAULT_SPACY_MODEL,
    unit_ids: list[str] | None = None,
    max_entities: int = 500,
) -> dict[str, Any]:
    """Run NER and return a compact provenance payload (capped)."""
    from backend.modules.text_research.infrastructure.ner import extract_entities

    payload = extract_entities(texts, model_name=model_name, unit_ids=unit_ids)
    entities = list(payload.get("entities") or [])[:max_entities]
    return {
        **payload,
        "entities": entities,
        "truncated": len(payload.get("entities") or []) > max_entities,
        "spacy": describe_spacy_provenance(model_name),
    }
