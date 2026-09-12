"""Language metadata to PostgreSQL FTS configuration mapping."""

from __future__ import annotations

POSTGRES_TEXT_SEARCH_CONFIGS = {
    "en": "english",
    "de": "german",
    "fr": "french",
}


def normalized_language(language: str | None) -> str | None:
    if not language:
        return None
    value = language.strip().lower().replace("_", "-")
    return value.split("-", 1)[0] or None


def text_search_config_for_language(language: str | None) -> str:
    """Use a language stemmer only where PostgreSQL has a supported config."""
    return POSTGRES_TEXT_SEARCH_CONFIGS.get(normalized_language(language), "simple")
