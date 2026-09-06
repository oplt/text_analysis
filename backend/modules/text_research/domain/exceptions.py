from __future__ import annotations


class TextResearchError(Exception):
    """Base exception for the text_research module."""


class CodebookFrozenError(TextResearchError):
    """Raised when attempting to mutate a frozen codebook."""


class InsufficientDataError(TextResearchError):
    """Raised when a computation lacks enough data to produce a meaningful result."""
