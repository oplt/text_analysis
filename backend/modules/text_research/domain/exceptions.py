from __future__ import annotations


class TextResearchError(Exception):
    """Base exception for the text_research module."""


class CodebookFrozenError(TextResearchError):
    """Raised when attempting to mutate a frozen codebook."""


class InsufficientDataError(TextResearchError):
    """Raised when a computation lacks enough data to produce a meaningful result."""


class RRuntimeUnavailable(TextResearchError):
    """Raised when a request selects R but the controlled runtime is disabled."""


class RExecutionTimeout(TextResearchError):
    """Raised when the R subprocess exceeds its configured time budget."""


class RExecutionFailed(TextResearchError):
    """Raised when the R subprocess reports a safe, bounded failure."""


class RInvalidResult(TextResearchError):
    """Raised when R output fails the canonical result validation contract."""


class RUnsupportedAnalysis(TextResearchError):
    """Raised for an analysis unavailable in the versioned R implementation."""


class RArtifactTooLarge(TextResearchError):
    """Raised when the R result exceeds configured output bounds."""
