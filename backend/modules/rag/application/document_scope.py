"""Helpers for document allow-list filter semantics (invariant I1).

`document_ids=None`  → no document filter (generic /rag/* behavior)
`document_ids=[]`    → empty allow-list → zero candidates (never widen)
`document_ids=[...]` → restrict to those IDs
"""

from __future__ import annotations


class EmptyDocumentAllowList(Exception):
    """Raised / used as a signal when an explicit empty allow-list short-circuits."""


def document_ids_is_empty_allow_list(document_ids: list[str] | None) -> bool:
    return document_ids is not None and len(document_ids) == 0


def should_apply_document_id_filter(document_ids: list[str] | None) -> bool:
    """True when SQL should include an IN (...) clause."""
    return document_ids is not None and len(document_ids) > 0
