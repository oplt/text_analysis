"""Deterministic, language-preserving rewrites for research follow-up retrieval."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")
_MAX_ANCHOR_CHARS = 240
_MAX_FOLLOW_UP_CHARS = 220


class QueryRewriteService:
    """Build a compact retrieval query without copying assistant prose."""

    def rewrite(
        self,
        *,
        follow_up: str,
        prior_turns: list[dict[str, str]],
    ) -> str:
        current = self._compact(follow_up)
        prior_user = next(
            (
                self._compact(turn.get("content", ""))
                for turn in reversed(prior_turns)
                if turn.get("role", "").lower() == "user" and self._compact(turn.get("content", ""))
            ),
            "",
        )
        if not prior_user:
            return current
        if not current:
            return prior_user
        return f"{prior_user[:_MAX_ANCHOR_CHARS]} {current[:_MAX_FOLLOW_UP_CHARS]}".strip()

    @staticmethod
    def _compact(value: str) -> str:
        return _WHITESPACE_RE.sub(" ", (value or "").strip())
