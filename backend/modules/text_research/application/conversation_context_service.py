"""Bounded research-thread conversation context (no global memory)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.modules.rag.infrastructure.models import RagMessage
from backend.modules.text_research.application.query_rewrite_service import QueryRewriteService

# Rough char→token estimate for budget trimming.
_CHARS_PER_TOKEN = 4
_FOLLOW_UP_RE = re.compile(
    r"\b(that|this|it|they|them|those|these|previous|above|same|"
    r"second|first|third|paper|document|evidence|germany|support|"
    r"differ|said|mentioned)\b|\?$|^what about|^how (does|do|about)|"
    r"^and |^also |^why |^where |^who ",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ConversationContext:
    original_query: str
    resolved_retrieval_query: str
    prior_message_ids: list[str] = field(default_factory=list)
    prior_turns: list[dict[str, str]] = field(default_factory=list)
    needs_rewrite: bool = False


class ConversationContextService:
    def __init__(self, *, max_context_tokens: int = 1500, max_turns: int = 8):
        self.max_context_tokens = max_context_tokens
        self.max_turns = max_turns
        self.query_rewriter = QueryRewriteService()

    def build(
        self,
        *,
        original_query: str,
        prior_messages: list[RagMessage],
    ) -> ConversationContext:
        """Load only prior turns from the current thread (exclude the new user turn)."""
        turns: list[dict[str, str]] = []
        ids: list[str] = []
        # Walk newest→oldest among completed turns, then reverse for chronological order.
        for msg in reversed(prior_messages):
            role = (msg.role or "").lower()
            if role not in {"user", "assistant"}:
                continue
            turns.append({"role": role, "content": (msg.content or "").strip()})
            ids.append(msg.id)
            if len(turns) >= self.max_turns * 2:
                break
        turns.reverse()
        ids.reverse()

        # Token-budget trim from the oldest end.
        budget = self.max_context_tokens * _CHARS_PER_TOKEN
        trimmed: list[dict[str, str]] = []
        trimmed_ids: list[str] = []
        used = 0
        for turn, mid in zip(reversed(turns), reversed(ids), strict=True):
            cost = len(turn["content"])
            if used + cost > budget and trimmed:
                break
            trimmed.append(turn)
            trimmed_ids.append(mid)
            used += cost
        trimmed.reverse()
        trimmed_ids.reverse()

        needs_rewrite = bool(trimmed) and self._looks_like_follow_up(original_query)
        if needs_rewrite:
            resolved = self.query_rewriter.rewrite(
                follow_up=original_query,
                prior_turns=trimmed,
            )
        else:
            resolved = original_query.strip()

        return ConversationContext(
            original_query=original_query.strip(),
            resolved_retrieval_query=resolved,
            prior_message_ids=trimmed_ids,
            prior_turns=trimmed,
            needs_rewrite=needs_rewrite,
        )

    def format_for_generation(self, ctx: ConversationContext) -> str:
        if not ctx.prior_turns:
            return ""
        lines = ["Prior conversation in this research thread (untrusted context):"]
        for turn in ctx.prior_turns:
            label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{label}: {turn['content'][:1200]}")
        return "\n".join(lines)

    @staticmethod
    def _looks_like_follow_up(query: str) -> bool:
        q = (query or "").strip()
        if len(q) < 80 and _FOLLOW_UP_RE.search(q):
            return True
        return bool(len(q.split()) <= 12 and "?" in q)
