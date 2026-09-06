from __future__ import annotations

import re

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)ignore (all )?(previous|prior|system) (instructions|rules)"),
    re.compile(r"(?i)ignore (all )?(previous|prior) (system )?(instructions|rules)"),
    re.compile(r"(?i)you are now (in )?developer mode"),
    re.compile(r"(?i)reveal (the )?(secret|password|api[_-]?key|token)"),
    re.compile(r"(?i)override (system|developer|safety)"),
)


class RagPolicyService:
    def contains_prompt_injection(self, content: str) -> bool:
        return any(pattern.search(content) for pattern in INJECTION_PATTERNS)

    def is_allowed_file_type(self, filename: str, allowed: tuple[str, ...]) -> bool:
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return suffix in allowed
