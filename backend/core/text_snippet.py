DEFAULT_SNIPPET_LENGTH = 400


def text_snippet(text: str, *, max_length: int = DEFAULT_SNIPPET_LENGTH) -> str:
    normalized = text.strip()
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"
