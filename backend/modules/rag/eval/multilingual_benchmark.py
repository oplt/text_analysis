"""Measured PostgreSQL lexical baseline versus experimental Turkish normalization."""

from __future__ import annotations

import re
import unicodedata

import simplemma
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

CASES = (
    ("sorumluluğun", "sorumluluk", True),
    ("sorumluluklar", "sorumluluk", True),
    ("hesap verebilirliğin", "hesap verebilirlik", True),
    ("katılım", "katılımcı", False),
    ("kitaplardan", "kitap", True),
    ("araştırmalar", "araştırma", True),
    ("çalışmalar", "çalışma", True),
    ("kurumların", "kurum", True),
    ("katılımcılar", "katılımcı", True),
    ("İSTANBUL", "istanbul", True),
    ("ışık", "ışık", True),
    ("kurum", "kitap", False),
    ("katılımcı", "hesap", False),
    ("merkezi denetim", "yerel katılım", False),
)


def experimental_turkish_normalization(content: str) -> str:
    normalized = unicodedata.normalize("NFC", content).replace("I", "ı").replace("İ", "i").lower()
    return " ".join(
        simplemma.lemmatize(token, lang="tr") for token in re.findall(r"\w+", normalized)
    )


async def benchmark_turkish_lexical(db: AsyncSession) -> dict:
    rows = []
    for document, query, expected in CASES:
        results = {}
        for strategy, transform in (
            ("simple", lambda value: value),
            ("simplemma_experimental", experimental_turkish_normalization),
        ):
            result = await db.execute(
                text(
                    "SELECT to_tsvector('simple', :document) @@ plainto_tsquery('simple', :query)"
                ),
                {"document": transform(document), "query": transform(query)},
            )
            results[strategy] = bool(result.scalar_one())
        rows.append({"document": document, "query": query, "expected": expected, **results})
    return {
        "measurement": "MEASURED",
        "cases": rows,
        "default_changed": False,
        "reason": (
            "Small morphology fixture does not establish domain precision; "
            "normalization remains experimental"
        ),
        "accuracy": {
            strategy: sum(row[strategy] == row["expected"] for row in rows) / len(rows)
            for strategy in ("simple", "simplemma_experimental")
        },
    }
