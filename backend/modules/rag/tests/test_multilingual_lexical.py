"""Language-aware lexical retrieval retains a Unicode-safe simple fallback."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from backend.modules.rag.infrastructure.lexical_languages import text_search_config_for_language
from backend.modules.rag.infrastructure.repositories import RagRepository


class MultilingualLexicalTests(unittest.IsolatedAsyncioTestCase):
    def test_maps_supported_languages_and_keeps_simple_fallback(self):
        cases = [
            ("English evidence", "en", "english"),
            ("Deutsche Überprüfung", "de-DE", "german"),
            ("Türkçe çalışma: ç ğ ı ö ş ü", "tr", "simple"),
            ("Français: coopération", "fr-FR", "french"),
        ]

        for _query, language, expected_config in cases:
            with self.subTest(language=language):
                self.assertEqual(text_search_config_for_language(language), expected_config)

    async def test_lexical_sql_uses_language_config_and_simple_fallback(self):
        db = MagicMock()
        result = MagicMock()
        result.mappings.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result)
        repo = RagRepository(db)
        repo._retrieval_scope_filters = MagicMock(return_value=["TRUE"])

        await repo.lexical_search(
            user_id="user-1",
            project_id=None,
            document_ids=None,
            query="Türkçe çalışma",
            top_k=5,
        )

        sql = str(db.execute.await_args.args[0])
        self.assertIn("WHEN 'en' THEN 'english'", sql)
        self.assertIn("WHEN 'de' THEN 'german'", sql)
        self.assertIn("WHEN 'fr' THEN 'french'", sql)
        self.assertIn("plainto_tsquery('simple', :query)", sql)
