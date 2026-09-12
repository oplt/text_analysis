from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.modules.rag.application.query_analysis import analyze_query
from backend.modules.rag.application.retrieval_planner import plan_retrieval
from backend.modules.rag.domain.enums import RetrievalIntent


class QueryProfileTests(unittest.TestCase):
    def test_rules_are_deterministic_and_inspectable(self):
        cases = [
            ('What is "social cohesion"?', RetrievalIntent.DEFINITION, True),
            (
                "Compare education outcomes in Finland versus Sweden",
                RetrievalIntent.COMPARISON,
                False,
            ),
            ("What evidence contradicts this finding?", RetrievalIntent.CONTRADICTION, False),
            ("Which sources support the finding?", RetrievalIntent.EVIDENCE, False),
            ("Who wrote this report in 2024?", RetrievalIntent.FACT, False),
            ("Welche Quelle widerspricht dem Ergebnis?", RetrievalIntent.CONTRADICTION, False),
            (
                "Vergleich Finnland gegenüber Schweden",
                RetrievalIntent.COMPARISON,
                False,
            ),
        ]
        for query, intent, phrase_boost in cases:
            with self.subTest(query=query):
                analysis = analyze_query(query)
                self.assertEqual(analysis.intent, intent)
                self.assertEqual(analysis.lexical_phrase_boost, phrase_boost)
                self.assertTrue(analysis.to_dict()["reasons"])

    def test_comparison_requires_two_entities(self):
        self.assertEqual(analyze_query("Compare the outcomes").intent, RetrievalIntent.EVIDENCE)

    def test_profile_is_bounded_and_records_phrase_boost(self):
        analysis = analyze_query('What is "social cohesion"?')
        profile = plan_retrieval(
            analysis.intent,
            SimpleNamespace(top_k=5),
            lexical_phrase_boost=analysis.lexical_phrase_boost,
        )
        self.assertTrue(profile.lexical_phrase_boost)
        self.assertLessEqual(profile.top_k, 5)
