import unittest

from backend.core.text_snippet import DEFAULT_SNIPPET_LENGTH, text_snippet


class TextSnippetTest(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(text_snippet("hello"), "hello")

    def test_long_text_truncated_with_ellipsis(self):
        content = "a" * (DEFAULT_SNIPPET_LENGTH + 50)
        snippet = text_snippet(content)
        self.assertLessEqual(len(snippet), DEFAULT_SNIPPET_LENGTH)
        self.assertTrue(snippet.endswith("…"))


if __name__ == "__main__":
    unittest.main()
