import unittest

from backend.lib.vectors import can_index_embedding, cosine_similarity, vector_literal


class VectorUtilsTest(unittest.TestCase):
    def test_vector_literal_format(self):
        self.assertEqual(vector_literal([1.0, 0.5]), "[1.00000000,0.50000000]")

    def test_can_index_requires_matching_dimensions(self):
        self.assertTrue(can_index_embedding([0.1] * 1536, expected_dimensions=1536))
        self.assertFalse(can_index_embedding([0.1] * 32, expected_dimensions=1536))

    def test_cosine_similarity_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
