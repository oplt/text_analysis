import unittest

from backend.core.pagination import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    PaginatedResponse,
    paginated_response,
)


class PaginationDefaultsTest(unittest.TestCase):
    def test_paginated_response_shape(self):
        page = paginated_response(
            ["a", "b"],
            total=10,
            limit=2,
            offset=4,
        )
        self.assertEqual(page.items, ["a", "b"])
        self.assertEqual(page.total, 10)
        self.assertEqual(page.limit, 2)
        self.assertEqual(page.offset, 4)

    def test_constants(self):
        self.assertEqual(DEFAULT_PAGE_LIMIT, 50)
        self.assertEqual(MAX_PAGE_LIMIT, 200)

    def test_generic_response_model(self):
        response = PaginatedResponse[str](
            items=["x"],
            total=1,
            limit=50,
            offset=0,
        )
        self.assertEqual(response.items, ["x"])
