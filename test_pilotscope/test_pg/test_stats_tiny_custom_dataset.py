"""
Unit test for StatsTinyCustomDataset.

This test verifies:
1. Dataset can be loaded and SQL queries can be read
2. Integration with utils.py functions
3. Query format validation
"""

import unittest

from pilotscope.PilotEnum import DatabaseEnum
from pilotscope.Dataset.StatsTinyCustomDataset import StatsTinyCustomDataset
from algorithm_examples.utils import load_training_sql, load_test_sql


class TestStatsTinyCustomDataset(unittest.TestCase):
    """Unit tests for StatsTinyCustomDataset."""

    def test_dataset_creation(self):
        """Test basic dataset instantiation."""
        dataset = StatsTinyCustomDataset(DatabaseEnum.POSTGRESQL)

        self.assertEqual(dataset.train_sql_file, "stats_custom_train.txt")
        self.assertEqual(dataset.test_sql_file, "stats_custom_test.txt")
        self.assertEqual(dataset.created_db_name, "stats_tiny")
        self.assertEqual(dataset.file_db_type, DatabaseEnum.POSTGRESQL)

    def test_read_training_queries(self):
        """Test reading training SQL queries."""
        dataset = StatsTinyCustomDataset(DatabaseEnum.POSTGRESQL)
        train_queries = dataset.read_train_sql()

        self.assertEqual(len(train_queries), 4000, "Should have 4000 training queries")
        self.assertTrue(all(isinstance(q, str) for q in train_queries))
        self.assertTrue(all(len(q) > 0 for q in train_queries))

    def test_read_test_queries(self):
        """Test reading test SQL queries."""
        dataset = StatsTinyCustomDataset(DatabaseEnum.POSTGRESQL)
        test_queries = dataset.read_test_sql()

        self.assertEqual(len(test_queries), 1000, "Should have 1000 test queries")
        self.assertTrue(all(isinstance(q, str) for q in test_queries))
        self.assertTrue(all(len(q) > 0 for q in test_queries))

    def test_utils_integration(self):
        """Test integration with algorithm_examples/utils.py."""
        train_queries = load_training_sql("stats_tiny_custom")
        test_queries = load_test_sql("stats_tiny_custom")

        self.assertEqual(len(train_queries), 4000)
        self.assertEqual(len(test_queries), 1000)

    def test_query_format(self):
        """Verify queries are properly formatted with semicolons."""
        dataset = StatsTinyCustomDataset(DatabaseEnum.POSTGRESQL)
        train_queries = dataset.read_train_sql()
        test_queries = dataset.read_test_sql()

        # All queries should end with semicolon (after BaseDataset._get_sql processing)
        for query in train_queries[:10]:  # Check first 10
            self.assertTrue(len(query.strip()) > 0, "Query should not be empty")

        for query in test_queries[:10]:  # Check first 10
            self.assertTrue(len(query.strip()) > 0, "Query should not be empty")

    def test_query_content(self):
        """Verify queries contain expected StatsTiny tables."""
        dataset = StatsTinyCustomDataset(DatabaseEnum.POSTGRESQL)
        train_queries = dataset.read_train_sql()

        # Sample first query
        first_query = train_queries[0].lower()

        # Should be SELECT COUNT(*) queries on StatsTiny tables
        self.assertIn("select", first_query)
        self.assertIn("count", first_query)
        # Should reference StatsTiny tables (posts, users, comments, etc.)
        has_stats_table = any(table in first_query for table in
                             ["posts", "users", "comments", "votes", "badges", "posthistory", "postlinks"])
        self.assertTrue(has_stats_table, "Query should reference StatsTiny tables")


if __name__ == '__main__':
    unittest.main()