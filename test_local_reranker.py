"""Unit tests for LocalReranker.

This script tests the LocalReranker class using unittest, mimicking test_remote_reranker.py.
"""

import os
import sys
import unittest
import requests
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from EasyRerank import LocalReranker


class TestLocalRerankerUnit(unittest.TestCase):
    """Offline unit tests for LocalReranker using mock responses."""

    def setUp(self):
        self.reranker = LocalReranker()

    def test_init_defaults(self):
        """Test initialization with default arguments."""
        self.assertEqual(self.reranker.host, "localhost")
        self.assertEqual(self.reranker.port, 8080)
        self.assertIsNone(self.reranker.model)

    def test_validate_batch_size(self):
        """Test batch size validation logic."""
        self.assertEqual(self.reranker._validate_batch_size(16), 16)
        self.assertEqual(self.reranker._validate_batch_size(32), 32)
        self.assertEqual(self.reranker._validate_batch_size(64), 64)
        
        with self.assertRaises(ValueError):
            self.reranker._validate_batch_size(5)

    def test_batch_documents(self):
        """Test document batching logic."""
        docs = [f"doc {i}" for i in range(10)]
        batches_3 = self.reranker._batch_documents(docs, 3)
        self.assertEqual(len(batches_3), 4)

    @patch('requests.post')
    def test_call_rerank_api_success(self, mock_post):
        """Test successful single API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.15}
            ]
        }
        mock_post.return_value = mock_response

        query = "test query"
        docs = ["doc1", "doc2"]
        result = self.reranker._call_rerank_api(query, docs, top_n=2)

        mock_post.assert_called_once_with(
            "http://localhost:8080/v1/rerank",
            json={
                'query': query,
                'documents': docs,
                'top_n': 2
            },
            timeout=120
        )
        self.assertIn("results", result)

    @patch('requests.post')
    def test_rerank_multi_batch_index_mapping(self, mock_post):
        """Test batch indices mapping with 17 documents and batch_size = 16."""
        # Batch 1 response (16 docs)
        mock_response_1 = MagicMock()
        mock_response_1.json.return_value = {
            "results": [
                {"index": i, "relevance_score": 0.05 * i}
                for i in range(16)
            ]
        }
        # Batch 2 response (1 doc)
        mock_response_2 = MagicMock()
        mock_response_2.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.99}
            ]
        }
        
        mock_post.side_effect = [mock_response_1, mock_response_2]

        docs = [f"doc {i}" for i in range(17)]
        results = self.reranker.rerank("query", docs, batch_size=16)

        # Expected global indexes:
        # doc 16: index 16 (batch 2, index 0 offset by 16), score 0.99
        # doc 15: index 15 (batch 1, index 15), score 0.75
        self.assertEqual(results[0]["index"], 16)
        self.assertEqual(results[0]["relevance_score"], 0.99)
        self.assertEqual(results[0]["document"]["text"], "doc 16")
        
        self.assertEqual(results[1]["index"], 15)
        self.assertEqual(results[1]["document"]["text"], "doc 15")



if __name__ == "__main__":
    unittest.main()
