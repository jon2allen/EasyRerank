"""Unit tests and integration tests for RemoteReranker.

This script tests the RemoteReranker class using unittest. It includes:
1. Unit tests with mocked API responses (runs anywhere, offline).
2. Live integration test (runs only if JINA_API_KEY environment variable is set).

Usage:
    python3 test_remote_reranker.py
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import requests

from remote_reranker import RemoteReranker


class TestRemoteRerankerUnit(unittest.TestCase):
    """Offline unit tests for RemoteReranker using mock responses."""

    def setUp(self):
        self.mock_api_key = "jina_test_key_12345"
        self.reranker = RemoteReranker(api_key=self.mock_api_key)

    def test_init_with_argument(self):
        """Test initialization with explicit API key."""
        reranker = RemoteReranker(api_key="explicit_key")
        self.assertEqual(reranker.api_key, "explicit_key")
        self.assertEqual(reranker.model, "jina-reranker-v3")

    @patch.dict(os.environ, {"JINA_API_KEY": "env_key"})
    def test_init_with_env_var(self):
        """Test initialization utilizing JINA_API_KEY environment variable."""
        reranker = RemoteReranker()
        self.assertEqual(reranker.api_key, "env_key")

    def test_init_missing_key_raises_value_error(self):
        """Test that initialization raises ValueError if no key is found."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                RemoteReranker()
            self.assertIn("Jina AI API key must be provided", str(context.exception))

    def test_validate_batch_size(self):
        """Test batch size validation logic."""
        self.assertEqual(self.reranker._validate_batch_size(16), 16)
        self.assertEqual(self.reranker._validate_batch_size(32), 32)
        self.assertEqual(self.reranker._validate_batch_size(64), 64)
        
        with self.assertRaises(ValueError):
            self.reranker._validate_batch_size(5)
            
        with self.assertRaises(ValueError):
            self.reranker._validate_batch_size(100)

    def test_batch_documents(self):
        """Test document batching logic."""
        docs = [f"doc {i}" for i in range(10)]
        
        # Batch size 3
        batches_3 = self.reranker._batch_documents(docs, 3)
        self.assertEqual(len(batches_3), 4)
        self.assertEqual(len(batches_3[0]), 3)
        self.assertEqual(len(batches_3[3]), 1)
        
        # Batch size 16
        batches_16 = self.reranker._batch_documents(docs, 16)
        self.assertEqual(len(batches_16), 1)
        self.assertEqual(len(batches_16[0]), 10)

    @patch('requests.post')
    def test_call_rerank_api_success(self, mock_post):
        """Test successful single API call with correct headers and payload."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95, "document": {"text": "doc1"}},
                {"index": 1, "relevance_score": 0.15, "document": {"text": "doc2"}}
            ]
        }
        mock_post.return_value = mock_response

        query = "test query"
        docs = ["doc1", "doc2"]
        result = self.reranker._call_rerank_api(query, docs, top_n=2)

        # Verify correct HTTP request arguments
        mock_post.assert_called_once_with(
            "https://api.jina.ai/v1/rerank",
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.mock_api_key}'
            },
            json={
                'model': 'jina-reranker-v3',
                'query': query,
                'documents': docs,
                'top_n': 2
            },
            timeout=120
        )
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 2)

    @patch('requests.post')
    def test_call_rerank_api_http_error(self, mock_post):
        """Test that API HTTP errors raise RequestException with details."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Bad Request")
        mock_response.json.return_value = {"detail": "Invalid model parameter"}
        mock_post.return_value = mock_response

        with self.assertRaises(requests.exceptions.RequestException) as context:
            self.reranker._call_rerank_api("query", ["doc"])
        self.assertIn("Invalid model parameter", str(context.exception))

    @patch('requests.post')
    def test_rerank_multi_batch_index_mapping(self, mock_post):
        """Test batch indices mapping with 17 documents and batch_size = 16."""
        # Batch 1 response (16 docs)
        mock_response_1 = MagicMock()
        mock_response_1.json.return_value = {
            "results": [
                {"index": i, "relevance_score": 0.05 * i, "document": {"text": f"doc {i}"}}
                for i in range(16)
            ]
        }
        # Batch 2 response (1 doc)
        mock_response_2 = MagicMock()
        mock_response_2.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.99, "document": {"text": "doc 16"}}
            ]
        }
        
        mock_post.side_effect = [mock_response_1, mock_response_2]

        docs = [f"doc {i}" for i in range(17)]
        results = self.reranker.rerank("query", docs, batch_size=16)

        # Expected global indexes:
        # doc 16: index 16, score 0.99
        # doc 15: index 15, score 0.75
        # ...
        # Let's verify that the top result is doc 16 with absolute index 16
        self.assertEqual(results[0]["index"], 16)
        self.assertEqual(results[0]["relevance_score"], 0.99)
        self.assertEqual(results[0]["document"]["text"], "doc 16")

        # Let's verify other indexes are mapped correctly
        # The next highest score should be index 15 with score 0.75 (from batch 1)
        self.assertEqual(results[1]["index"], 15)
        self.assertEqual(results[1]["relevance_score"], 0.75)

    @patch('requests.post')
    def test_rerank_chunks(self, mock_post):
        """Test rerank_chunks maps metadata and preserves original fields."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95, "document": {"text": "text B"}},
                {"index": 0, "relevance_score": 0.45, "document": {"text": "text A"}}
            ]
        }
        mock_post.return_value = mock_response

        chunks = [
            {"chunk_id": 100, "filename": "file_a.txt", "chunk": "text A"},
            {"chunk_id": 101, "filename": "file_b.txt", "chunk": "text B"}
        ]

        reranked = self.reranker.rerank_chunks("query", chunks, batch_size=16)

        self.assertEqual(len(reranked), 2)
        # Top rank should be chunk 101
        self.assertEqual(reranked[0]["chunk_id"], 101)
        self.assertEqual(reranked[0]["relevance_score"], 0.95)
        self.assertEqual(reranked[0]["rank"], 1)
        self.assertEqual(reranked[0]["filename"], "file_b.txt")
        self.assertEqual(reranked[0]["server_document"], "text B")

        # Second rank should be chunk 100
        self.assertEqual(reranked[1]["chunk_id"], 100)
        self.assertEqual(reranked[1]["relevance_score"], 0.45)
        self.assertEqual(reranked[1]["rank"], 2)
        self.assertEqual(reranked[1]["server_document"], "text A")


class TestRemoteRerankerIntegration(unittest.TestCase):
    """Live integration tests for RemoteReranker.
    
    Only runs if the JINA_API_KEY environment variable is set.
    """

    @unittest.skipIf(not os.getenv("JINA_API_KEY"), "JINA_API_KEY environment variable not set")
    def test_live_rerank(self):
        """Test remote reranking with actual Jina AI API."""
        print("\nRunning live integration test against Jina AI Cloud API...")
        reranker = RemoteReranker()
        
        # Test connection status
        self.assertTrue(reranker.check_connection())
        
        query = "What is the capital of France?"
        documents = [
            "Berlin is the capital of Germany.",
            "Paris is the capital and largest city of France.",
            "London is the capital of the United Kingdom."
        ]
        
        results = reranker.rerank(query, documents, batch_size=16)
        
        self.assertEqual(len(results), 3)
        # Paris should be the top result (index 1)
        self.assertEqual(results[0]["index"], 1)
        self.assertGreater(results[0]["relevance_score"], results[1]["relevance_score"])
        self.assertIn("Paris", results[0]["document"]["text"])
        print("Live integration test succeeded!")


if __name__ == "__main__":
    unittest.main()
