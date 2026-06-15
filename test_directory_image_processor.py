"""Unit tests for DirectoryImageProcessor."""

import os
import io
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from EasyRerank import DirectoryImageProcessor, RemoteReranker


class TestDirectoryImageProcessor(unittest.TestCase):
    """Offline unit tests for DirectoryImageProcessor."""

    def setUp(self):
        # We patch os.path.exists and os.path.isdir to allow initializing with dummy dir
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True):
            self.processor = DirectoryImageProcessor("/dummy/images")

    @patch('os.listdir')
    @patch('os.path.isfile')
    def test_list_images(self, mock_isfile, mock_listdir):
        """Test scanning files for image extensions."""
        mock_listdir.return_value = [
            "image1.jpg", "image2.PNG", "document.txt", "photo.webp", "vector.svg", "subfolder"
        ]
        # Treat all listed names as files except "subfolder"
        mock_isfile.side_effect = lambda path: not path.endswith("subfolder")

        images = self.processor.list_images()
        # Should match only .jpg, .png, and .webp extensions (case-insensitive)
        self.assertEqual(images, ["image1.jpg", "image2.PNG", "photo.webp"])

    @patch('os.path.isfile', return_value=True)
    def test_process_images_with_batched_top_n(self, mock_isfile):
        """Test batching by count, payload size, and collecting top_n results."""
        # 3 dummy image files
        filenames = ["cat.jpg", "dog.png", "car.webp"]
        
        # Mock open to return dummy binary image data
        # "cat.jpg" = 10 bytes, "dog.png" = 20 bytes, "car.webp" = 15 bytes
        mock_file_data = {
            "/dummy/images/cat.jpg": b"catdata123",
            "/dummy/images/dog.png": b"dogdata1234567890123",
            "/dummy/images/car.webp": b"cardata1234567"
        }
        
        def mock_open_file(filepath, mode='r', *args, **kwargs):
            content = mock_file_data.get(filepath, b"")
            return io.BytesIO(content)

        # Mock reranker instance
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = [
            # Batch 1 results (cat.jpg)
            [
                {"index": 0, "relevance_score": 0.95}
            ],
            # Batch 2 results (dog.png)
            [
                {"index": 0, "relevance_score": 0.15}
            ],
            # Batch 3 results (car.webp)
            [
                {"index": 0, "relevance_score": 0.85}
            ]
        ]

        # Use a small max_payload_bytes limit to force 2 batches
        # "cat" data URI is data:image/jpeg;base64,Y2F0ZGF0YTEyMw== which is 40 bytes.
        # "dog" data URI is 56 bytes.
        # Let's set max_payload_bytes = 50.
        # This will force cat.jpg in Batch 1, and dog.png in Batch 2 (since cat + dog = 96 > 50).
        # Then car.webp goes into Batch 3.
        
        # For simplicity, let's mock open
        with patch('builtins.open', side_effect=mock_open_file):
            # Run with top_n = 1 (top 1 from each batch)
            collected, reached_limit = self.processor.process_images_with_batched_top_n(
                query="find animal",
                reranker=mock_reranker,
                filenames=filenames,
                top_n=1,
                max_limit=5,
                max_payload_bytes=50  # Very small limit to trigger multi-batching
            )

        # Reranker should be called 3 times (since each image exceeded 50 bytes boundary when combined)
        self.assertEqual(mock_reranker.rerank.call_count, 3)
        
        # We collected top 1 from each batch (total 3 batches -> 3 collected items)
        self.assertEqual(len(collected), 3)
        self.assertFalse(reached_limit)
        
        # Verify result metadata
        self.assertEqual(collected[0]['filename'], 'cat.jpg')
        self.assertEqual(collected[0]['batch_origin'], 1)
        
        self.assertEqual(collected[1]['filename'], 'dog.png')
        self.assertEqual(collected[1]['batch_origin'], 2)
        
        self.assertEqual(collected[2]['filename'], 'car.webp')
        self.assertEqual(collected[2]['batch_origin'], 3)


if __name__ == "__main__":
    unittest.main()
