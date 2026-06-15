"""quick_test15.py - DirectoryImageProcessor Batched Pre-selection Test

Demonstrates process_images_with_batched_top_n() on the local images/ directory.
Uses a low max_payload_bytes limit to demonstrate how it splits files into
multiple API requests to avoid the 30MB payload limit.

Usage:
  python fetch_test_images.py   # Make sure images are downloaded
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python quick_test15.py
"""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EasyRerank import DirectoryImageProcessor, RemoteReranker


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    images_dir = os.path.join(os.path.dirname(__file__), "images")
    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        print("ERROR: Images cache empty. Run: python fetch_test_images.py")
        sys.exit(1)

    print()
    print("======================================================================")
    print("quick_test15.py -- DirectoryImageProcessor Batched Test")
    print("======================================================================")
    print(f"Scanning directory: {images_dir}")

    # 1. Initialize Processor and Reranker
    processor = DirectoryImageProcessor(images_dir)
    reranker = RemoteReranker(
        api_key=api_key,
        model="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        base_url="https://openrouter.ai/api/v1/rerank"
    )

    image_files = processor.list_images()
    print(f"Found {len(image_files)} image files: {image_files}")
    print()

    # 2. Run batched pre-selection with small payload size limit (e.g. 150KB)
    # This will force the 4 images (totaling ~600KB base64 size) to be processed
    # across multiple distinct API request batches!
    query = "a photograph of a cat"
    max_payload_bytes = 150 * 1024  # 150KB limit to force batching
    top_n = 1                      # Pick 1 top image per batch

    print(f"Querying: '{query}'")
    print(f"Batch constraints: max 64 files OR {max_payload_bytes / 1024:.0f}KB payload size")
    print(f"Collecting top {top_n} images per batch...")
    print("-" * 70)

    top_images, reached_limit = processor.process_images_with_batched_top_n(
        query=query,
        reranker=reranker,
        top_n=top_n,
        max_limit=4,
        max_payload_bytes=max_payload_bytes
    )

    print("-" * 70)
    print(f"Collected {len(top_images)} top images across all batches:")
    print(f"Reached limit? {reached_limit}")
    print()

    for i, item in enumerate(top_images, 1):
        print(f"Rank {i:2d} (Score: {item['relevance_score']:.4f}) from Batch {item['batch_origin']}")
        print(f"        File: {item['filename']}")
        print(f"        Path: {item['image_path']}")
        print()


if __name__ == "__main__":
    main()
