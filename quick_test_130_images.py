"""quick_test_130_images.py - Test DirectoryImageProcessor with 130 images.

Usage:
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python quick_test_130_images.py
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

    images_dir = os.path.join(os.path.dirname(__file__), "image_test")
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) < 130:
        print("ERROR: image_test directory not fully populated. Populate first.")
        sys.exit(1)

    print()
    print("======================================================================")
    print("quick_test_130_images.py -- 130 Image Batched Reranking Test")
    print("======================================================================")
    print(f"Scanning directory: {images_dir}")

    # 1. Initialize
    processor = DirectoryImageProcessor(images_dir)
    reranker = RemoteReranker(
        api_key=api_key,
        model="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        base_url="https://openrouter.ai/api/v1/rerank"
    )

    image_files = processor.list_images()
    print(f"Total image files discovered: {len(image_files)}")
    print()

    # 2. Run batched pre-selection
    # Batch size is default (64 images count limit or 30MB payload bytes limit)
    query = "a photograph of a cat"
    top_n = 3                      # Pick top 3 images per batch
    max_limit = 10                 # Collect up to 10 total

    print(f"Querying: '{query}'")
    print(f"Collecting top {top_n} images per batch (max_limit = {max_limit})...")
    print("-" * 70)

    top_images, reached_limit = processor.process_images_with_batched_top_n(
        query=query,
        reranker=reranker,
        top_n=top_n,
        max_limit=max_limit
    )

    print("-" * 70)
    print(f"Collected {len(top_images)} top images across all batches:")
    print(f"Reached limit? {reached_limit}")
    print()

    # Sort globally by score
    sorted_top = sorted(top_images, key=lambda x: x['relevance_score'], reverse=True)

    for i, item in enumerate(sorted_top, 1):
        print(f"Rank {i:2d} (Score: {item['relevance_score']:.4f}) from Batch {item['batch_origin']}")
        print(f"        File: {item['filename']}")
        print(f"        Path: {item['image_path']}")
        print()


if __name__ == "__main__":
    main()
