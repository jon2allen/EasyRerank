"""Quick test 4: Process ALL Madison documents with top_n=2, then rerank remotely.

This script demonstrates processing all text files, extracting candidate chunks, 
and reranking them using the RemoteReranker class communicating with Jina AI's Remote API.

Usage:
    python3 quick_test4.py
"""

import os
import sys
import re
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EasyRerank import DirectoryTextProcessor, RemoteReranker

# Use the Madison directory
madison_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Madison')

# Query to search for
QUERY = "Character of people"

print("=" * 80)
print("Quick Test 4: All Madison documents with top_n=2 (Remote Reranking)")
print("=" * 80)
print()

# Initialize processor
processor = DirectoryTextProcessor(madison_dir)

# Get all .txt files in Madison directory
all_files = processor.list_files()
print(f"Found {len(all_files)} .txt files in Madison directory:")
for f in all_files:
    print(f"  - {f}")
print()

# Process all files with batched top_n=2 and max_sentence_length
MAX_SENTENCE_LENGTH = 1500  # Approximately 375 tokens

print(f"Processing all files with batched top_n=2, max_limit=64, max_sentence_length={MAX_SENTENCE_LENGTH}...")
print("Batch processing details:")
top_chunks, reached_limit = processor.process_with_batched_top_n(
    filenames=None,  # Process all files
    chunk_size=1,
    top_n=2,              # Top 2 from each batch
    max_limit=64,
    batch_size=64,
    max_sentence_length=MAX_SENTENCE_LENGTH
)

print(f"\n=> Collected {len(top_chunks)} top chunks from all batches")
print(f"=> Reached max_limit: {reached_limit}")
print()

# Show collected chunks
print("=" * 80)
print("STEP 1: Collected Top Chunks (by length, from each batch)")
print("=" * 80)
for i, chunk in enumerate(top_chunks, 1):
    batch_origin = chunk.get('batch_origin', 'N/A')
    print(f"{i}. [ID: {chunk['chunk_id']}] from Batch {batch_origin} | {chunk['filename']}")
    print(f"   Length: {len(chunk['chunk'])} chars | "
          f"Content: {chunk['chunk'][:120]}...")
print()

# Initialize RemoteReranker
print("=" * 80)
print("Reranking collected chunks with RemoteReranker")
print("=" * 80)
print()

# Look for API key in environment variable
api_key = os.getenv("JINA_API_KEY")

# Fallback to local 'api_key' file if not set in environment
if not api_key:
    api_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_key')
    if os.path.exists(api_key_path):
        try:
            with open(api_key_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Parse JINA_API_KEY = "key" format or just raw key
                match = re.search(r'["\'](jina_[a-zA-Z0-9_]+)["\']', content)
                if match:
                    api_key = match.group(1)
                else:
                    # Try simple string extraction if quotes aren't standard
                    parts = content.split("=")
                    api_key = parts[-1].replace('"', '').replace("'", "").strip()
        except Exception as e:
            print(f"Warning: Failed to parse api_key file: {e}")

try:
    if not api_key:
        raise ValueError(
            "Jina AI API key not found. Please set JINA_API_KEY environment variable "
            "or create an 'api_key' file in this directory."
        )

    reranker = RemoteReranker(api_key=api_key)

    # Check connection
    if not reranker.check_connection():
        print("ERROR: Remote rerank server could not be reached or authorization failed.")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print(f"STEP 2: Sending {len(top_chunks)} chunks to RemoteReranker")
    print(f"Query: '{QUERY}'")
    print(f"{'=' * 80}")
    print()

    print("Sending request to remote Jina AI API...")
    reranked = reranker.rerank_chunks(
        query=QUERY,
        chunks=top_chunks,
        batch_size=64
    )

    print("\n" + "=" * 80)
    print("STEP 3: Final Remote Reranked Results (sorted by relevance score)")
    print("=" * 80)
    print()

    for i, chunk in enumerate(reranked, 1):
        score = chunk['relevance_score']
        batch_origin = chunk.get('batch_origin', 'N/A')
        score_bar = "█" * int(score * 50) if score > 0 else ""
        print(f"Rank {i:2d} (Score: {score:.4f}) from Batch {batch_origin} {score_bar}")
        print(f"        File: {chunk['filename']}")
        print(f"        Chunk ID: {chunk['chunk_id']}")
        print(f"        Text: {chunk['chunk'][:120]}...")
        print()

    print("=" * 80)
    print(f"Total: {len(reranked)} chunks reranked")
    print("=" * 80)

    # Summary statistics
    if reranked:
        scores = [c['relevance_score'] for c in reranked]
        print(f"\nScore Statistics:")
        print(f"  Min:  {min(scores):.4f}")
        print(f"  Max:  {max(scores):.4f}")
        print(f"  Mean: {sum(scores)/len(scores):.4f}")

except Exception as e:
    print(f"ERROR: Failed to run remote reranking: {e}")
    sys.exit(1)
