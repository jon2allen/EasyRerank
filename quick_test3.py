"""Quick test: Process ALL Madison documents with top_n=2, then rerank."""

import os
import sys
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EasyRerank import DirectoryTextProcessor, LocalReranker

# Use the Madison directory
madison_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Madison')

# Query to search for
QUERY = "Character of people"

print("=" * 80)
print("Quick Test 3: All Madison documents with top_n=2")
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
# Server error: "input (599 tokens) is too large to process. current batch size: 512"
# So each document + query must fit within 512 tokens.
# Rough estimate: 4 chars per token, so 512 tokens ≈ 2048 chars
# Account for query length (~10 tokens for "Justice") = ~40 chars
# Be conservative: limit sentences to 1500 chars (~375 tokens) to leave room
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

# Show collected chunks with ranking info
print("=" * 80)
print("STEP 1: Collected Top Chunks (by length, from each batch)")
print("=" * 80)
for i, chunk in enumerate(top_chunks, 1):
    batch_origin = chunk.get('batch_origin', 'N/A')
    print(f"{i}. [ID: {chunk['chunk_id']}] from Batch {batch_origin} | {chunk['filename']}")
    print(f"   Length: {len(chunk['chunk'])} chars | "
          f"Content: {chunk['chunk'][:120]}...")
print()

# Now rerank with LocalReranker
print("=" * 80)
print("Reranking collected chunks with LocalReranker")
print("=" * 80)
print()

# Note: Server requires model parameter
reranker = LocalReranker(model='sinjab/bge-reranker-large-F16-GGUF:F16')

if not reranker.check_server():
    print("ERROR: Local rerank server is not running.")
    print("\nPlease start the server with the correct model path:")
    print("  # Option A: Use full cache path")
    print("  llama-server -m /Users/jon2allen/.cache/huggingface/hub/models--jinaai--jina-reranker-v3-GGUF/snapshots/4bbace80cf59987f6fec850519012341c06810d5/jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080")
    print("\n  # Option B: Copy model to current directory first")
    print("  cp /Users/jon2allen/.cache/huggingface/hub/models--jinaai--jina-reranker-v3-GGUF/snapshots/4bbace80cf59987f6fec850519012341c06810d5/jina-reranker-v3-Q4_K_M.gguf .")
    print("  llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080")
    print("\n  # Option C: Use llama-cpp-python")
    print("  python3 -m llama_cpp.server --model jina-reranker-v3-Q4_K_M.gguf --port 8080 --rerank")
    print("\nNote: Your server requires the model parameter.")
    print("      Use: LocalReranker(model='jinaai/jina-reranker-v3-GGUF:Q4_K_M')")
    sys.exit(1)

print(f"\n{'=' * 80}")
print(f"STEP 2: Sending {len(top_chunks)} chunks to LocalReranker")
print(f"Query: '{QUERY}'")
print(f"{'=' * 80}")
print()

try:
    print("Sending request to rerank server...")
    
    reranked = reranker.rerank_chunks(
        query=QUERY,
        chunks=top_chunks,
        batch_size=64
    )
    
    # Check if all scores are zero
    all_zero = all(c.get('relevance_score', 0) == 0 for c in reranked)
    if all_zero and len(reranked) > 0:
        print(f"\nWARNING: All {len(reranked)} results have relevance_score = 0.0")
        print("This may indicate a server or model loading issue.")
        print("\nPossible causes:")
        print("  1. Model not loaded in --rerank mode")
        print("  2. Server started in router mode (--models-max 1) - DON'T use this")
        print("  3. Wrong model path")
        print("\nCorrect server startup:")
        print("  llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080")
        print("  (NO --models-max flag, DO use --rerank)")
    
    print("\n" + "=" * 80)
    print("STEP 3: Final Reranked Results (sorted by relevance score)")
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
    
except requests.exceptions.RequestException as e:
    print(f"ERROR: Failed to connect to rerank server: {e}")
    sys.exit(1)
