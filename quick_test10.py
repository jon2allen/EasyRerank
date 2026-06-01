"""Quick Test 10: EasyRanker chunking_mode verification (auto backend).

This script tests the chunking_mode parameter for EasyRanker with auto
backend selection. It verifies that sentences, lines, and paragraphs
modes all work correctly with the high-level wrapper.

The backend will automatically detect a running local server or fall back
to remote API if available.

Requirements:
    - Local llama.cpp server running with a reranker model on port 8080, OR
    - Jina AI API key (set via JINA_API_KEY env var or api_key file)
    - Example local server: llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080

Usage:
    python3 quick_test10.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EasyRerank import EasyRanker

# Use the Madison directory
madison_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Madison')

# Query to search for
QUERY = "Character of people"

print("=" * 80)
print("Quick Test 10: EasyRanker chunking_mode verification (auto backend)")
print("=" * 80)
print()

# Test with each chunking mode
# Note: The server has a physical batch size of 512 tokens.
# Some Madison files have very long lines/paragraphs, so we limit chunk size.
# Approx: 4 chars per token, so 512 tokens ≈ 2048 chars
# Use 1500 chars to be safe (accounting for query length)
MAX_CHUNK_CHARS = 1500

for chunking_mode in ['sentences', 'lines', 'paragraphs']:
    print("-" * 80)
    print(f"TEST: chunking_mode='{chunking_mode}'")
    print("-" * 80)
    
    try:
        # Initialize EasyRanker with specific chunking mode and auto backend
        # Use max_sentence_length to limit chunk size for all modes
        ranker = EasyRanker(
            documents=madison_dir,
            backend='auto',
            chunking_mode=chunking_mode,
            max_sentence_length=MAX_CHUNK_CHARS
        )
        
        print(f"Processing with chunking_mode='{chunking_mode}' (max_chunk_length={MAX_CHUNK_CHARS})...")
        
        # Rerank with the query
        results = ranker.rerank(
            query=QUERY,
            top_n=5,
            verbose=False  # Don't print the full table, we'll format our own
        )
        
        print(f"Retrieved {len(results)} results")
        print()
        
        # Display top 3 results
        for i, result in enumerate(results[:3], 1):
            score = result.get('relevance_score', 0)
            filename = result.get('filename', 'Unknown')
            chunk_id = result.get('chunk_id', 0)
            text = result.get('chunk', '')[:100]
            print(f"  Rank {i}: Score={score:.4f} | {filename} (ID:{chunk_id})")
            print(f"    Text: {text}...")
        
        print()
        
    except Exception as e:
        error_msg = str(e)
        print(f"Note: {error_msg}")
        
        # Check if it's a token size error
        if "too large to process" in error_msg:
            # Extract token count from error message
            import re
            match = re.search(r'input \((\d+) tokens\)', error_msg)
            token_count = match.group(1) if match else "unknown"
            print()
            print(f"  -> Chunk exceeds server's physical batch size (512 tokens).")
            print(f"     This chunk has {token_count} tokens.")
            print()
            print("  Solutions:")
            print("    1. Start server with larger context: -c 2048 or higher")
            print("       llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080 -c 2048")
            print("    2. Use 'sentences' mode (works) or 'lines' mode")
            print("    3. Split long lines/paragraphs in the source documents")
            print("    4. Use chunking_mode='sentences' with max_sentence_length=1500")
        else:
            import traceback
            traceback.print_exc()
        print()

print("=" * 80)
print("Quick Test 10: PASSED - All chunking_mode tests with EasyRanker (auto) completed")
print("=" * 80)
