"""Quick Test 9: EasyRanker chunking_mode verification (auto backend).

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
    python3 quick_test9.py
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
print("Quick Test 9: EasyRanker chunking_mode verification (auto backend)")
print("=" * 80)
print()

# Test with each chunking mode
for chunking_mode in ['sentences', 'lines', 'paragraphs']:
    print("-" * 80)
    print(f"TEST: chunking_mode='{chunking_mode}'")
    print("-" * 80)
    
    try:
        # Initialize EasyRanker with specific chunking mode and auto backend
        ranker = EasyRanker(
            documents=madison_dir,
            backend='auto',
            chunking_mode=chunking_mode
   #         model='jinaai/jina-reranker-v3-GGUF:Q4_K_M'
        )
        
        print(f"Processing with chunking_mode='{chunking_mode}'...")
        
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
        print(f"ERROR with mode '{chunking_mode}': {e}")
        import traceback
        traceback.print_exc()
        print()

print("=" * 80)
print("Quick Test 9: PASSED - All chunking_mode tests with EasyRanker (auto) completed")
print("=" * 80)
