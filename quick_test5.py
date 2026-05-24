"""Quick test 5: Showcase the EasyRanker meta-wrapper class.

This script tests the automatic backend routing, list-based reranking,
directory-based reranking, and cached latest outputs of EasyRanker.

Usage:
    python3 quick_test5.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EasyRerank import EasyRanker

# 1. Setup in-memory list and directory path
in_memory_docs = [
    "London is the capital and largest city of the United Kingdom.",
    "Berlin is the capital and largest city of Germany.",
    "Paris is the capital and largest city of France, located on the river Seine.",
    "Tokyo is the capital city of Japan, known for its bustling streets."
]
madison_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Madison')

print("=" * 80)
print("EASYRANKER MULTI-BACKEND AND DUAL-MODE DEMO")
print("=" * 80)
print()

try:
    # 2. Initialize in 'auto' mode
    # This will automatically detect the first matching reranker model on your local server.
    print("Step 1: Initializing EasyRanker in 'auto' routing mode...")
    ranker = EasyRanker(backend='auto')
    
    # 3. Rerank in-memory list
    print("\nStep 2: Reranking in-memory list of capital cities...")
    results_list = ranker.rerank(
        query="What is the capital of France?",
        documents=in_memory_docs,
        verbose=True
    )
    
    # Verify top result is Paris
    top_score = results_list[0]['relevance_score']
    # Remote uses structure: results[0]['document']['text'] or we reconstruct
    # Let's inspect index
    top_index = results_list[0]['index']
    print(f"Top parsed document: \"{in_memory_docs[top_index]}\" (index {top_index}, score {top_score:.4f})")
    
    # 4. Rerank Madison speeches directory
    print("\nStep 3: Reranking speeches in the 'Madison' directory...")
    # This will load, parse text files into chunks, and rerank them remotely
    results_dir = ranker.rerank(
        query="Separation of religious institutions from civil government authority",
        documents=madison_dir,
        top_n=3,
        verbose=True,
        # max_sentence_length=3000  # Uncomment to override & leverage larger context windows (e.g. 131K tokens on Jina Remote)
        max_sentence_length=1500    # Safe default limit for local llama.cpp servers (512 token limit)
    )
    
    # 5. Access latest output via convenience method
    print("\nStep 4: Demonstrating get_latest_output() convenience access...")
    latest = ranker.get_latest_output()
    print(f"Retrieved cached results list length: {len(latest)}")
    for i, res in enumerate(latest, 1):
        print(f"  Rank {i}: {res['filename']} (ID: {res['chunk_id']}) -> score: {res['relevance_score']:.4f}")
        print(f"  Text: \"{res['chunk'].strip()}\"\n")
        
    print("\n" + "=" * 80)
    print("EasyRanker demo successfully completed!")
    print("=" * 80)

except Exception as e:
    print(f"\nERROR: Demo failed: {e}")
    sys.exit(1)
