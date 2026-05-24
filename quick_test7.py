"""Quick test 7: Showcase the EasyRanker meta-wrapper class explicitly targeting the zz BGE reranker.

This script demonstrates forcing EasyRanker to run locally using the newly-ordered 
'zz2Felladrin/gguf-Q8_0-bge-reranker-v2-m3:Q8_0' model on your local server.

Usage:
    python3 quick_test7.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from easy_ranker import EasyRanker

# 1. Setup in-memory list and directory path
in_memory_docs = [
    "London is the capital and largest city of the United Kingdom.",
    "Berlin is the capital and largest city of Germany.",
    "Paris is the capital and largest city of France, located on the river Seine.",
    "Tokyo is the capital city of Japan, known for its bustling streets."
]
madison_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Madison')

print("=" * 80)
print("EASYRANKER DEMO (EXPLICIT LOCAL ZZ RERANK MODE)")
print("=" * 80)
print()

try:
    # 2. Initialize in 'local' mode forcing one of the 'zz' models
    model_name = "zz2Felladrin/gguf-Q8_0-bge-reranker-v2-m3:Q8_0"
    print(f"Step 1: Initializing EasyRanker in local mode forcing model '{model_name}'...")
    ranker = EasyRanker(backend='local', model=model_name)
    
    # 3. Rerank in-memory list
    print("\nStep 2: Reranking in-memory list of capital cities...")
    results_list = ranker.rerank(
        query="What is the capital of France?",
        documents=in_memory_docs,
        verbose=True
    )
    
    # Verify top result is Paris
    top_score = results_list[0]['relevance_score']
    top_index = results_list[0]['index']
    print(f"Top parsed document: \"{in_memory_docs[top_index]}\" (index {top_index}, score {top_score:.4f})")
    
    # 4. Rerank Madison speeches directory
    print("\nStep 3: Reranking speeches in the 'Madison' directory...")
    results_dir = ranker.rerank(
        query="Separation of religious institutions from civil government authority",
        documents=madison_dir,
        top_n=3,
        verbose=True,
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
    print("EasyRanker local zz-model demo successfully completed!")
    print("=" * 80)

except Exception as e:
    print(f"\nERROR: Demo failed: {e}")
    sys.exit(1)
