"""Test 2: Run a local native Reranker using the Hugging Face transformers pipeline.

This program expands upon test1.py's methodology by loading 'jinaai/jina-reranker-v3'
locally in-memory and executing our standard capitals query test against a list
of documents.

Requirements:
    pip install transformers torch sentencepiece

Usage:
    python3 test2.py
"""

import os
import sys
import time

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("ERROR: PyTorch and/or Transformers libraries are missing.")
    print("Please install them using: pip install torch transformers sentencepiece")
    sys.exit(1)


# 1. Define query and documents (same as the quick series capital test)
QUERY = "What is the capital of France?"
in_memory_docs = [
    "London is the capital and largest city of the United Kingdom.",
    "Berlin is the capital and largest city of Germany.",
    "Paris is the capital and largest city of France, located on the river Seine.",
    "Tokyo is the capital city of Japan, known for its bustling streets."
]

print("=" * 80)
print("LOCAL NATIVE RERANKER (HUGGING FACE TRANSFORMERS METHODOLOGY)")
print("=" * 80)
print(f"Query: \"{QUERY}\"")
print("=" * 80)
print()

# 2. Load model and tokenizer
print("Loading model and tokenizer from Hugging Face ('jinaai/jina-reranker-v3')...")
print("Note: This will download weights (~1.2GB) and run inference natively in-memory.")
start_time = time.time()

try:
    # trust_remote_code=True is required to fetch Jina's custom modeling class
    tokenizer = AutoTokenizer.from_pretrained(
        'jinaai/jina-reranker-v3', 
        trust_remote_code=True
    )
    
    # Auto-detect hardware (Apple Silicon MPS / CUDA / CPU)
    if torch.backends.mps.is_available():
        device = "mps"
        print("-> Apple Silicon GPU (MPS) detected. Loading on MPS...")
    elif torch.cuda.is_available():
        device = "cuda"
        print("-> Nvidia CUDA GPU detected. Loading on CUDA...")
    else:
        device = "cpu"
        print("-> No compatible GPU detected. Loading on CPU...")

    # Load model
    model = AutoModel.from_pretrained(
        'jinaai/jina-reranker-v3',
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device)
    model.eval()
    
    load_time = time.time() - start_time
    print(f"Model successfully loaded on {device.upper()} in {load_time:.2f} seconds.\n")

    # 3. Perform Reranking
    print("Executing native cross-encoder rerank in-memory...")
    rerank_start = time.time()
    
    with torch.no_grad():
        # Jina's custom modeling class provides a built-in .rerank() method
        results = model.rerank(
            QUERY,
            in_memory_docs
        )
        
    inference_time = time.time() - rerank_start

    # 4. Display Results
    # Jina's .rerank returns a list of dictionaries sorted by relevance score
    print("=" * 80)
    print("NATIVE TRANSFORMERS RERANKED RESULTS")
    print("=" * 80)
    print(" Rank |  Score | Original Index & Text Snippet")
    print("------+--------+--------------------------------------------------------------")
    
    for idx, item in enumerate(results, 1):
        score = item['relevance_score']
        original_idx = item['index']
        text = item['document']
        
        print(f"  {idx:2d}  | {score:.4f} | Index {original_idx}")
        print(f"      |        | \"{text}\"")
        if idx < len(results):
            print("------+--------+--------------------------------------------------------------")
            
    print("=" * 80)
    print(f"Inference latency: {inference_time:.4f} seconds | Loading latency: {load_time:.2f} seconds")
    print("=" * 80)

except Exception as e:
    print(f"\nERROR: Failed to load or execute model: {e}")
    sys.exit(1)
