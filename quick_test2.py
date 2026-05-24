"""Quick test program: Rerank Madison's inaugural addresses on query 'Justice'."""

import os
import sys
import requests

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from directory_text_processor import DirectoryTextProcessor
from local_reranker import LocalReranker

# Use the actual Madison directory
madison_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Madison')

# Query to search for - change this to test different queries
QUERY = "Justice"

# Get chunks from both inaugural addresses
print(f"Loading Madison's inaugural addresses...\n")
processor = DirectoryTextProcessor(madison_dir)

files = [
    '01_1809-03-04_First_Inaugural_Address.txt',
    '12_1813-03-04_Second_Inaugural_Address.txt'
]

iterator = processor.process_with_index(filenames=files, chunk_size=1)
chunks = list(iterator)[:50]  # Only first 50 chunks

# Debug: Count chunks per file
from collections import Counter
file_counts = Counter(chunk['filename'] for chunk in chunks)
print(f"Loaded {len(chunks)} chunks (first 50) from {len(files)} files")
print(f"Chunks per file: {dict(file_counts)}\n")

# Rerank with query
print(f"Reranking chunks with query: '{QUERY}'\n")
reranker = LocalReranker()

try:
    # Check if server is running
    if not reranker.check_server():
        print("ERROR: Local rerank server is not running.")
        print("Please start the server first:")
        print("  llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080")
        print("or")
        print("  python3 -m llama_cpp.server --model jina-reranker-v3-Q4_K_M.gguf --port 8080")
        sys.exit(1)

    # Debug: Show sample chunks from each file being sent (one batch of 64)
    print("\nDEBUG: Sample chunks being sent to reranker (ONE batch of up to 64):")
    file_samples = {}
    for chunk in chunks:
        fname = chunk['filename']
        if fname not in file_samples:
            file_samples[fname] = chunk['chunk'][:80]
    for fname, sample in file_samples.items():
        print(f"  {fname}: {sample}...")

    # Rerank chunks - one batch
    print(f"DEBUG: Sending {len(chunks)} chunks in ONE batch of 64")
    reranked = reranker.rerank_chunks(
        query=QUERY,
        chunks=chunks,
        batch_size=64
    )

    # Debug: Count results per file in reranked output
    rerank_file_counts = Counter(chunk['filename'] for chunk in reranked)
    print(f"\nDEBUG: Reranked results per file: {dict(rerank_file_counts)}")
    
    # Debug: Check what indices the server returned
    print(f"DEBUG: Server returned {len(reranked)} results")
    
    # Debug: Show first few results with server document vs original
    print("\nDEBUG: First 3 results - comparing server document vs original chunk:")
    for i, chunk in enumerate(reranked[:3]):
        server_doc = chunk.get('server_document', '')[:80]
        original_chunk = chunk.get('chunk', '')[:80]
        match = "✓" if server_doc and original_chunk and server_doc in original_chunk else "✗"
        print(f"  Rank {i+1}: server_doc={server_doc[:60]}... | orig={original_chunk[:60]}... {match}")
    
    # Debug: Check chunk_id distribution in results
    chunk_file_map = {chunk['chunk_id']: chunk['filename'] for chunk in chunks}
    result_files_by_id = {chunk['chunk_id']: chunk['filename'] for chunk in reranked}
    print(f"\nDEBUG: First 10 result chunk_ids and their files: {[(cid, result_files_by_id.get(cid, 'MISSING')) for cid in sorted(result_files_by_id.keys())[:10]]}")
    
    # Debug: Show chunk_id ranges per file
    print(f"DEBUG: Chunk ID ranges - First file: 0-20, Second file: 21-53")

    # Display results sorted by highest score
    print("=" * 80)
    print("RERANKED RESULTS (sorted by relevance score, highest first):\n")
    print("=" * 80)

    for i, chunk in enumerate(reranked, 1):
        print(f"\nRank {i} (Score: {chunk['relevance_score']:.4f})")
        print(f"  File: {chunk['filename']}")
        print(f"  Chunk ID: {chunk['chunk_id']}")
        server_doc = chunk.get('server_document', '')
        print(f"  Server returned: {server_doc[:100]}{'...' if len(server_doc) > 100 else ''}")
        print(f"  Original chunk: {chunk['chunk'][:100]}{'...' if len(chunk['chunk']) > 100 else ''}")

    print("\n" + "=" * 80)
    print(f"Total: {len(reranked)} chunks reranked")
    print("=" * 80)

    # Show top 5
    print(f"\n\nTOP 5 MOST RELEVANT TO '{QUERY}':\n")
    for i, chunk in enumerate(reranked[:5], 1):
        server_doc = chunk.get('server_document', '')
        print(f"{i}. [Score: {chunk['relevance_score']:.4f}] {chunk['filename']} (chunk {chunk['chunk_id']})")
        print(f"   Server: {server_doc[:100]}{'...' if len(server_doc) > 100 else ''}")
        print(f"   Original: {chunk['chunk'][:100]}{'...' if len(chunk['chunk']) > 100 else ''}\n")

except requests.exceptions.RequestException as e:
    print(f"ERROR: Failed to connect to rerank server: {e}")
    print("\nPlease start the local rerank server:")
    print("  llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080")
    sys.exit(1)
