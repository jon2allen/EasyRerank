"""quick_test12.py - Vision/Image Reranking Test

Tests RemoteReranker with a mixed list of {"image": url} and {"text": "..."}
documents against the NVIDIA llama-nemotron-rerank-vl-1b-v2:free model
via OpenRouter.

Validates:
  - Mixed image URL + text dict payloads are formatted correctly
  - Plain strings alongside dicts are auto-wrapped as {"text": s}
  - Scores are returned for both image and text entries
  - Results are sorted by relevance score descending

Requirements:
  - OPENROUTER_API_KEY environment variable must be set

Usage:
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python quick_test12.py
"""

import os
import sys

from EasyRerank import RemoteReranker


def run_test(label: str, query: str, documents: list, top_n: int = None) -> None:
    """Run a single rerank test and print results."""
    print("=" * 70)
    print(f"TEST: {label}")
    print(f"Query: \"{query}\"")
    print(f"Documents ({len(documents)}):")
    for i, d in enumerate(documents):
        if isinstance(d, dict):
            src = d.get("image") or d.get("text", "")
            kind = "image" if "image" in d else "text"
            display = src if len(src) <= 60 else src[:57] + "..."
            print(f"  [{i}] ({kind}) {display}")
        else:
            display = d if len(d) <= 60 else d[:57] + "..."
            print(f"  [{i}] (str)  {display}")
    print("-" * 70)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set. Skipping test.")
        print("=" * 70)
        print()
        return

    reranker = RemoteReranker(
        api_key=api_key,
        model="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        base_url="https://openrouter.ai/api/v1/rerank"
    )

    try:
        results = reranker.rerank(query=query, documents=documents, top_n=top_n)

        print(f" Rank | Score  | Type  | Source")
        print(f"------+--------+-------+--------------------------------------------------")
        for rank, res in enumerate(results, 1):
            score = res.get("relevance_score", 0.0)
            doc = res.get("document", {})
            if isinstance(doc, dict):
                if "image" in doc:
                    kind = "image"
                    src = doc["image"]
                elif "text" in doc:
                    kind = "text"
                    src = doc["text"]
                else:
                    kind = "?"
                    src = str(doc)
            else:
                kind = "str"
                src = str(doc)
            display = src if len(src) <= 50 else src[:47] + "..."
            print(f"  {rank:2d}  | {score:.4f} | {kind:<5} | {display}")

        print(f"Total results: {len(results)}")
        print("PASS" if results else "WARN: no results returned")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 70)
    print()


def main() -> None:
    print()
    print("quick_test12.py — Vision/Image Reranking Test")
    print("EasyRerank RemoteReranker with NVIDIA VL model via OpenRouter")
    print()

    # --- Test 1: Pure image URLs ---
    run_test(
        label="Test 1: Pure image URL list",
        query="a photograph of a cat",
        documents=[
            {"image": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"},
            {"image": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"},
            {"image": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"},
        ],
        top_n=3,
    )

    # --- Test 2: Mixed image + text dicts ---
    run_test(
        label="Test 2: Mixed image URLs and text dicts",
        query="a photograph of a cat",
        documents=[
            {"image": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"},
            {"text": "A fluffy cat sitting on a windowsill in the sun."},
            {"text": "A street map of downtown Berlin."},
            {"image": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"},
        ],
        top_n=3,
    )

    # --- Test 3: Mixed dicts + plain strings (auto-wrap test) ---
    run_test(
        label="Test 3: Mixed dicts + plain strings (auto-wrap)",
        query="a photograph of a cat",
        documents=[
            {"image": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"},
            "A fluffy cat sitting on a windowsill in the sun.",   # plain string — auto-wrapped
            "A street map of downtown Berlin.",                    # plain string — auto-wrapped
        ],
        top_n=3,
    )

    # --- Test 4: Non-image query against text + image mix ---
    run_test(
        label="Test 4: Semantic text query against mixed docs",
        query="famous paintings and visual art",
        documents=[
            {"image": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"},
            {"text": "The Starry Night is an oil painting by Dutch post-impressionist painter Vincent van Gogh."},
            {"text": "Quantum computing uses qubits to perform parallel calculations."},
            {"image": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"},
        ],
        top_n=4,
    )

    print("All tests complete.")


if __name__ == "__main__":
    main()
