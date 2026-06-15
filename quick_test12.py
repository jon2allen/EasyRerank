"""quick_test12.py - Vision/Image Reranking Test

Tests RemoteReranker with a mixed list of {"image": url} and {"text": "..."}
documents against the NVIDIA llama-nemotron-rerank-vl-1b-v2:free model
via OpenRouter.

Usage:
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python quick_test12.py
"""

import json
import os
import requests

from EasyRerank import RemoteReranker


NVIDIA_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
BASE_URL = "https://openrouter.ai/api/v1/rerank"

CAT_URL = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
DOG_URL = "https://upload.wikimedia.org/wikipedia/commons/4/43/Cute_dog.jpg"
# Van Gogh full-res (~47MB) exceeds the 30MB NVIDIA API limit - use 300px thumbnail instead
ART_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"


def raw_api_call(label: str, query: str, documents: list, top_n: int = 3) -> None:
    """Make a direct requests.post() and dump the full raw JSON response."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    print("=" * 70)
    print(f"RAW API DEBUG: {label}")
    payload = {
        "model": NVIDIA_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }
    print("Payload:")
    print(json.dumps(payload, indent=2))
    print("-" * 70)
    try:
        resp = requests.post(
            BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        print(f"HTTP {resp.status_code}")
        print("Response JSON:")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text[:1000])
    except Exception as e:
        print(f"Request failed: {e}")
    print("=" * 70)
    print()


def run_test(label: str, query: str, documents: list, top_n: int = None) -> None:
    """Run via EasyRerank RemoteReranker and print formatted results."""
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
        print("ERROR: OPENROUTER_API_KEY not set. Skipping.")
        print("=" * 70)
        print()
        return

    reranker = RemoteReranker(
        api_key=api_key,
        model=NVIDIA_MODEL,
        base_url=BASE_URL,
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
                    kind, src = "image", doc["image"]
                elif "text" in doc:
                    kind, src = "text", doc["text"]
                else:
                    kind, src = "?", str(doc)
            else:
                kind, src = "str", str(doc)
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
    print("quick_test12.py -- Vision/Image Reranking Test")
    print("EasyRerank RemoteReranker with NVIDIA VL model via OpenRouter")
    print()

    # -------------------------------------------------------------------
    # RAW DEBUG SECTION: Bypass EasyRerank to see exact API response JSON
    # -------------------------------------------------------------------
    raw_api_call(
        "Debug A: pure image dicts",
        query="a photograph of a cat",
        documents=[
            {"image": CAT_URL},
            {"image": DOG_URL},
            {"image": ART_URL},
        ],
    )

    raw_api_call(
        "Debug B: image + explicit text dicts",
        query="a photograph of a cat",
        documents=[
            {"image": CAT_URL},
            {"text": "A fluffy cat sitting on a windowsill in the sun."},
            {"text": "A street map of downtown Berlin."},
        ],
    )

    raw_api_call(
        "Debug C: plain strings only (text-only control)",
        query="a photograph of a cat",
        documents=[
            "A fluffy cat sitting on a windowsill in the sun.",
            "A street map of downtown Berlin.",
            "A dog playing fetch in a park.",
        ],
    )

    # -------------------------------------------------------------------
    # EASYRERANK TESTS
    # -------------------------------------------------------------------

    # Test 0: text-only sanity check
    run_test(
        label="Test 0: Text-only sanity check (should always pass)",
        query="a fluffy cat",
        documents=[
            "A fluffy cat sitting on a windowsill in the sun.",
            "A street map of downtown Berlin.",
            "A dog playing fetch in a park.",
        ],
        top_n=3,
    )

    # Test 1: Pure image URL dicts
    run_test(
        label="Test 1: Pure image URL list",
        query="a photograph of a cat",
        documents=[
            {"image": CAT_URL},
            {"image": DOG_URL},
            {"image": ART_URL},
        ],
        top_n=3,
    )

    # Test 2: Mixed image + explicit text dicts
    run_test(
        label="Test 2: Mixed image URLs and text dicts",
        query="a photograph of a cat",
        documents=[
            {"image": CAT_URL},
            {"text": "A fluffy cat sitting on a windowsill in the sun."},
            {"text": "A street map of downtown Berlin."},
            {"image": DOG_URL},
        ],
        top_n=3,
    )

    # Test 3: Mixed image dict + plain strings (auto-wrap)
    run_test(
        label="Test 3: Mixed dicts + plain strings (auto-wrap)",
        query="a photograph of a cat",
        documents=[
            {"image": CAT_URL},
            "A fluffy cat sitting on a windowsill in the sun.",
            "A street map of downtown Berlin.",
        ],
        top_n=3,
    )

    # Test 4: Semantic text query against mixed docs
    run_test(
        label="Test 4: Semantic text query against mixed docs",
        query="famous paintings and visual art",
        documents=[
            {"image": ART_URL},
            {"text": "The Starry Night is an oil painting by Dutch post-impressionist painter Vincent van Gogh."},
            {"text": "Quantum computing uses qubits to perform parallel calculations."},
            {"image": CAT_URL},
        ],
        top_n=4,
    )

    print("All tests complete.")


if __name__ == "__main__":
    main()
