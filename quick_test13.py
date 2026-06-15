"""quick_test13.py - Base64 Image Input Test

Proves the NVIDIA llama-nemotron-rerank-vl-1b-v2:free model (via OpenRouter)
accepts base64-encoded images as data URIs interchangeably with remote URLs.

Images are loaded from the local images/ cache (run fetch_test_images.py first).

Tests:
  A. URL-based baseline       - known working, establishes reference scores
  B. Base64 data URI          - same image as A, must produce identical scores
  C. Mixed URL + base64       - both image types in same request
  D. Pure base64 images only  - no URLs, no text

Usage:
  python fetch_test_images.py   # run once to populate images/
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python quick_test13.py
"""

import os
import sys

import requests

# Shared image cache utilities
sys.path.insert(0, os.path.dirname(__file__))
from fetch_test_images import fetch_all, load_image_b64, IMAGE_CATALOG

MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
BASE_URL = "https://openrouter.ai/api/v1/rerank"

# Reference URLs (used in Test A/C to show URL vs base64 parity)
CAT_URL = IMAGE_CATALOG["cat.jpg"]["url"]

TEXT_DOCS = [
    {"text": "A fluffy cat sitting on a windowsill in the sun."},
    {"text": "A street map of downtown Berlin."},
    {"text": "A dog playing fetch in the park."},
]

QUERY = "a photograph of a cat"


def rerank(query: str, documents: list, top_n: int = None):
    """Direct API call - returns (status_code, response_dict)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    payload = {
        "model": MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n or len(documents),
    }
    resp = requests.post(
        BASE_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    return resp.status_code, resp.json()


def sep(title: str = "") -> None:
    print("=" * 70)
    if title:
        print(f"  {title}")
        print("-" * 70)


def print_test_header(query: str, documents: list) -> None:
    print(f"  Query   : \"{query}\"")
    print(f"  Docs ({len(documents)}):")
    for i, d in enumerate(documents):
        if isinstance(d, dict):
            if "image" in d:
                src = d["image"]
                if src.startswith("data:"):
                    display = f"[base64 data URI, {len(src):,} chars]"
                else:
                    display = src if len(src) <= 65 else src[:62] + "..."
                print(f"    [{i}] image : {display}")
            elif "text" in d:
                txt = d["text"]
                print(f"    [{i}] text  : {txt if len(txt) <= 65 else txt[:62] + '...'}")
        else:
            s = str(d)
            print(f"    [{i}] str   : {s if len(s) <= 65 else s[:62] + '...'}")
    print()


def print_results(status: int, data: dict) -> None:
    print(f"  HTTP {status}")
    if "error" in data and "results" not in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        print(f"  API ERROR: {msg}")
        return
    results = data.get("results", [])
    if not results:
        print("  WARN: No results returned")
        return
    print(f"  {'Rank':<5} {'Score':<9} {'Type':<6} Result")
    print(f"  {'-'*5} {'-'*9} {'-'*6} {'-'*45}")
    for rank, res in enumerate(results, 1):
        score = res.get("relevance_score", 0.0)
        doc = res.get("document", {})
        if isinstance(doc, dict):
            if "image" in doc:
                src = doc["image"]
                kind = "image"
                display = f"[base64, {len(src):,} chars]" if src.startswith("data:") else src
            elif "text" in doc:
                kind, display = "text", doc["text"]
            else:
                kind, display = "?", str(doc)
        else:
            kind, display = "str", str(doc)
        if len(display) > 55:
            display = display[:52] + "..."
        print(f"  {rank:<5} {score:<9.4f} {kind:<6} {display}")
    print(f"  Total: {len(results)} results | Provider: {data.get('provider', '?')}")


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    print()
    print("quick_test13.py -- Base64 Image Input Test")
    print(f"Model  : {MODEL}")
    print()

    # ---------------------------------------------------------------
    # Ensure images are cached locally
    # ---------------------------------------------------------------
    sep("Loading images from local cache (images/)")
    needed = ["cat.jpg", "dog.jpg"]
    fetch_all(needed)
    cat_b64 = load_image_b64("cat.jpg")
    dog_b64 = load_image_b64("dog.jpg")
    print()

    if not cat_b64:
        print("ERROR: cat.jpg not available. Run: python fetch_test_images.py")
        sys.exit(1)

    second_b64 = dog_b64 if dog_b64 else cat_b64
    second_label = "dog" if dog_b64 else "cat (duplicate)"
    print(f"  cat.jpg : {len(cat_b64):,} chars base64")
    if dog_b64:
        print(f"  dog.jpg : {len(dog_b64):,} chars base64")
    print()

    # ---------------------------------------------------------------
    # Test A: URL-based baseline
    # ---------------------------------------------------------------
    sep("Test A: URL-based baseline (establishes reference scores)")
    docs_a = [{"image": CAT_URL}] + TEXT_DOCS
    print_test_header(QUERY, docs_a)
    status, data = rerank(QUERY, docs_a)
    print_results(status, data)
    print()

    # ---------------------------------------------------------------
    # Test B: Same image as A but via base64 — scores must match
    # ---------------------------------------------------------------
    sep("Test B: Base64 data URI — must produce identical scores to Test A")
    docs_b = [{"image": cat_b64}] + TEXT_DOCS
    print_test_header(QUERY, docs_b)
    status, data = rerank(QUERY, docs_b)
    print_results(status, data)
    print()

    # ---------------------------------------------------------------
    # Test C: Mixed — URL image + base64 image + text in same request
    # ---------------------------------------------------------------
    sep(f"Test C: Mixed URL + base64 ({second_label}) in same request")
    docs_c = [
        {"image": CAT_URL},
        {"image": second_b64},
        {"text": "A fluffy cat sitting on a windowsill in the sun."},
        {"text": "A street map of downtown Berlin."},
    ]
    print_test_header(QUERY, docs_c)
    status, data = rerank(QUERY, docs_c)
    print_results(status, data)
    print()

    # ---------------------------------------------------------------
    # Test D: Pure base64 images only — no URLs, no text
    # ---------------------------------------------------------------
    sep(f"Test D: Pure base64 images only (cat + {second_label}, no text, no URLs)")
    docs_d = [{"image": cat_b64}, {"image": second_b64}]
    print_test_header(QUERY, docs_d)
    status, data = rerank(QUERY, docs_d, top_n=2)
    print_results(status, data)
    print()

    sep("All tests complete")
    print()


if __name__ == "__main__":
    main()
