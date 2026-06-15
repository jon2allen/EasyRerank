"""quick_test14.py - Visual Discrimination Test

Sends 4 images (cat, dog, horse, car) to the NVIDIA reranker and shows
which document ranks #1 for each query. Documents are always in the same
fixed order so you can read the results directly.

Documents (fixed order for every test):
  Doc 1: cat.jpg
  Doc 2: dog.jpg
  Doc 3: horse.jpg
  Doc 4: car.jpg

Usage:
  python fetch_test_images.py   # run once to populate images/
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python quick_test14.py
"""

import os
import sys

import requests

sys.path.insert(0, os.path.dirname(__file__))
from fetch_test_images import fetch_all, load_image_b64

MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
BASE_URL = "https://openrouter.ai/api/v1/rerank"

# Fixed document order — never changes between tests
SUBJECTS = ["cat", "dog", "horse", "car"]
FILENAMES = {s: f"{s}.jpg" for s in SUBJECTS}

QUERIES = [
    "a photograph of a cat",
    "a photograph of a dog",
    "an automobile, car or vehicle",
    "a horse or equine animal",
]


def rerank(query: str, documents: list):
    """Direct API call - returns (status_code, response_dict)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    payload = {
        "model": MODEL,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
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


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    print()
    print("quick_test14.py -- Visual Discrimination Test")
    print(f"Model  : {MODEL}")
    print()

    # ---------------------------------------------------------------
    # Load all images from local cache
    # ---------------------------------------------------------------
    sep("Loading images from local cache (images/)")
    fetch_all(list(FILENAMES.values()))
    print()

    image_b64s = {}
    for subject, fname in FILENAMES.items():
        b64 = load_image_b64(fname)
        if b64:
            image_b64s[subject] = b64
        else:
            print(f"  ERROR: {fname} not available. Run: python fetch_test_images.py")
            sys.exit(1)

    # Print fixed document index for reference
    print("  Fixed document order (same for every test):")
    for i, subject in enumerate(SUBJECTS, 1):
        fname = FILENAMES[subject]
        size = len(image_b64s[subject]) // 1024
        print(f"    Doc {i}: {subject:<6}  ({size}KB base64)  images/{fname}")
    print()

    # Build the fixed docs list once
    docs = [{"image": image_b64s[s]} for s in SUBJECTS]

    # ---------------------------------------------------------------
    # Run each query against the same fixed document list
    # ---------------------------------------------------------------
    for query in QUERIES:
        sep(f'Query: "{query}"')
        print()

        status, data = rerank(query, docs)

        if "error" in data and "results" not in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            print(f"  HTTP {status} | API ERROR: {msg}")
            print()
            continue

        results = data.get("results", [])
        if not results:
            print(f"  HTTP {status} | WARN: No results returned")
            print()
            continue

        print(f"  HTTP {status} | Provider: {data.get('provider', '?')}")
        print()

        # Build a score lookup: original doc index -> relevance_score
        score_by_idx = {res.get("index", 0): res.get("relevance_score", 0.0)
                        for res in results}

        # --- Raw scores: original document order ---
        print(f"  Raw scores (original document order):")
        print(f"  {'Doc#':<6} {'Subject':<8} {'Score':<10} Raw index from API")
        print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*20}")
        for i, subject in enumerate(SUBJECTS):
            score = score_by_idx.get(i, 0.0)
            print(f"  Doc {i+1:<2}  {subject:<8} {score:.6f}   index={i}")
        print()

        # --- Ranked results: sorted by relevance descending ---
        print(f"  Ranked results (sorted by score):")
        print(f"  {'Rank':<6} {'Doc#':<6} {'Subject':<8} {'Score'}")
        print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
        for rank, res in enumerate(results, 1):
            idx = res.get("index", 0)
            score = res.get("relevance_score", 0.0)
            subject = SUBJECTS[idx] if idx < len(SUBJECTS) else f"idx{idx}"
            doc_num = idx + 1
            marker = "  <-- #1 winner" if rank == 1 else ""
            print(f"  {rank:<6} Doc {doc_num:<3} {subject:<8} {score:.6f}{marker}")
        print()

    sep("Done")
    print()


if __name__ == "__main__":
    main()
