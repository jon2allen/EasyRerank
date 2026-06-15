"""quick_test14.py - Visual Discrimination Test

Proves the NVIDIA llama-nemotron-rerank-vl-1b-v2:free model can visually
distinguish between fundamentally different subjects (cat, dog, horse, car)
by correctly ranking the matching image #1 for each query.

All 4 images are loaded from the local images/ cache (fast, no re-downloading).

Tests:
  1. Query "a photograph of a cat"    -> cat must rank #1
  2. Query "a photograph of a dog"    -> dog must rank #1
  3. Query "an automobile or vehicle" -> car must rank #1
  4. Query "a horse or equine animal" -> horse must rank #1

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

DISCRIMINATION_TESTS = [
    {
        "label": "Test 1: Find the CAT",
        "query": "a photograph of a cat",
        "expected": "cat",
    },
    {
        "label": "Test 2: Find the DOG",
        "query": "a photograph of a dog",
        "expected": "dog",
    },
    {
        "label": "Test 3: Find the CAR",
        "query": "an automobile, car or vehicle",
        "expected": "car",
    },
    {
        "label": "Test 4: Find the HORSE",
        "query": "a horse or equine animal",
        "expected": "horse",
    },
]


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


def run_discrimination_test(
    label: str,
    query: str,
    expected: str,
    image_b64s: dict,
) -> bool:
    """
    Rerank all loaded images against the query.
    Returns True if the expected subject image ranks #1.
    """
    sep(label)

    subjects = list(image_b64s.keys())
    docs = [{"image": image_b64s[s]} for s in subjects]

    print(f"  Query    : \"{query}\"")
    print(f"  Expected : '{expected}' image ranks #1")
    print(f"  Subjects : {', '.join(subjects)}")
    print()

    status, data = rerank(query, docs)

    if "error" in data and "results" not in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        print(f"  HTTP {status} | API ERROR: {msg}")
        return False

    results = data.get("results", [])
    if not results:
        print(f"  HTTP {status} | WARN: No results returned")
        return False

    print(f"  HTTP {status} | Provider: {data.get('provider', '?')}")
    print()
    print(f"  {'Rank':<5} {'Score':<9} {'Subject':<8} ")
    print(f"  {'-'*5} {'-'*9} {'-'*8}")

    winner = None
    for rank, res in enumerate(results, 1):
        idx = res.get("index", 0)
        score = res.get("relevance_score", 0.0)
        subject = subjects[idx] if idx < len(subjects) else f"idx{idx}"
        marker = "  <-- TOP" if rank == 1 else ""
        if rank == 1:
            winner = subject
        print(f"  {rank:<5} {score:<9.4f} {subject:<8}{marker}")

    passed = winner == expected
    verdict = "PASS" if passed else f"FAIL (got '{winner}', expected '{expected}')"
    print()
    print(f"  Result: {verdict}")
    return passed


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    print()
    print("quick_test14.py -- Visual Discrimination Test")
    print(f"Model  : {MODEL}")
    print(f"Images : cat | dog | horse | car")
    print()

    # ---------------------------------------------------------------
    # Load all 4 images from local cache (download if missing)
    # ---------------------------------------------------------------
    sep("Loading images from local cache (images/)")
    needed = ["cat.jpg", "dog.jpg", "horse.jpg", "car.jpg"]
    fetch_all(needed)
    print()

    image_b64s = {}
    for fname in needed:
        label = fname.replace(".jpg", "")
        b64 = load_image_b64(fname)
        if b64:
            image_b64s[label] = b64
            print(f"  {label:<6} : {len(b64):>9,} chars  (images/{fname})")
        else:
            print(f"  {label:<6} : MISSING - run: python fetch_test_images.py")

    if len(image_b64s) < 2:
        print()
        print("ERROR: Need at least 2 images. Run: python fetch_test_images.py")
        sys.exit(1)

    missing = [t["expected"] for t in DISCRIMINATION_TESTS if t["expected"] not in image_b64s]
    if missing:
        print(f"\n  WARNING: Missing images for tests: {missing}")
        print(f"  These tests will be skipped.\n")

    print()

    # ---------------------------------------------------------------
    # Run discrimination tests
    # ---------------------------------------------------------------
    test_results = []
    for test in DISCRIMINATION_TESTS:
        if test["expected"] not in image_b64s:
            sep(test["label"])
            print(f"  SKIPPED: '{test['expected']}' image not in cache")
            test_results.append(None)
        else:
            passed = run_discrimination_test(
                label=test["label"],
                query=test["query"],
                expected=test["expected"],
                image_b64s=image_b64s,
            )
            test_results.append(passed)
        print()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    sep("Summary")
    run = [r for r in test_results if r is not None]
    passed = sum(run)
    total = len(run)
    skipped = len(test_results) - total
    print(f"  Passed  : {passed}/{total} tests")
    if skipped:
        print(f"  Skipped : {skipped} (missing images)")
    print(f"  Model correctly identified the target image in {passed} of {total} queries.")
    print()


if __name__ == "__main__":
    main()
