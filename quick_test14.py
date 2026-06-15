"""quick_test14.py - Visual Discrimination Test

Proves the NVIDIA llama-nemotron-rerank-vl-1b-v2:free model can visually
distinguish between fundamentally different subjects (cat, dog, horse, car)
and correctly rank the matching image highest.

Tests:
  1. Query "a photograph of a cat"    -> cat image should rank #1
  2. Query "a photograph of a dog"    -> dog image should rank #1
  3. Query "an automobile or vehicle" -> car image should rank #1

All images are downloaded once as base64 data URIs to avoid CDN
reliability issues with repeated URL fetches.

Usage:
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python quick_test14.py
"""

import base64
import os
import sys
import time

import requests

MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
BASE_URL = "https://openrouter.ai/api/v1/rerank"

# Small, stable Wikimedia images covering clearly distinct subjects
IMAGE_CATALOG = [
    {
        "label": "cat",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        "description": "A domestic cat (tabby)",
    },
    {
        "label": "dog",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg",
        "description": "A yellow Labrador Retriever dog",
    },
    {
        "label": "horse",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/de/Nokota_Horses_cropped.jpg",
        "description": "Nokota horses in a field",
    },
    {
        "label": "car",
        "url": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=400&q=80",
        "description": "A Porsche sports car (automobile)",
    },
]

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def fetch_as_b64(label: str, url: str, retries: int = 2):
    """Download image with retry on 429, return data_uri or None."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=20, allow_redirects=True)
            if resp.status_code == 429:
                wait = 4 * (attempt + 1)
                print(f"    [429 rate-limit] waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"    ERROR: HTTP {resp.status_code}")
                return None
            size_kb = len(resp.content) // 1024
            if size_kb == 0:
                print(f"    ERROR: 0 bytes received")
                return None
            ct = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
            if not ct.startswith("image/"):
                ct = "image/jpeg"
            b64 = base64.b64encode(resp.content).decode()
            data_uri = f"data:{ct};base64,{b64}"
            print(f"    OK  [{label:5s}] {size_kb:>5}KB  {url.split('/')[-1][:50]}")
            return data_uri
        except Exception as e:
            print(f"    ERROR: {e}")
            return None
    print(f"    FAILED after {retries} attempts")
    return None


def rerank(query: str, documents: list, top_n: int = None):
    """Call the rerank API, return (status, data)."""
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
    test_label: str,
    query: str,
    expected_winner: str,
    image_b64s: dict,
) -> bool:
    """
    Rerank all 4 images against the query.
    Returns True if the expected subject ranks #1.
    """
    sep(f"{test_label}")
    print(f"  Query    : \"{query}\"")
    print(f"  Expected : '{expected_winner}' image should rank #1")
    print(f"  Docs     : {', '.join(image_b64s.keys())}")
    print()

    # Build doc list with index tracking
    labels = list(image_b64s.keys())
    docs = [{"image": image_b64s[lbl]} for lbl in labels]

    status, data = rerank(query, docs)

    if "error" in data and "results" not in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        print(f"  API ERROR: {msg}")
        return False

    results = data.get("results", [])
    if not results:
        print("  WARN: No results returned")
        return False

    print(f"  {'Rank':<5} {'Score':<9} {'Subject':<8} Match?")
    print(f"  {'-'*5} {'-'*9} {'-'*8} {'-'*6}")
    winner_label = None
    for rank, res in enumerate(results, 1):
        idx = res.get("index", 0)
        score = res.get("relevance_score", 0.0)
        subject = labels[idx] if idx < len(labels) else f"idx{idx}"
        is_winner = rank == 1
        match_str = "<-- TOP" if is_winner else ""
        if is_winner:
            winner_label = subject
        print(f"  {rank:<5} {score:<9.4f} {subject:<8} {match_str}")

    passed = winner_label == expected_winner
    verdict = "PASS" if passed else f"FAIL (got '{winner_label}', expected '{expected_winner}')"
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
    # Step 1: Download all images as base64
    # ---------------------------------------------------------------
    sep("Step 1: Downloading images as base64 data URIs")
    image_b64s = {}
    for entry in IMAGE_CATALOG:
        lbl = entry["label"]
        b64 = fetch_as_b64(lbl, entry["url"])
        if b64:
            image_b64s[lbl] = b64
        else:
            print(f"  WARNING: '{lbl}' image failed to download - test may be incomplete")
        time.sleep(1)  # gentle pacing between Wikimedia requests
    print()
    print(f"  Downloaded: {list(image_b64s.keys())}")
    print()

    if len(image_b64s) < 2:
        print("ERROR: Need at least 2 images to run discrimination tests.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Discrimination tests
    # ---------------------------------------------------------------
    results = []

    results.append(run_discrimination_test(
        test_label="Test 1: Find the CAT",
        query="a photograph of a cat",
        expected_winner="cat",
        image_b64s=image_b64s,
    ))
    print()

    results.append(run_discrimination_test(
        test_label="Test 2: Find the DOG",
        query="a photograph of a dog",
        expected_winner="dog",
        image_b64s=image_b64s,
    ))
    print()

    results.append(run_discrimination_test(
        test_label="Test 3: Find the CAR",
        query="an automobile, car or vehicle",
        expected_winner="car",
        image_b64s=image_b64s,
    ))
    print()

    results.append(run_discrimination_test(
        test_label="Test 4: Find the HORSE",
        query="a horse or equine animal",
        expected_winner="horse",
        image_b64s=image_b64s,
    ))
    print()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    sep("Summary")
    passed = sum(results)
    total = len(results)
    print(f"  Passed: {passed}/{total} tests")
    print(f"  Model correctly identified the target image in {passed} out of {total} queries.")
    print()


if __name__ == "__main__":
    main()
