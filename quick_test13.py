"""quick_test13.py - Base64 Image Input Test for Vision Reranking

Tests whether the NVIDIA llama-nemotron-rerank-vl-1b-v2:free model
(via OpenRouter) accepts base64-encoded images as data URIs in addition
to remote URLs.

Key findings:
  - Base64 data URIs (data:image/jpeg;base64,...) ARE supported
  - Scores are identical to URL-based requests for the same image
  - API has a 30MB limit on image content

Usage:
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python quick_test13.py
"""

import base64
import os
import sys
import time

import requests

MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
BASE_URL = "https://openrouter.ai/api/v1/rerank"

IMAGE_SOURCES = [
    {
        "label": "Cat (Wikimedia)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
    },
    {
        "label": "Dog (Wikimedia)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/43/Cute_dog.jpg",
    },
]

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

TEXT_DOCS = [
    {"text": "A fluffy cat sitting on a windowsill in the sun."},
    {"text": "A street map of downtown Berlin."},
    {"text": "A dog playing fetch in the park."},
]

QUERY = "a photograph of a cat"


def fetch_image_as_b64(url: str, label: str):
    """Download image and return (data_uri, mime_type) or (None, None)."""
    print(f"  Downloading: {label}")
    print(f"  URL: {url}")
    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=20, allow_redirects=True)
        if resp.status_code == 429:
            print(f"  Rate limited (429) - waiting 3s and retrying...")
            time.sleep(3)
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=20, allow_redirects=True)
        if resp.status_code != 200:
            print(f"  ERROR: HTTP {resp.status_code}")
            return None, None
        content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        if not content_type.startswith("image/"):
            content_type = "image/jpeg"
        size_kb = len(resp.content) // 1024
        if size_kb == 0:
            print(f"  ERROR: Got 0 bytes (CDN likely blocked request)")
            return None, None
        if size_kb > 30 * 1024:
            print(f"  ERROR: {size_kb}KB exceeds 30MB API limit")
            return None, None
        print(f"  OK: {size_kb}KB, MIME: {content_type}")
        b64 = base64.b64encode(resp.content).decode()
        data_uri = f"data:{content_type};base64,{b64}"
        return data_uri, content_type
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None


def raw_rerank(query: str, documents: list, top_n: int = 4):
    """Direct API call - returns (status_code, response_dict)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    payload = {"model": MODEL, "query": query, "documents": documents, "top_n": top_n}
    resp = requests.post(
        BASE_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    return resp.status_code, resp.json()


def print_test_header(query: str, documents: list) -> None:
    """Print the query and a summary of documents being sent."""
    print(f"  Query   : \"{query}\"")
    print(f"  Docs ({len(documents)}):")
    for i, d in enumerate(documents):
        if isinstance(d, dict):
            if "image" in d:
                src = d["image"]
                if src.startswith("data:"):
                    label = f"[base64, {len(src):,} chars]"
                else:
                    label = src if len(src) <= 65 else src[:62] + "..."
                print(f"    [{i}] image : {label}")
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
    print(f"  {'Rank':<5} {'Score':<8} {'Type':<6} Result")
    print(f"  {'-'*5} {'-'*8} {'-'*6} {'-'*45}")
    for rank, res in enumerate(results, 1):
        score = res.get("relevance_score", 0.0)
        doc = res.get("document", {})
        if isinstance(doc, dict):
            if "image" in doc:
                kind = "image"
                src = doc["image"]
                if src.startswith("data:"):
                    display = f"[base64, {len(src):,} chars]"
                else:
                    display = src if len(src) <= 55 else src[:52] + "..."
            elif "text" in doc:
                kind = "text"
                display = doc["text"]
            else:
                kind = "?"
                display = str(doc)
        else:
            kind = "str"
            display = str(doc)
        if len(display) > 55:
            display = display[:52] + "..."
        print(f"  {rank:<5} {score:<8.4f} {kind:<6} {display}")
    print(f"  Total: {len(results)} results | Provider: {data.get('provider', '?')}")


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
    print("quick_test13.py -- Base64 Image Input Test")
    print(f"Model  : {MODEL}")
    print()

    # ---------------------------------------------------------------
    # Step 1: Download test images
    # ---------------------------------------------------------------
    sep("Step 1: Downloading test images")
    images = {}
    for src in IMAGE_SOURCES:
        data_uri, mime = fetch_image_as_b64(src["url"], src["label"])
        images[src["label"]] = {"url": src["url"], "data_uri": data_uri}
        print()

    cat_url = images["Cat (Wikimedia)"]["url"]
    cat_b64 = images["Cat (Wikimedia)"]["data_uri"]
    dog_b64 = images["Dog (Wikimedia)"]["data_uri"]
    second_b64 = dog_b64 if dog_b64 else cat_b64
    second_label = "dog" if dog_b64 else "cat (duplicate - dog rate-limited)"

    # ---------------------------------------------------------------
    # Test A: URL-based baseline
    # ---------------------------------------------------------------
    sep("Test A: URL-based baseline (known working)")
    docs_a = [{"image": cat_url}] + TEXT_DOCS
    print_test_header(QUERY, docs_a)
    status, data = raw_rerank(QUERY, docs_a)
    print_results(status, data)
    print()

    # ---------------------------------------------------------------
    # Test B: Base64 data URI
    # ---------------------------------------------------------------
    sep("Test B: Base64 data URI (single image + text docs)")
    if cat_b64:
        docs_b = [{"image": cat_b64}] + TEXT_DOCS
        print_test_header(QUERY, docs_b)
        status, data = raw_rerank(QUERY, docs_b)
        print_results(status, data)
    else:
        print("  SKIPPED: cat image download failed")
    print()

    # ---------------------------------------------------------------
    # Test C: Mixed URL + base64 + text
    # ---------------------------------------------------------------
    sep(f"Test C: Mixed URL + base64 ({second_label}) + text")
    if cat_b64:
        docs_c = [
            {"image": cat_url},
            {"image": second_b64},
            {"text": "A fluffy cat sitting on a windowsill in the sun."},
            {"text": "A street map of downtown Berlin."},
        ]
        print_test_header(QUERY, docs_c)
        status, data = raw_rerank(QUERY, docs_c)
        print_results(status, data)
    else:
        print("  SKIPPED: cat image download failed")
    print()

    # ---------------------------------------------------------------
    # Test D: Pure base64 images only
    # ---------------------------------------------------------------
    sep(f"Test D: Pure base64 images only (cat + {second_label}, no text)")
    if cat_b64:
        docs_d = [{"image": cat_b64}, {"image": second_b64}]
        print_test_header(QUERY, docs_d)
        status, data = raw_rerank(QUERY, docs_d, top_n=2)
        print_results(status, data)
    else:
        print("  SKIPPED: cat image download failed")
    print()

    sep("All tests complete")
    print()


if __name__ == "__main__":
    main()
