"""fetch_test_images.py - Download and cache test images locally

Run this once to populate the images/ directory before running
quick_test13.py or quick_test14.py.

Usage:
    python fetch_test_images.py
"""

import os
import sys
import time
import requests

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

# All test images: filename -> URL
# These are known-good URLs verified to return valid image content.
# - Wikimedia direct (non-thumbnail) URLs work reliably for small images.
# - Unsplash w= param keeps sizes small and within the 30MB API limit.
IMAGE_CATALOG = {
    "cat.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        "description": "Tabby domestic cat (273KB)",
    },
    "dog.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg",
        "description": "Yellow Labrador Retriever (81KB)",
    },
    "horse.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/de/Nokota_Horses_cropped.jpg",
        "description": "Nokota horses in a field (76KB)",
    },
    "car.jpg": {
        "url": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=400&q=80",
        "description": "Porsche sports car - Unsplash (19KB)",
    },
}


def download_image(filename: str, url: str, description: str, retries: int = 3) -> bool:
    """Download one image to images/<filename>. Returns True on success."""
    dest = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        size_kb = os.path.getsize(dest) // 1024
        print(f"  [cached] {filename:<12} {size_kb:>5}KB  {description}")
        return True

    print(f"  [fetch ] {filename:<12}          {description}")
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=20, allow_redirects=True)
            if resp.status_code == 429:
                wait = 4 * (attempt + 1)
                print(f"           Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"           ERROR: HTTP {resp.status_code}")
                return False
            if len(resp.content) == 0:
                print(f"           ERROR: 0 bytes received")
                return False
            with open(dest, "wb") as f:
                f.write(resp.content)
            size_kb = len(resp.content) // 1024
            print(f"           Saved {size_kb}KB -> images/{filename}")
            return True
        except Exception as e:
            print(f"           ERROR: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    print(f"           FAILED after {retries} attempts")
    return False


def fetch_all(names: list = None) -> dict:
    """
    Download all (or named subset of) images to images/.
    Returns dict mapping filename -> True/False success.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    catalog = IMAGE_CATALOG if names is None else {
        k: v for k, v in IMAGE_CATALOG.items() if k in names
    }
    results = {}
    for i, (filename, meta) in enumerate(catalog.items()):
        results[filename] = download_image(filename, meta["url"], meta["description"])
        if i < len(catalog) - 1:
            time.sleep(1)  # gentle pacing between requests
    return results


def load_image_b64(filename: str) -> str:
    """Load a cached image and return as a base64 data URI, or None."""
    import base64
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    # Detect MIME from extension
    ext = os.path.splitext(filename)[1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def image_path(filename: str) -> str:
    """Return the absolute path to a cached image file."""
    return os.path.join(IMAGES_DIR, filename)


def image_size_kb(filename: str) -> int:
    """Return file size in KB (0 if missing)."""
    path = os.path.join(IMAGES_DIR, filename)
    return os.path.getsize(path) // 1024 if os.path.exists(path) else 0


if __name__ == "__main__":
    print()
    print("fetch_test_images.py — Downloading test image cache")
    print(f"Destination: {IMAGES_DIR}/")
    print()
    results = fetch_all()
    print()
    ok = sum(results.values())
    total = len(results)
    print(f"Done: {ok}/{total} images ready in images/")
    if ok < total:
        failed = [f for f, ok in results.items() if not ok]
        print(f"Failed: {failed}")
        sys.exit(1)
    print()
