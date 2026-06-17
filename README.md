# EasyRerank 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Backend: llama.cpp / Jina AI / OpenRouter](https://img.shields.io/badge/Backend-llama.cpp%20%7C%20Jina%20AI%20%7C%20OpenRouter-orange.svg)](https://github.com/ggml-org/llama.cpp)

A premium, production-ready, self-contained Python module for local and remote semantic document reranking. Bridging the gap between 200-year-old historical texts (like James Madison's presidential speeches) and modern user search queries, **EasyRerank** enables cross-encoder intelligence through a unified, elegant API.

---

## Key Features

- **Dual-Backend Capabilities**: Auto-routes between a locally running `llama.cpp` server (using models like `Qwen3-Reranker` or `bge-reranker-v2-m3`) and the remote `Jina AI Cloud API` (`jina-reranker-v3`).
- **Vision/Image Reranking** *(v0.2.2)*: Supports mixed text + image document lists for vision-language rerankers (e.g. `nvidia/llama-nemotron-rerank-vl-1b-v2:free` via OpenRouter). Pass `{"image": url}` and `{"text": "..."}` dicts alongside plain strings — the API payload is formatted automatically.
- **Directory Image Processing** *(v0.2.4)*: Scans and batches folders of image files safely (up to 64 images or 30MB per batch, whichever comes first) using `DirectoryImageProcessor` and filters top $N$ candidates from each batch.
- **Robust Text Processing**: Automatically loads, parses, and dynamically chunks `.txt` directories into sentence or paragraph blocks with built-in protection against local context size crashes (512-token limits).
- **Intelligent Pre-filtering**: Provides length-based batched pre-selection (`process_with_batched_top_n`) to extract candidate summaries before sending them to the scoring models.
- **Unified Meta-Wrapper**: The high-level `EasyRanker` wrapper supports seamless list-based in-memory reranking and directory-based file reranking with automatic backend detection, score-caching, and beautiful CLI output tables.
- **MIT Licensed**: Fully open-source and free for commercial and private use.

---

## Architectural & Process Flow

### Object Diagram

The diagram below illustrates the relationship between the unified wrapper, the processing components, the backend classes, and the underlying servers:

```
                                 +--------------------+
                                 |     EasyRanker     |
                                 |  (Unified Wrapper) |
                                 +----------+---------+
                                            |
                                            v
                      +---------------------+---------------------+
                      |                                           |
             [Directory Mode]                              [In-Memory List]
                      |                                           |
                      v                                           v
         +------------+------------+                     +--------+--------+
         | DirectoryTextProcessor  |                     |  Raw Text List  |
         +------------+------------+                     |  [Doc1, Doc2...] |
                      |                                  +--------+--------+
                      v (uses)                                    |
         +------------+------------+                              |
         |       TextParser        |                              |
         | (Sentence/Para Chunking)|                              |
         +------------+------------+                              |
                      |                                           |
                      +---------------------+---------------------+
                                            |
                                            v (Routing)
                                  +---------+---------+
                                  |   Backend Router  |
                                  +----+-----------+--+
                                       |           |
                     [backend='local'] |           | [backend='remote']
                                       v           v
                            +----------+---+   +---+----------+
                            | LocalReranker|   |RemoteReranker|
                            +------+-------+   +---+----------+
                                   |               |
                    (llama.cpp API)|               |(Jina Cloud API)
                                   v               v
                            +------+-------+   +---+----------+
                            | localhost:8080|   | api.jina.ai  |
                            +--------------+   +--------------+
```

### Process Flow

1. **Initialization**: Configure `EasyRanker` with a directory path or lists of strings. 
2. **Text Chunking** (Directory Mode): `DirectoryTextProcessor` reads all `.txt` documents, using `TextParser` to partition sentences under safety boundaries (e.g. 1500 characters) to avoid local server token constraints.
3. **Backend Detection**:
   * If `backend='auto'`, the library queries `http://localhost:8080/v1/models`. 
   * If a local server is responsive, it binds to `LocalReranker` and auto-detects running models.
   * If offline, it looks for a Jina credentials fallback (`api_key` argument, `JINA_API_KEY` environment variable, or a local `api_key` file) and binds to `RemoteReranker`.
4. **Execution & Batching**: Chunks are segmented into standard batch sizes (e.g., 64) and sent to the selected model to calculate cross-encoder relevance scores.
5. **Caching & Formatting**: Combines batched API outputs, sorts documents in descending order of relevance score, displays a formatted CLI table, and caches the output locally.

---

## Installation & Server Setup

### 1. Local Rerank Server (llama.cpp)
To run local reranking, launch `llama-server` with a GGUF cross-encoder model loaded in rerank mode:

```bash
# Start llama.cpp server with embedding/rerank enabled
llama-server -m models/jina-reranker-v3-Q4_K_M.gguf \
  --rerank \
  --pooling rank \
  --port 8080
```

### 2. Remote Reranker Credentials (Jina AI)
To use the Cloud API, retrieve an API key from Jina AI and make it accessible to the library in one of three ways:
1. Pass it directly: `EasyRanker(backend='remote', api_key='jina_...')`
2. Define the environment variable: `export JINA_API_KEY="jina_..."`
3. Create a plain-text file in your project root named `api_key` containing the key.

---

## Sample Use Cases

### Use Case 1: High-Level Unified `EasyRanker` (Auto-Mode)
This sample highlights how `EasyRanker` automatically determines the backend and parses a folder of Speeches to locate concepts that traditional keyword matching fails to bridge:

```python
import os
from EasyRerank import EasyRanker

# 1. Initialize ranker - will automatically detect if llama.cpp is running
# and fallback to Jina Cloud if offline.
speeches_directory = "./Madison"

ranker = EasyRanker(
    documents=speeches_directory,
    backend="auto",
    chunk_size=1,               # 1 sentence per chunk
    max_sentence_length=1500    # Protect local server context bounds
)

# 2. Rerank directory documents on a highly semantic modern concept
query = "Separation of religious institutions from civil government authority"

print(f"Reranking documents for query: '{query}'...")
results = ranker.rerank(
    query=query,
    top_n=3,
    verbose=True  # Automatically prints a formatted ASCII results table
)

# 3. Retrieve latest cached output programmatically
latest_outputs = ranker.get_latest_output()
if latest_outputs:
    top_match = latest_outputs[0]
    print(f"\nTop Match: {top_match['filename']} (Score: {top_match['relevance_score']:.4f})")
```

### Use Case 2: List-Based In-Memory Reranking
Suitable for reranking small sets of candidate texts (such as search database matches) in-memory:

```python
from EasyRerank import EasyRanker

documents = [
    "London is the capital and largest city of the United Kingdom.",
    "Berlin is the capital and largest city of Germany.",
    "Paris is the capital and largest city of France, located on the river Seine.",
    "Tokyo is the capital city of Japan, known for its bustling streets."
]

ranker = EasyRanker(backend="local") # explicitly enforce local llama.cpp
results = ranker.rerank(
    query="What is the capital of France?",
    documents=documents,
    verbose=True
)
```

### Use Case 3: Advanced Directory Batched Pre-Filtering
For large datasets, use the `DirectoryTextProcessor` directly to filter out and accumulate the longest text segments before scoring, staying safe of local server token ceilings:

```python
from EasyRerank import DirectoryTextProcessor, LocalReranker

# Initialize components
processor = DirectoryTextProcessor("./Madison")
reranker = LocalReranker()

# 1. Collect the top 2 longest chunks from every batch of 64, capping at 64 total
top_chunks, reached_limit = processor.process_with_batched_top_n(
    chunk_size=1,
    top_n=2,
    max_limit=64,
    batch_size=64,
    max_sentence_length=1500
)

# 2. Score these filtered candidates
reranked = reranker.rerank_chunks(
    query="Paper currency and financial inflation during military conflicts",
    chunks=top_chunks,
    batch_size=64
)

for rank, item in enumerate(reranked[:3], 1):
    print(f"Rank {rank}: {item['filename']} (ID: {item['chunk_id']}) -> Score: {item['relevance_score']:.4f}")
```

---

### Use Case 4: Vision/Image Reranking (v0.2.2)

For vision-language rerankers like `nvidia/llama-nemotron-rerank-vl-1b-v2:free` (available free on OpenRouter), you can pass mixed lists of image URLs and text documents. The `RemoteReranker` automatically detects dict-format documents and formats the payload correctly:

```python
from EasyRerank import RemoteReranker
import os

# Initialize with the NVIDIA vision-language reranker via OpenRouter
reranker = RemoteReranker(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
    base_url="https://openrouter.ai/api/v1/rerank"
)

# Mix of image URLs and plain text documents
documents = [
    {"image": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"},
    {"text": "A fluffy cat sitting on a windowsill in the sun."},
    {"text": "A street map of downtown Berlin."},
    {"image": "https://example.com/dog.jpg"},
]

results = reranker.rerank(
    query="a photograph of a cat",
    documents=documents,
    top_n=3
)

for result in results:
    doc = result["document"]
    source = doc.get("image") or doc.get("text", "")
    print(f"Rank {result['index']+1} | Score: {result['relevance_score']:.4f} | {source}")
```

> [!NOTE]
> Plain strings mixed into a vision document list are automatically wrapped as `{"text": s}`. Pure string lists (text-only reranking) are sent as-is for full backward compatibility with Cohere and Jina APIs.

---

## Token Limits and Chunk Sizing

**Important:** Local rerank servers (llama.cpp) have a physical batch size limit, typically 512 tokens. If your chunks exceed this limit, you'll see errors like:

```
Rerank API call failed: 500 Server Error
Server response: {'error': {'code': 500, 'message': 'input (1481 tokens) is too large to process. increase the physical batch size (current batch size: 512)', 'type': 'server_error'}}
```

### Why `max_sentence_length` Matters

The `max_sentence_length` parameter (used in `EasyRanker`, `DirectoryTextProcessor`, and `TextParser`) controls the maximum character length for text chunks. This is essential because:

- **Token to Character Ratio**: Approximately 4 characters ≈ 1 token
- **Server Limit**: Most local servers have a 512-token physical batch size
- **Query Overhead**: The query itself consumes tokens, leaving less for documents
- **Safe Limit**: ~1500 characters ensures chunks fit within typical server limits

### Solutions for Token Limit Errors

1. **Recommended: Use `max_sentence_length`**
   ```python
   ranker = EasyRanker(
       documents=my_dir,
       chunking_mode='paragraphs',
       max_sentence_length=1500  # Split long chunks
   )
   ```

2. **Increase Server Context Size**
   Start your server with a larger context:
   ```bash
   llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080 -c 2048
   ```

3. **Use Smaller Chunking Modes**
   - `chunking_mode='sentences'` - Produces smaller, sentence-level chunks
   - `chunking_mode='lines'` - Produces line-level chunks
   - `chunking_mode='paragraphs'` - May produce large chunks; use with `max_sentence_length`

4. **Pre-filter with `process_with_batched_top_n`**
   ```python
   chunks, _ = processor.process_with_batched_top_n(
       top_n=2,
       max_limit=64,
       max_sentence_length=1500  # Critical for long documents
   )
   ```

---

## Supported File Extensions & macOS MIME-Type Handling

### Supported File Extensions
`EasyRerank` dynamically detects and processes text and code documents in a directory. By default, it supports:
- **Documentation & Markup:** `.txt`, `.text`, `.md`, `.markdown`, `.rst`, `.html`, `.htm`, `.css`, `.xml`, `.log`, `.conf`
- **Code files:** `.py`, `.c`, `.cpp`, `.h`, `.hh`, `.hpp`, `.java`, `.js`, `.mjs`, `.ts`, `.tsx`, `.sh`, `.bash`

> [!NOTE]
> Tabular data structures like `.csv` and `.tsv` are **ignored by default** because feeding raw table rows into a semantic reranker loses the column header context.

### Custom Extra Extensions
If you need to include other files (like raw `.json` or `.csv` files) or force-include specific extensions, pass the `extra_extensions` list during initialization:

```python
ranker = EasyRanker(
    documents="./my_dir",
    extra_extensions=[".json", ".csv"]  # Allows dot-prefixed or raw extensions
)
```

### macOS MIME-Type Registry Issue
On macOS systems, the native Launch Services database does not always map common developer extensions (like `.md` or `.markdown`) to a standard `text/` MIME type in Python's default registry. This can result in Python's standard `mimetypes.guess_type()` returning `(None, None)` and ignoring those documents during scanning.

**How EasyRerank handles this:**
1. **Explicit Pre-Registration:** At module import time, `EasyRerank` explicitly registers `.md`, `.markdown`, and `.rst` into Python's active `mimetypes` database.
2. **Robust Extension Fallback:** If the system MIME database still fails to classify a file, `EasyRerank` checks the extension against a hardcoded set of standard plain-text and code extensions before discarding the file.

---

## Included Quick Tests

The project includes ten standard Python verification scripts (`quick_test*.py`) in the root directory to test different components and modes. All of these scripts are fully tracked and checked into the Git repository:

| Script Name | Purpose & Features Tested | Backend Mode |
|:---|:---|:---|
| **[quick_test1.py](quick_test1.py)** | Evaluates file loading and text parsing mechanics. Loads and splits Madison inaugural addresses into chunks and prints the resulting segments and index. | **None** (Tests processing only) |
| **[quick_test2.py](quick_test2.py)** | Basic local query verification. Reranks the first 50 chunks of Madison speeches against the query `"Justice"`. | **Local** (`LocalReranker`) |
| **[quick_test3.py](quick_test3.py)** | Pre-filtering and scaling local tests. Performs batched length-based pre-selection (`top_n=2`) under a safe `1500` character limit to query `"Character of people"`. | **Local** (`LocalReranker`) |
| **[quick_test4.py](quick_test4.py)** | Pre-filtering and scaling remote tests. Mirrors the pre-selection logic of `quick_test3.py` but routes cross-encoder scoring to Jina AI's Cloud API. | **Remote** (`RemoteReranker`) |
| **[quick_test5.py](quick_test5.py)** | High-level `EasyRanker` wrapper test. Tests auto-routing, in-memory list reranking, directory document loading, and cached results caching. | **Both / Auto-routing** (`EasyRanker`) |
| **[quick_test6.py](quick_test6.py)** | Cloud context capabilities demonstration. Forces cloud routing to exploit the 131K token window, handling large chunks (up to `3000` characters) safely. | **Remote Forced** (`EasyRanker` remote) |
| **[quick_test7.py](quick_test7.py)** | Explicit model endpoint routing. Tests local mode forcing the server to evaluate a specific model key (`zz2Felladrin/...`). | **Local Forced** (`EasyRanker` local) |
| **[quick_test8.py](quick_test8.py)** | Chunking mode verification. Tests `DirectoryTextProcessor` with all three chunking modes: `"sentences"`, `"lines"`, and `"paragraphs"`. Validates mode selection and error handling. | **None** (Tests processing only) |
| **[quick_test9.py](quick_test9.py)** | Chunking mode with auto backend. Tests `EasyRanker` with auto-detected backend (local or remote) using all three chunking modes. Verifies that different text segmentation approaches work with the high-level wrapper and `max_sentence_length` splitting. | **Auto** (`EasyRanker`) |
| **[quick_test10.py](quick_test10.py)** | Chunking mode with auto backend. Tests `EasyRanker` with auto-detected backend using all three chunking modes. Demonstrates cloud-based or local reranking with different text segmentation and automatic chunk splitting for long paragraphs/lines. | **Auto** (`EasyRanker`) |
| **[quick_test11.py](quick_test11.py)** | Inline markdown with line-based chunking. Tests processing of structured markdown content (30 Western European foods with descriptions) with 4-line chunking, then feeds all chunks to `EasyRanker` with `backend='auto'` and no model specified. | **Auto** (`EasyRanker`) |
| **[quick_test12.py](quick_test12.py)** | Vision/image reranking. Tests `RemoteReranker` with a mixed list of `{"image": url}` and `{"text": "..."}` documents against the NVIDIA `llama-nemotron-rerank-vl-1b-v2:free` model via OpenRouter. Validates that plain strings are auto-wrapped and that scores are returned for both image and text entries. | **Remote** (`RemoteReranker` / OpenRouter) |
| **[quick_test13.py](quick_test13.py)** | Base64 image input test. Verifies that base64-encoded image data URIs (loaded from local cached files) produce identical scores to their HTTP URL baselines. | **Remote** (`RemoteReranker` / OpenRouter) |
| **[quick_test14.py](quick_test14.py)** | Visual discrimination test. Performs multi-query evaluation against 4 base64 cached images to verify that the model correctly identifies and ranks subjects (cat, dog, horse, car). | **Remote** (`RemoteReranker` / OpenRouter) |
| **[quick_test15.py](quick_test15.py)** | Directory image pre-selection test. Demonstrates `DirectoryImageProcessor` batching images by file size/count and selecting the top scoring candidate per batch. | **Remote** (`RemoteReranker` / OpenRouter) |

---


## Included Shell Utilities

The project includes several shell scripts in the root directory to assist with local server development:

- **[list_local_models.sh](list_local_models.sh)**: Fetches and displays all models loaded onto your local `llama-server`. It automatically filters out and highlights active models containing `"rerank"` (case-insensitive).
  ```bash
  ./list_local_models.sh [optional_port]
  ```
- **[test_rerank_return.sh](test_rerank_return.sh)**: Verifies that your local `llama-server` configuration is successfully returning full document text alongside scores. It prints an educational manual `curl` command structure at the start of execution for training purposes.
  ```bash
  ./test_rerank_return.sh
  ```
- **[rerank_jina_local.sh](rerank_jina_local.sh)**: A sample cURL wrapper script that executes a direct POST request using the `jinaai/jina-reranker-v3-GGUF` model layout on a running local server (`localhost:8080`) to quickly test semantic capital city queries.
  ```bash
  ./rerank_jina_local.sh
  ```

---

## Changelog

### v0.2.4 — Directory Image Processing
- **`DirectoryImageProcessor`**: Implemented class to scan, batch, and load image files from a directory.
  - Supports memory-safe batching (up to 64 images or 30MB payload sizes).
  - Retrieves top $N$ scoring candidates per batch.
- **New Tests & Scripts**:
  - Added [test_directory_image_processor.py](test_directory_image_processor.py) for offline unit testing.
  - Added `quick_test15.py` and `quick_test_130_images.py` to validate directory-based visual reranking.

### v0.2.3 — Local Reranker Bugfixes & Typings
- **`LocalReranker` Indexing Fix**: Fixed index offset mapping when processing documents across multiple batches; local indices are now correctly offset by the batch index to map back to original global indices.
- **`LocalReranker` Compatibility**: Ensured the `document` text dictionary is populated in the returned result list for full parity with the `RemoteReranker` response format.
- **Type Signatures & Safety**: Updated type signatures on `LocalReranker` and `EasyRanker` to support `List[Union[str, Dict[str, Any]]]`. Local reranker checks inputs and raises a clear `ValueError` if image inputs are passed.
- **Enhanced Formatting**: Improved `EasyRanker` verbose output formatting to print clear descriptions for image/base64 documents without raising `AttributeError`.
- **New Tests**: Added [test_local_reranker.py](test_local_reranker.py) for offline unit testing of local batching and type checks.

### v0.2.2 — Vision/Image Reranking
- **`RemoteReranker`**: `_call_rerank_api` and `rerank()` now accept `List[Union[str, Dict[str, Any]]]`.
  - If any document is a `dict` (`{"image": url}` or `{"text": "..."}`), the full payload is sent as a list of dicts — plain strings are automatically wrapped as `{"text": s}`.
  - Pure string lists continue to be sent as-is (backward compatible with Cohere, Jina).
- **Vision Tests & Cache Helpers**:
  - Added `fetch_test_images.py` to cache test images locally.
  - Added `quick_test12.py`, `quick_test13.py` (Base64 test), and `quick_test14.py` (Visual discrimination test) for vision reranking validation.
- Updated `pyproject.toml` description and keywords to reflect multimodal support.

### v0.2.1
- Added automatic chunk splitting via `max_sentence_length` for `lines` and `paragraphs` modes.
- Added `max_length` parameter to `TextParser.lines()`, `paragraphs()`, and all `DirectoryTextProcessor` methods.
- Added `quick_test9`, `quick_test10`, `quick_test11`.

### v0.2.0
- Initial release of `EasyRanker` unified wrapper.
- Auto-routing between `LocalReranker` and `RemoteReranker`.
- Directory-based and in-memory reranking.
- Batched pre-filtering via `process_with_batched_top_n`.

---

## License

This project is licensed under the MIT License.

### MIT License (MIT)

Copyright (c) May 2026 Jon Allen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
