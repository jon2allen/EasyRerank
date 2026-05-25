# EasyRerank 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Backend: llama.cpp / Jina AI](https://img.shields.io/badge/Backend-llama.cpp%20%7C%20Jina%20AI-orange.svg)](https://github.com/ggml-org/llama.cpp)

A premium, production-ready, self-contained Python module for local and remote semantic document reranking. Bridging the gap between 200-year-old historical texts (like James Madison's presidential speeches) and modern user search queries, **EasyRerank** enables cross-encoder intelligence through a unified, elegant API.

---

## Key Features

- **Dual-Backend Capabilities**: Auto-routes between a locally running `llama.cpp` server (using models like `Qwen3-Reranker` or `bge-reranker-v2-m3`) and the remote `Jina AI Cloud API` (`jina-reranker-v3`).
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

## Included Quick Tests

The project includes seven standard Python verification scripts (`quick_test*.py`) in the root directory to test different components and modes. All of these scripts are fully tracked and checked into the Git repository:

| Script Name | Purpose & Features Tested | Backend Mode |
|:---|:---|:---|
| **[quick_test1.py](file:///Users/jon2allen/projects/rerank/quick_test1.py)** | Evaluates file loading and text parsing mechanics. Loads and splits Madison inaugural addresses into chunks and prints the resulting segments and index. | **None** (Tests processing only) |
| **[quick_test2.py](file:///Users/jon2allen/projects/rerank/quick_test2.py)** | Basic local query verification. Reranks the first 50 chunks of Madison speeches against the query `"Justice"`. | **Local** (`LocalReranker`) |
| **[quick_test3.py](file:///Users/jon2allen/projects/rerank/quick_test3.py)** | Pre-filtering and scaling local tests. Performs batched length-based pre-selection (`top_n=2`) under a safe `1500` character limit to query `"Character of people"`. | **Local** (`LocalReranker`) |
| **[quick_test4.py](file:///Users/jon2allen/projects/rerank/quick_test4.py)** | Pre-filtering and scaling remote tests. Mirrors the pre-selection logic of `quick_test3.py` but routes cross-encoder scoring to Jina AI's Cloud API. | **Remote** (`RemoteReranker`) |
| **[quick_test5.py](file:///Users/jon2allen/projects/rerank/quick_test5.py)** | High-level `EasyRanker` wrapper test. Tests auto-routing, in-memory list reranking, directory document loading, and cached results caching. | **Both / Auto-routing** (`EasyRanker`) |
| **[quick_test6.py](file:///Users/jon2allen/projects/rerank/quick_test6.py)** | Cloud context capabilities demonstration. Forces cloud routing to exploit the 131K token window, handling large chunks (up to `3000` characters) safely. | **Remote Forced** (`EasyRanker` remote) |
| **[quick_test7.py](file:///Users/jon2allen/projects/rerank/quick_test7.py)** | Explicit model endpoint routing. Tests local mode forcing the server to evaluate a specific model key (`zz2Felladrin/...`). | **Local Forced** (`EasyRanker` local) |

---


## Included Shell Utilities

The project includes several shell scripts in the root directory to assist with local server development:

- **[list_local_models.sh](file:///Users/jon2allen/projects/rerank/list_local_models.sh)**: Fetches and displays all models loaded onto your local `llama-server`. It automatically filters out and highlights active models containing `"rerank"` (case-insensitive).
  ```bash
  ./list_local_models.sh [optional_port]
  ```
- **[test_rerank_return.sh](file:///Users/jon2allen/projects/rerank/test_rerank_return.sh)**: Verifies that your local `llama-server` configuration is successfully returning full document text alongside scores. It prints an educational manual `curl` command structure at the start of execution for training purposes.
  ```bash
  ./test_rerank_return.sh
  ```
- **[rerank_jina_local.sh](file:///Users/jon2allen/projects/rerank/rerank_jina_local.sh)**: A sample cURL wrapper script that executes a direct POST request using the `jinaai/jina-reranker-v3-GGUF` model layout on a running local server (`localhost:8080`) to quickly test semantic capital city queries.
  ```bash
  ./rerank_jina_local.sh
  ```

---

## License

This project is licensed under the MIT License.

### MIT License (MIT)

Copyright (c) May 2026 Jon Allen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
