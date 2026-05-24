# Rerank API Guide: llama.cpp Server Mode with Jina AI Reranker v3

## Table of Contents

- [Introduction to Reranking](#introduction-to-reranking)
- [Jina AI Reranker v3 Overview](#jina-ai-reranker-v3-overview)
- [Using Jina AI Reranker v3 API](#using-jina-ai-reranker-v3-api)
  - [API Endpoint](#api-endpoint)
  - [Authentication](#authentication)
  - [Request Format](#request-format)
  - [Response Format](#response-format)
  - [cURL Examples](#curl-examples)
- [llama.cpp Server Mode](#llamacpp-server-mode)
  - [Installation](#installation)
  - [Starting the Server](#starting-the-server)
  - [OpenAI-Compatible Endpoints](#openai-compatible-endpoints)
- [Reranking with llama.cpp](#reranking-with-llamacpp)
  - [Prerequisites](#prerequisites)
  - [Server Configuration](#server-configuration)
  - [Rerank Endpoint](#rerank-endpoint)
  - [cURL Examples](#curl-examples-1)
- [Local vs. Cloud Reranking](#local-vs-cloud-reranking)
- [Comparison Table](#comparison-table)
- [References](#references)

---

## Introduction to Reranking

Reranking is a technique used in information retrieval and search systems to improve the relevance of results. It involves taking an initial set of candidate documents (typically retrieved by a first-stage retriever like BM25 or a bi-encoder) and reordering them based on a more sophisticated relevance scoring model.

**Cross-encoder rerankers** (like Jina AI Reranker v3) jointly encode the query and each document, allowing for rich interactions between them. This leads to more accurate relevance scores but at a higher computational cost. Listwise rerankers take this further by considering all query-document pairs simultaneously within a single context window.

---

## Jina AI Reranker v3 Overview

**jina-reranker-v3** is a state-of-the-art, 0.6-billion-parameter multilingual listwise reranker designed for advanced document retrieval.

### Key Features

| Feature | Description |
|---------|-------------|
| **Architecture** | Listwise reranking (Last but Not Late Interaction) |
| **Parameters** | 0.6B |
| **Context Length** | 131,072 tokens |
| **Max Documents** | 64 documents per request |
| **Multilingual** | Yes - supports 18+ languages |
| **Transformer Layers** | 28 layers (based on Qwen3-0.6B) |

### Performance Highlights

- **MIRACL Benchmark**: 66.50 average score across 18 languages
  - Arabic: 78.69
  - Thai: 81.06
- **BEIR**: 61.94 nDCG@10
- **HotpotQA**: 78.56
- **FEVER**: 93.95

### Advantages

- **10× Smaller** than generative listwise rerankers while matching or exceeding their performance
- **Single Forward Pass**: Scores all documents in one pass, enabling rich cross-document interactions
- **Efficient**: Compact MLP projector architecture optimized for performance

### Downloading the Model for Local Use with llama.cpp

The jina-reranker-v3 model is available in GGUF format on Hugging Face for direct use with llama.cpp:

**Hugging Face Repository:** [jinaai/jina-reranker-v3-GGUF](https://huggingface.co/jinaai/jina-reranker-v3-GGUF)

**Working download command:**
```bash
llama-cli -hf jinaai/jina-reranker-v3-GGUF:Q4_K_M
```

**Alternative download methods:**
```bash
# Using git with LFS
git lfs install
git clone https://huggingface.co/jinaai/jina-reranker-v3-GGUF

# Using Python
huggingface-cli download jinaai/jina-reranker-v3-GGUF jina-reranker-v3-Q4_K_M.gguf

# Using curl (get URL from model files tab)
curl -L -o jina-reranker-v3-Q4_K_M.gguf \
  "https://huggingface.co/jinaai/jina-reranker-v3-GGUF/resolve/main/jina-reranker-v3-Q4_K_M.gguf"
```

**Available quantizations:** Q2_K, Q3_K_M, Q4_0, **Q4_K_M**, Q5_K_M, Q6_K, Q8_0

**Note:** Ensure your llama.cpp is built with `LLAMA_CURL=1` for direct Hugging Face downloads, and with reranking support for the `/v1/rerank` endpoint.

---

## Using Jina AI Reranker v3 API

### API Endpoint

```
POST https://api.jina.ai/v1/rerank
```

### Authentication

Required header:
```
Authorization: Bearer YOUR_JINA_API_KEY
```

Get your API key from: [Jina AI Dashboard](https://jina.ai/dashboard/)

### Request Format

```json
{
  "model": "jina-reranker-v3",
  "query": "your search query here",
  "documents": [
    "Document text 1",
    "Document text 2",
    "Document text 3"
  ],
  "top_n": 3
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model identifier (`jina-reranker-v3`) |
| `query` | string | Yes | - | The search query |
| `documents` | array | Yes | - | List of document texts to rerank |
| `top_n` | integer | No | all | Number of top documents to return |

### Response Format

```json
{
  "results": [
    {
      "index": 1,
      "relevance_score": 0.98,
      "document": {
        "text": "Document text 2"
      }
    },
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": {
        "text": "Document text 1"
      }
    },
    {
      "index": 2,
      "relevance_score": 0.90,
      "document": {
        "text": "Document text 3"
      }
    }
  ]
}
```

### cURL Examples

#### Example 1: Basic Reranking

```bash
curl -X POST \
  https://api.jina.ai/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JINA_API_KEY" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "Organic skincare products for sensitive skin",
    "top_n": 3,
    "documents": [
      "Organic skincare for sensitive skin with aloe vera and chamomile: Imagine the soothing embrace of nature with our organic skincare range, crafted specifically for sensitive skin.",
      "Our new line of synthetic skincare products is designed for all skin types and includes advanced chemical formulations.",
      "Discover our mineral-based makeup, perfect for sensitive skin and providing natural coverage."
    ]
  }'
```

#### Example 2: Multilingual Reranking

```bash
curl -X POST \
  https://api.jina.ai/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JINA_API_KEY" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "¿Cuál es la capital de Francia?",
    "documents": [
      "París es la capital y la ciudad más grande de Francia.",
      "Berlín es la capital de Alemania.",
      "Londres es la capital del Reino Unido."
    ]
  }'
```

#### Example 3: Technical Query Reranking

```bash
curl -X POST \
  https://api.jina.ai/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JINA_API_KEY" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "SLM markdown",
    "documents": [
      "SLM stands for Small Language Model, which is a compact version of large language models.",
      "Markdown is a lightweight markup language for creating formatted text using a plain-text editor.",
      "SLM Markdown refers to the use of small language models to generate or process markdown content."
    ]
  }'
```

#### Python Example

```python
import requests
import os

JINA_API_KEY = os.getenv("JINA_API_KEY", "your-api-key-here")

def rerank_with_jina(query: str, documents: list[str], top_n: int = 3, model: str = "jina-reranker-v3"):
    """Rerank documents using Jina AI Reranker v3 API."""
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# Usage example
if __name__ == "__main__":
    query = "Organic skincare products for sensitive skin"
    documents = [
        "Organic skincare for sensitive skin with aloe vera and chamomile: Imagine the soothing embrace of nature with our organic skincare range, crafted specifically for sensitive skin.",
        "Our new line of synthetic skincare products is designed for all skin types and includes advanced chemical formulations.",
        "Discover our mineral-based makeup, perfect for sensitive skin and providing natural coverage."
    ]
    
    result = rerank_with_jina(query, documents, top_n=3)
    
    print("Reranked Results:")
    for i, item in enumerate(result["results"]):
        print(f"{i+1}. Score: {item['relevance_score']:.4f}")
        print(f"   Document: {item['document']['text'][:100]}...")
        print()

# Async version using aiohttp
async def rerank_with_jina_async(query: str, documents: list[str], top_n: int = 3):
    """Async version for high-throughput applications."""
    import aiohttp
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    payload = {
        "model": "jina-reranker-v3",
        "query": query,
        "documents": documents,
        "top_n": top_n
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            response.raise_for_status()
            return await response.json()
```

---

## llama.cpp Server Mode

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a popular inference framework for running large language models locally. It supports server mode, which exposes an OpenAI-compatible API.

### Installation

#### From Source

```bash
# Clone the repository
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build (requires cmake and a C++ compiler)
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

#### Using Docker

```bash
# Pull the official image
docker pull ghcr.io/ggml-org/llama.cpp:latest

# Or build locally
docker build -t llama.cpp .
```

### Starting the Server

#### Basic Server

```bash
# Navigate to build directory
cd llama.cpp/build/bin/Release

# Start server with a model
./llama-server -m path/to/model.gguf --port 8080
```

#### With Python (llama-cpp-python)

```bash
# Install the package
pip install llama-cpp-python

# Start server
python3 -m llama_cpp.server --model path/to/model.gguf --port 8080
```

### OpenAI-Compatible Endpoints

When running in server mode, llama.cpp exposes the following OpenAI-compatible endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (OpenAI format) |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Generate embeddings |
| `/v1/models` | GET | List available models |
| `/v1/rerank` | POST | Rerank documents (requires reranker model) |

---

## Reranking with llama.cpp

To use reranking with llama.cpp, you need a **reranker model** in GGUF format.

### Prerequisites

1. **Reranker Model**: Download a GGUF-format reranker model:
   - `bge-reranker-v2-m3`
   - `qwen3-reranker-0.6b`
   - `jina-reranker-v3` (when available in GGUF format)

2. **Proper Model Conversion**: Ensure the model was converted with correct metadata:
   - `pooling_type=RANK`
   - `cls.output.weight` tensor present

### Server Configuration

#### Option 1: Command Line Flags

```bash
# Start with reranking enabled
llama-server -m path/to/reranker-model.gguf \
  --rerank \
  --embedding \
  --pooling rank \
  --host 127.0.0.1 \
  --port 8080
```

### Document Processing Limits

The number of documents a llama.cpp rerank server can process is determined by several factors:

#### Context Length (`-c` or `--ctx-size`)
- **Primary limiting factor**: Total tokens available for all requests
- **Default**: Typically 4096, but can be increased (e.g., `-c 16384` or `-c 131072`)
- **Model-specific**: jina-reranker-v3 supports 131,072 token context, allowing up to 64 documents per request
- **Behavior**: If total tokens in a request exceed context size, server returns **HTTP 400 error** (refuses to process, does not truncate)

#### Factors Affecting Document Capacity

| Factor | Description |
|--------|-------------|
| **Context Size** | Set with `-c N`; must be >= total tokens from all documents + query + overhead |
| **Document Size** | Token count per document; longer documents = fewer can fit in context |
| **Hardware Memory** | VRAM/RAM limits maximum context size you can configure |
| **Parallel Requests** | Context is shared; with `-np N` (parallel requests), each gets a slice of total context |
| **Model Architecture** | Reranker models may have specific context requirements (e.g., listwise rerankers need all docs in one forward pass) |
| **Tokenization** | Different tokenizers produce different token counts for the same text |

#### Practical Examples

**Example 1: jina-reranker-v3 (131,072 context)**
```bash
# Can handle up to 64 documents in one request
llama-server -m jina-reranker-v3-Q4_K_M.gguf \
  -c 131072 \
  --rerank \
  --pooling rank \
  --port 8080
```

**Example 2: Custom context for bge-reranker**
```bash
# Set context based on your document sizes
llama-server -m bge-reranker-v2-m3-Q8_0.gguf \
  -c 16384 \
  --rerank \
  --port 8080
```

#### Calculating Your Limit

To estimate how many documents you can process:

```python
# Pseudo-calculation
tokens_per_document = avg_tokens_per_doc  # Estimate based on your data
total_context = 131072  # jina-reranker-v3 context
query_tokens = estimate_query_tokens(query)
available_for_docs = total_context - query_tokens - overhead (typically ~100-200 tokens)
max_documents = available_for_docs // tokens_per_document
```

**Rule of thumb**: For jina-reranker-v3, plan for ~2,000 tokens per document to stay well under the 131K limit with 64 documents.

#### Real-World Example: Apple M4 (16GB RAM)

Based on actual server logs, here are real numbers you may encounter:

```
[53125] 0.00.480.522 I common_init_result: fitting params to device memory ...
[53125] 0.00.887.224 W llama_context: n_ctx_seq (82688) < n_ctx_train (131072) 
                           -- the full capacity of the model will not be utilized
[53125] 0.00.477.585 I   - MTL0    : Apple M4 (10922 MiB, 10922 MiB free)
[53125] 0.00.477.585 I   - CPU     : Apple M4 (16384 MiB, 16384 MiB free)
[53125] 0.02.330.403 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 82688
```

**Interpretation:**
- **Model**: jina-reranker-v3-GGUF:Q4_K_M
- **Hardware**: Apple M4 with 16GB RAM
- **Configured context**: 82,688 tokens per slot (`n_ctx`)
- **Trained context**: 131,072 tokens (`n_ctx_train`)
- **Parallel slots**: 4 (auto-detected based on n_threads=4)
- **Available memory**: ~11GB GPU (MTL) + 16GB CPU

**Practical capacity:**
- With ~82K tokens available, and typical documents of 1,500-2,000 tokens
- You can rerank approximately **40-55 documents per request**
- The server **will reject requests** (HTTP 400) if the total tokens exceed 82,688

**To increase context:**
- Reduce `n_parallel` (fewer slots = more context per slot)
- Use a larger quantization (Q5_K_M, Q6_K, Q8_0 have larger files but may allow more context)
- Close other memory-intensive applications
- Use `--ctx-size` flag to explicitly set context (if hardware allows)

#### Option 2: Using models.ini

Create a `models.ini` file:

```ini
[my-reranker]
path = path/to/reranker-model.gguf
reranking = true
embedding = true
pooling = rank
```

Then start the server:

```bash
llama-server --models models.ini --host 127.0.0.1 --port 8080
```

### Rerank Endpoint

**Endpoint:**
```
POST http://localhost:8080/v1/rerank
```

**Request Format:**
```json
{
  "model": "your-reranker-model-name",
  "query": "Your search query here",
  "documents": [
    "Document text 1",
    "Document text 2",
    "Document text 3"
  ],
  "top_n": 3
}
```

**Response Format:**
```json
{
  "results": [
    {
      "index": 1,
      "relevance_score": 0.98,
      "document": {
        "text": "Document text 2"
      }
    },
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": {
        "text": "Document text 1"
      }
    },
    {
      "index": 2,
      "relevance_score": 0.90,
      "document": {
        "text": "Document text 3"
      }
    }
  ]
}
```

### cURL Examples

#### Example 1: Local Reranking with llama.cpp

```bash
curl -X POST \
  http://127.0.0.1:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital and largest city of France.",
      "Berlin is the capital of Germany.",
      "The Eiffel Tower is located in Paris."
    ],
    "top_n": 3
  }'
```

#### Example 2: Without top_n (return all)

```bash
curl -X POST \
  http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-reranker",
    "query": "Best programming language for AI",
    "documents": [
      "Python is widely used for AI and machine learning.",
      "JavaScript is popular for web development.",
      "Rust offers memory safety guarantees."
    ]
  }'
```

#### Example 3: Using Command Line Embedding

For models that support direct embedding:

```bash
llama-embedding -m qwen3-reranker-0.6b_Q8_0.gguf \
  --embd-normalize -1 \
  -p "<question>\t<document>"
```

#### Python Example

```python
import requests

def rerank_with_llamacpp(
    query: str, 
    documents: list[str], 
    base_url: str = "http://localhost:8080",
    model: str = "bge-reranker-v2-m3",
    top_n: int = 3
):
    """Rerank documents using a local llama.cpp rerank server."""
    url = f"{base_url}/v1/rerank"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# Usage example
if __name__ == "__main__":
    # Make sure llama-server is running:
    # ./llama-server -m path/to/reranker-model.gguf --rerank --port 8080
    
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital and largest city of France.",
        "Berlin is the capital of Germany.",
        "The Eiffel Tower is located in Paris."
    ]
    
    result = rerank_with_llamacpp(query, documents, top_n=3)
    
    print("Reranked Results from Local Server:")
    for i, item in enumerate(result["results"]):
        print(f"{i+1}. Score: {item['relevance_score']:.4f}")
        print(f"   Index: {item['index']}")
        print(f"   Document: {item['document']['text']}")
        print()

# Using with llama-cpp-python directly
try:
    from llama_cpp import Llama
    
    def rerank_local_model(query: str, documents: list[str], model_path: str = "reranker.gguf"):
        """Rerank using llama-cpp-python library directly."""
        llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            embedding=True,
            pooling="rank"
        )
        
        # For reranking, you typically score query-document pairs
        # This is a simplified example - actual implementation depends on model
        results = []
        for doc in documents:
            # Combine query and document for scoring
            prompt = f"Query: {query}\nDocument: {doc}"
            output = llm(prompt, max_tokens=1, logits_all=True)
            # Extract score from logits (model-specific)
            score = output["logits"][0] if "logits" in output else 0.0
            results.append({
                "document": doc,
                "score": score
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
        
except ImportError:
    pass
```

---

## Local vs. Cloud Reranking

### Jina AI Cloud API

**Pros:**
- No setup required
- Always up-to-date with latest models
- Scalable for production workloads
- Managed infrastructure
- Multilingual support out of the box

**Cons:**
- Requires API key
- Rate limits apply
- Internet connection required
- Cost for high-volume usage

### Local llama.cpp Server

**Pros:**
- Complete data privacy
- No API costs
- Works offline
- Full control over models and configuration
- Customizable

**Cons:**
- Requires local hardware resources
- Setup and maintenance overhead
- Model files can be large
- Limited by local GPU/CPU capabilities

---

## Comparison Table

| Feature | Jina AI API | llama.cpp Local |
|---------|-------------|-----------------|
| **Setup** | API key only | Model download + server |
| **Cost** | Pay-as-you-go | Free (hardware costs) |
| **Privacy** | Cloud-hosted | Local only |
| **Internet** | Required | Optional |
| **Model** | jina-reranker-v3 | Any GGUF reranker |
| **Context Length** | 131,072 tokens | Model-dependent |
| **Multilingual** | Yes (18+ languages) | Model-dependent |
| **Listwise** | Yes | Model-dependent |
| **Endpoint** | `https://api.jina.ai/v1/rerank` | `http://localhost:8080/v1/rerank` |

---

## Troubleshooting

### Common Issues with llama.cpp Reranking

**Problem: Near-zero scores**
- Solution: Ensure your GGUF model was properly converted with `pooling_type=RANK` and includes the `cls.output.weight` tensor

**Problem: /v1/rerank endpoint not available**
- Solution: Start server with `--rerank` flag and verify model supports reranking

**Problem: Model not loading**
- Solution: Check model path and ensure it's a valid GGUF file with reranker architecture

### Common Issues with Jina AI API

**Problem: Authentication failed**
- Solution: Verify your API key is correct and has sufficient credits

**Problem: Rate limit exceeded**
- Solution: Implement retry logic with exponential backoff or upgrade your plan

**Problem: Document limit exceeded**
- Solution: Reduce the number of documents per request (max 64 for jina-reranker-v3)

---

## Best Practices

### For Jina AI API

1. **Batch Documents**: Send up to 64 documents per request to maximize efficiency
2. **Pre-filter**: Use a first-stage retriever to reduce candidates before reranking
3. **Cache Results**: Cache reranked results for repeated queries
4. **Error Handling**: Implement retry logic for rate limits and network issues

### For Local llama.cpp

1. **Model Selection**: Choose a reranker model that fits your hardware
2. **Quantization**: Use appropriate quantization (Q4, Q5, Q8) for your use case
3. **Resource Management**: Monitor memory usage, especially with large context windows
4. **Warm-up**: Send a test request to load the model into memory before production use

---

## References

- [Jina AI Reranker Documentation](https://jina.ai/reranker/)
- [Jina Reranker v3 Announcement](https://jina.ai/news/jina-reranker-v3-0-6b-listwise-reranker-for-sota-multilingual-retrieval/)
- [Jina Reranker v3 on Hugging Face](https://huggingface.co/jinaai/jina-reranker-v3)
- [llama.cpp Server Documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [Qwen3 Reranker Setup Guide](https://gist.github.com/VooDisss/42bce4eb5c76d3c325633886c5e348ee)
- [ArXiv: jina-reranker-v3 Paper](https://arxiv.org/abs/2509.25085)

---

## Other Open Source Reranker APIs

This section compares API formats across different reranker models. While many follow similar OpenAI-compatible patterns, there are important differences.

---

### Cohere Rerank API

Cohere offers a managed reranker API with a slightly different format from Jina.

#### Endpoint
```
POST https://api.cohere.ai/v2/rerank
```

#### Authentication
```
Authorization: Bearer <COHERE_API_KEY>
```

#### Request Format
```json
{
  "model": "rerank-english-v3.0",
  "query": "your search query",
  "documents": ["document1", "document2", "document3"],
  "top_n": 3,
  "return_documents": true
}
```

#### Response Format
```json
{
  "id": "cmpl-abc123",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.98,
      "document": {
        "text": "document1"
      }
    },
    {
      "index": 2,
      "relevance_score": 0.95,
      "document": {
        "text": "document3"
      }
    }
  ],
  "meta": {
    "api_version": {
      "version": "2"
    }
  }
}
```

#### Python Example
```python
import requests
import os

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def rerank_with_cohere(query: str, documents: list[str], top_n: int = 3):
    url = "https://api.cohere.ai/v2/rerank"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "rerank-english-v3.0",
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": True
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()
```

#### Key Differences from Jina
- **Endpoint path**: `/v2/rerank` vs `/v1/rerank`
- **Additional field**: `return_documents` to include document text in response
- **Model naming**: Uses versioned model names like `rerank-english-v3.0`
- **Document limit**: Supports up to 1,000 documents per request

---

### Qwen3 Reranker API

Qwen3 reranker models are available through various providers with different endpoints.

#### DeepInfra Endpoint
```
POST https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-4B
```

#### Request Format (DeepInfra)
```json
{
  "queries": ["your search query"],
  "documents": ["document1", "document2", "document3"]
}
```

**Note**: Uses `queries` (plural) instead of `query` (singular).

#### Response Format (DeepInfra)
```json
{
  "scores": [0.98, 0.95, 0.90],
  "input_tokens": 42,
  "request_id": null,
  "inference_status": { ... }
}
```

#### Response Format (Fireworks AI / OpenAI-compatible)
```json
{
  "data": [
    {"index": 0, "score": 0.98},
    {"index": 1, "score": 0.95}
  ],
  "object": "list"
}
```

#### Python Example (DeepInfra)
```python
import requests
import os

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")

def rerank_with_qwen_deepinfra(query: str, documents: list[str]):
    url = "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-4B"
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "queries": [query],
        "documents": documents
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# Fireworks AI (OpenAI-compatible)
def rerank_with_qwen_fireworks(query: str, documents: list[str]):
    url = "https://api.fireworks.ai/inference/v1/rerank"
    headers = {
        "Authorization": f"Bearer {os.getenv('FIREWORKS_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen3-reranker-4b",
        "query": query,
        "documents": documents,
        "top_n": len(documents)
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()
```

#### Key Differences from Jina
- **DeepInfra**: Uses `queries` array (not `query` string), returns just `scores` array
- **Fireworks AI**: OpenAI-compatible format, similar to Jina but with different model naming
- **Local llama.cpp**: Uses `/v1/rerank` with standard OpenAI-compatible format

---

### BGE Reranker (BAAI) API

BAAI's bge-reranker models use OpenAI-compatible API when self-hosted.

#### Local Server Endpoint
```
POST /v1/rerank
```

#### Request Format
```json
{
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "your search query",
  "documents": ["document1", "document2", "document3"]
}
```

#### Response Format
```json
{
  "results": [
    {
      "index": 1,
      "relevance_score": 0.98,
      "document": {
        "text": "document2"
      }
    },
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": {
        "text": "document1"
      }
    }
  ]
}
```

#### Python Example
```python
import requests

def rerank_with_bge(
    query: str, 
    documents: list[str],
    base_url: str = "http://localhost:8000"
):
    url = f"{base_url}/v1/rerank"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": query,
        "documents": documents
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()
```

#### Key Differences from Jina
- **Nearly identical** to Jina's API format when using OpenAI-compatible serving
- Model names follow Hugging Face format (`BAAI/bge-reranker-v2-m3`)
- Can be served via vLLM, FastAPI, or other frameworks

---

### Flag Reranker API

FlagEmbedding reranker (from BAAI) has a simpler API focused on score computation.

#### Local Python Usage (No HTTP API)
```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

# Score query-document pairs
pairs = [
    ["query1", "document1"],
    ["query1", "document2"],
    ["query1", "document3"]
]

scores = reranker.compute_score(pairs)
# Returns: [[score1], [score2], [score3]]
```

#### HTTP Service Format (if wrapped)
```json
{
  "query": "your query",
  "documents": [
    {"id": 1, "text": "document1"},
    {"id": 2, "text": "document2"}
  ]
}
```

#### Key Differences from Jina
- **Primary interface**: Python library, not HTTP API
- **Input format**: Takes pairs of `[query, document]` for scoring
- **Output**: Returns raw scores, not ranked results
- **No standardization**: HTTP wrapper formats vary by implementation

---

### API Comparison Table

| Feature | Jina AI | Qwen3 (DeepInfra) | Qwen3 (Fireworks) | Cohere | BGE (Local) | Flag Reranker |
|---------|---------|------------------|-------------------|--------|-------------|---------------|
| **Base Endpoint** | `api.jina.ai/v1/rerank` | `api.deepinfra.com/v1/inference/Qwen/...` | `api.fireworks.ai/inference/v1/rerank` | `api.cohere.ai/v2/rerank` | `/v1/rerank` | N/A (Python) |
| **Query Field** | `query` | `queries` (array) | `query` | `query` | `query` | `query` |
| **Documents Field** | `documents` | `documents` | `documents` | `documents` | `documents` | `documents` |
| **Model Field** | `model` | None (in URL) | `model` | `model` | `model` | None |
| **Top N Field** | `top_n` | None | `top_n` | `top_n` | Optional | None |
| **Response Format** | `results` with index, score | `scores` array | `data` with index, score | `results` with index, score, document | `results` with index, score, document | Raw scores |
| **Return Documents** | Optional | No | Yes | Configurable | Yes | No |
| **Max Documents** | 64 | Provider-dependent | Provider-dependent | 1,000 | No hard limit | No hard limit |
| **Authentication** | Bearer token | Bearer token | Bearer token | Bearer token | Optional | None |
| **OpenAI-Compatible** | Yes | Partial | Yes | No | Yes | No |
| **Multilingual** | Yes (18+ languages) | Yes | Yes | Yes (model-dependent) | Yes | Yes |

---

### Summary of Key Differences

1. **Jina AI**: Most consistent OpenAI-compatible format, listwise reranking, 131K context
2. **Qwen3**: Varies by provider - DeepInfra uses different field names (`queries` vs `query`), Fireworks uses OpenAI-compatible format
3. **Cohere**: Similar to Jina but with `return_documents` flag and higher document limit
4. **BGE**: OpenAI-compatible when self-hosted, nearly identical to Jina
5. **Flag Reranker**: Primarily a Python library, not standardized HTTP API

**Recommendation**: If you want maximum compatibility across providers, use the OpenAI-compatible format with `query`, `documents`, `model`, and `top_n` fields. Most modern reranker serving implementations support this format.
