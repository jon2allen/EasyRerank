#!/bin/bash

# Sample curl command for jina-reranker-v3 on local llama.cpp server
# Usage: ./rerank_jina_local.sh
# Make sure to first start the server with:
#   llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080

curl -X POST \
  http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jinaai/jina-reranker-v3-GGUF:Q4_K_M",
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital and largest city of France.",
      "Berlin is the capital of Germany.",
      "The Eiffel Tower is located in Paris.",
      "London is the capital of the United Kingdom.",
      "Nanjing is the heart of the Ming Empire",
      "Paris has good food",
      "A trip to Paris is expensive"
    ],
    "top_n": 3
  }'

#echo ""
#echo "To start the server:"
#echo "  llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080"
