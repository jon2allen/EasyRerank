#!/bin/bash

# Test script to verify the rerank server returns the document text that was sent
# Usage: ./test_rerank_return.sh
# Make sure to first start the server with:
#   llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080

echo "Testing rerank server - verifying document text is returned"
echo "=========================================================="
echo ""

cat << 'EOF'
Educational Note: You can execute this manual curl command to query the server directly:
---------------------------------------------------------------------------------------
curl -X POST http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0",
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
    "top_n": 7,
    "return_documents" : true
  }'
---------------------------------------------------------------------------------------
EOF

echo ""

# Send request with known documents
RESPONSE=$(curl -s -X POST \
  http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0",
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
    "top_n": 7,
    "return_documents" : true
  }')

# Pretty print the response
echo "Server Response:"
echo "----------------"
echo "$RESPONSE" | python3 -m json.tool

echo ""
echo "Verification:"
echo "-------------"

# Extract and display the returned documents
python3 -c "
import json, sys
response = json.loads(sys.stdin.read())
print(f'Number of results returned: {len(response.get(\"results\", []))}')
print()
for i, result in enumerate(response.get('results', [])):
    doc_text = result.get('document', {}).get('text', '')
    score = result.get('relevance_score', 0)
    index = result.get('index', 0)
    print(f'Result {i+1} (index={index}, score={score:.4f}):')
    print(f'  {doc_text}')
    print()
" <<< "$RESPONSE"

echo "=========================================================="
echo "If you see the original document texts above, the server is"
echo "returning documents correctly."
