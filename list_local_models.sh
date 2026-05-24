#!/bin/bash

# Port can be overridden by command argument (defaults to 8080)
PORT=${1:-8080}
URL="http://localhost:${PORT}/v1/models"

echo "=========================================================="
echo "Fetching models from local server at: $URL"
echo "=========================================================="
echo ""

# Fetch models via curl with 5s timeout
RESPONSE=$(curl -s --max-time 5 "$URL")

if [ $? -ne 0 ] || [ -z "$RESPONSE" ]; then
  echo "Error: Could not reach local server at http://localhost:${PORT}"
  echo "Make sure llama-server is running on port ${PORT}."
  exit 1
fi

# Parse and print results using Python
python3 -c '
import sys, json

try:
    data = json.loads(sys.stdin.read())
except Exception as e:
    print(f"Error parsing JSON response: {e}")
    sys.exit(1)

models = data.get("data", [])
if not models:
    print("No models found on the server.")
    sys.exit(0)

# Print all models
print("All Available Models:")
print("--------------------")
for i, m in enumerate(models, 1):
    model_id = m.get("id", "Unknown")
    print(f" {i:2d}. {model_id}")

print("\n--------------------")

# Print rerank models (case-insensitive)
rerank_models = [m.get("id", "Unknown") for m in models if "rerank" in m.get("id", "").lower()]

print("Reranker Models (containing \"rerank\"):")
print("---------------------------------------")
if not rerank_models:
    print(" (None found)")
else:
    for i, m in enumerate(rerank_models, 1):
        print(f"  - {m}")

print("==========================================================")
' <<< "$RESPONSE"
