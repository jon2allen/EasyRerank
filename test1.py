from transformers import AutoModel

model = AutoModel.from_pretrained(
    'jinaai/jina-reranker-v3',
    dtype="auto",
    trust_remote_code=True,
)
model.eval()

