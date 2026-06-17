import os
import sys
import base64

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EasyRerank import RemoteReranker

def load_api_key():
    # 1. Check environment variable first
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    
    # 2. Walk up parent directories to look for .env file or chatybot/.env
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        ".env",
        "../.env",
        "../../.env",
        "../chatybot/.env",
        "../../chatybot/.env"
    ]
    for rel_path in search_paths:
        abs_path = os.path.abspath(os.path.join(script_dir, rel_path))
        if os.path.exists(abs_path):
            with open(abs_path, "r") as f:
                for line in f:
                    if line.strip().startswith("OPENROUTER_API_KEY="):
                        val = line.strip().split("=", 1)[1]
                        return val.strip("'\"")
    return None

def encode_image_to_data_uri(path):
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
        
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def main():
    api_key = load_api_key()
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment, local .env, or parent/sibling directories.")
        sys.exit(1)
        
    # Directories (fully relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "images")
    sibling_claim_dir = os.path.abspath(os.path.join(script_dir, "..", "nvidia_rerank", "claim_images"))
    
    # List images to check
    existing_names = ["car.jpg", "cat.jpg", "dog.jpg", "horse.jpg"]
    claim_names = [
        "personal_injury_claim.jpg",
        "war_damage_claim.jpg",
        "va_disability.jpg",
        "va_intent.jpg",
        "dol_ls203.jpg",
        "dol_ls201.jpg"
    ]
    
    documents = []
    metadata = []
    
    # Add existing images
    for name in existing_names:
        path = os.path.join(images_dir, name)
        if os.path.exists(path):
            documents.append({"image": encode_image_to_data_uri(path)})
            metadata.append({"filename": name, "origin": "non-document"})
            
    # Add claim images
    for name in claim_names:
        path = os.path.join(images_dir, name)
        if not os.path.exists(path):
            path = os.path.join(sibling_claim_dir, name)
            
        if os.path.exists(path):
            documents.append({"image": encode_image_to_data_uri(path)})
            # Label origin based on file name prefix
            if name.startswith("va_"):
                origin = "Veterans Affairs (VA)"
            elif name.startswith("dol_"):
                origin = "Department of Labor (DOL)"
            elif name.startswith("war_"):
                origin = "UK War Damage Commission"
            else:
                origin = "Generic Personal Injury"
            metadata.append({"filename": name, "origin": origin})
            
    if not documents:
        print(f"ERROR: No images found. Make sure you have JPEGs in {images_dir} or {sibling_claim_dir}.")
        sys.exit(1)
        
    print(f"Loaded {len(documents)} total images for testing.")
    
    # Initialize Reranker
    reranker = RemoteReranker(
        api_key=api_key,
        model="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        base_url="https://openrouter.ai/api/v1/rerank"
    )
    
    # Test Queries
    queries = [
        "a U.S. Department of Labor form or Office of Workers' Compensation Programs claim document",
        "a document from the Department of Veterans Affairs or VA benefits claim form"
    ]
    
    for query in queries:
        print("\n" + "="*85)
        print(f"QUERY: '{query}'")
        print("="*85)
        
        try:
            # Rerank
            results = reranker.rerank(
                query=query,
                documents=documents,
                batch_size=64
            )
            
            # Print results
            for rank, res in enumerate(results, 1):
                idx = res["index"]
                score = res["relevance_score"]
                meta = metadata[idx]
                print(f"Rank {rank:2d}: Score = {score:.6f} | File = {meta['filename']} (Origin: {meta['origin']})")
        except Exception as e:
            print(f"Error executing rerank: {e}")

if __name__ == "__main__":
    main()
