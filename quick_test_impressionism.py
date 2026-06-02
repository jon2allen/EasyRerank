import os
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EasyRerank import EasyRanker

def run_impressionism_test():
    doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'impressionism_docs')
    if not os.path.exists(doc_dir):
        print(f"Error: Directory {doc_dir} does not exist.")
        sys.exit(1)
        
    print("=" * 80)
    import EasyRerank
    print(f"EasyRerank imported from: {EasyRerank.__file__}")
    
    from EasyRerank.directory_text_processor import DirectoryTextProcessor
    proc = DirectoryTextProcessor(doc_dir)
    detected = proc.list_files()
    print(f"Detected files: {detected}")
    
    print("=" * 80)
    # Instantiate EasyRanker
    print(f"\nInstantiating EasyRanker with directory: {doc_dir}\n")
    # By default, it will now load and parse the markdown (.md) files in the directory!
    ranker = EasyRanker(
        documents=doc_dir,
        backend='auto',  # routes to Jina Cloud using api_key file or local llama.cpp server
        chunk_size=1,
        chunking_mode='paragraphs'  # Grouping by paragraphs to match section-level descriptions
    )
    
    # Define test queries designed to match specific paintings
    queries = [
        "ballet dancers tutu dress rehearsing in a studio",
        "water lilies pond reflections in Giverny garden",
        "rain soaked night street lighting reflections on boulevard"
    ]
    
    for query in queries:
        print(f"\nRunning query: \"{query}\"")
        results = ranker.rerank(query=query, top_n=3, verbose=True)
        
        # Verify the top result is highly relevant
        if results:
            top_match = results[0]
            print(f"Top Match: {top_match['filename']} (Score: {top_match['relevance_score']:.4f})")
        else:
            print("No results returned.")

if __name__ == "__main__":
    run_impressionism_test()
