"""Local Reranker - Uses a local llama.cpp server with jina-reranker-v3 model.

This module provides a class to rerank documents using a locally running
llama.cpp server with a reranker model (e.g., jina-reranker-v3).

Requirements:
    - Start a local llama.cpp server with a reranker model:
      llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080
    
    - Or use llama-cpp-python:
      python3 -m llama_cpp.server --model jina-reranker-v3-Q4_K_M.gguf --port 8080

Usage:
    from local_reranker import LocalReranker

    # Initialize with default localhost:8080
    reranker = LocalReranker()

    # Rerank with a query and list of documents
    results = reranker.rerank(
        query="What is the capital of France?",
        documents=["Paris is the capital...", "Berlin is..."],
        batch_size=32
    )
    
    # For servers that require model parameter:
    # reranker = LocalReranker(model='jinaai/jina-reranker-v3-GGUF:Q4_K_M')

    # Results are sorted by relevance score (descending)
    for result in results:
        print(f"Score: {result['relevance_score']}, Doc: {result['document']['text']}")

Reference:
    See rerank_api.md for API details.
    See rerank_jina_local.sh for cURL examples.
"""

import sys
import requests
from typing import List, Dict, Any, Optional, Union


class LocalReranker:
    """Reranks documents using a local llama.cpp server with a reranker model.
    
    This class communicates with a locally running llama.cpp server that has
    a reranker model loaded (e.g., jina-reranker-v3). It supports batching
    documents in chunks of 16, 32, or 64 per API call.
    
    Note: Some server configurations require the 'model' parameter to be sent
    in each request. If you get a "model name is missing" error, initialize with:
        LocalReranker(model='jinaai/jina-reranker-v3-GGUF:Q4_K_M')
    
    Attributes:
        host: Server hostname (default: 'localhost')
        port: Server port (default: 8080)
        model: Model identifier (default: None, which omits the parameter.
               Some servers require this to be set.)
        base_url: Full base URL for the API
    """

    # Valid batch sizes (multiples of 64, max is 64 for jina-reranker-v3)
    VALID_BATCH_SIZES = {16, 32, 64}
    
    # Maximum batch size
    MAX_BATCH_SIZE = 64

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8080,
        model: Optional[str] = None,  # None = don't send model parameter
        timeout: int = 120
    ):
        """Initialize the local reranker.
        
        Args:
            host: Server hostname (default: 'localhost')
            port: Server port (default: 8080)
            model: Model identifier for the request (default: None, which omits 
                   the model parameter - use when server has model pre-loaded)
            timeout: Request timeout in seconds (default: 120)
        """
        self.host = host
        self.port = port
        self.model = model
        self.timeout = timeout
        self.base_url = f'http://{host}:{port}/v1/rerank'

    def _validate_batch_size(self, batch_size: int) -> int:
        """Validate and return the batch size.
        
        Args:
            batch_size: Desired batch size (16, 32, or 64)
            
        Returns:
            Validated batch size
            
        Raises:
            ValueError: If batch_size is not in VALID_BATCH_SIZES
        """
        if batch_size not in self.VALID_BATCH_SIZES:
            raise ValueError(
                f"batch_size must be one of {self.VALID_BATCH_SIZES}, got {batch_size}"
            )
        return batch_size

    def _batch_documents(
        self,
        documents: List[str],
        batch_size: int
    ) -> List[List[str]]:
        """Split documents into batches of the specified size.
        
        Args:
            documents: List of document texts
            batch_size: Number of documents per batch
            
        Returns:
            List of batches, each containing up to batch_size documents
        """
        batches = []
        for i in range(0, len(documents), batch_size):
            batches.append(documents[i:i + batch_size])
        return batches

    def _call_rerank_api(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make a single API call to the rerank endpoint.
        
        Args:
            query: The search query
            documents: List of document texts (up to MAX_BATCH_SIZE)
            top_n: Number of top results to return (None = return all)
            
        Returns:
            API response as a dictionary
            
        Raises:
            requests.exceptions.RequestException: If the API call fails
        """
        if len(documents) > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"Cannot send more than {self.MAX_BATCH_SIZE} documents in one call, "
                f"got {len(documents)}"
            )

        payload = {
            'query': query,
            'documents': documents
            # Note: jina-reranker-v3 doesn't return document text, only index and score
        }
        
        if self.model is not None:
            payload['model'] = self.model
        
        if top_n is not None:
            payload['top_n'] = top_n

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Include server response in error message for debugging
            try:
                error_details = response.json()
                error_msg = f"Rerank API call failed: {e}. Server response: {error_details}"
            except:
                error_msg = f"Rerank API call failed: {e}. Server response: {response.text[:500]}"
            raise requests.exceptions.RequestException(error_msg)
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"Rerank API call failed: {e}"
            )

    def rerank(
        self,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        batch_size: int = 32,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank a list of documents against a query.
        
        This method splits the documents into batches and makes multiple
        API calls if needed, then combines and re-sorts the results by
        relevance score.
        
        Args:
            query: The search query
            documents: List of document texts to rerank (str) or text dictionaries.
                       Note: local rerankers do not support image/multimodal inputs.
            batch_size: Number of documents per API call (16, 32, or 64)
            top_n: Maximum number of results to return (None = return all)
            
        Returns:
            List of result dictionaries, each containing:
                - index: Original document index
                - relevance_score: Score from the reranker (0.0 to 1.0)
                - document: {'text': document text}
            Sorted by relevance_score in descending order.
            
        Raises:
            ValueError: If batch_size is invalid, no documents provided, or if image inputs are passed.
        """
        if not documents:
            return []

        # Validate documents and extract plain strings for local API
        plain_docs: List[str] = []
        for idx, doc in enumerate(documents):
            if isinstance(doc, dict):
                if 'image' in doc:
                    raise ValueError(
                        f"LocalReranker does not support image/multimodal documents (found at index {idx}). "
                        "Please use RemoteReranker with a supported vision model (e.g. via OpenRouter)."
                    )
                elif 'text' in doc:
                    plain_docs.append(doc['text'])
                else:
                    raise ValueError(
                        f"Document dictionary at index {idx} must contain either 'text' or 'image' key."
                    )
            elif isinstance(doc, str):
                plain_docs.append(doc)
            else:
                raise TypeError(f"Document at index {idx} must be a string or a dictionary.")

        batch_size = self._validate_batch_size(batch_size)

        # Split into batches
        batches = self._batch_documents(plain_docs, batch_size)

        # Collect all results
        all_results: List[Dict[str, Any]] = []

        for batch_idx, batch in enumerate(batches):
            response = self._call_rerank_api(query, batch, top_n=None)
            # Debug: Check for zero scores in response
            batch_results = response.get('results', [])
            if batch_results:
                scores = [r.get('relevance_score', 0) for r in batch_results]
                if all(s == 0 for s in scores):
                    print(f"DEBUG: Server returned all zero scores for batch of {len(batch)} documents", file=sys.stderr)
                    print(f"DEBUG: Query: {query[:50]}...", file=sys.stderr)
                    print(f"DEBUG: Sample document: {batch[0][:100]}...", file=sys.stderr)
                
                # Apply index offset
                global_offset = batch_idx * batch_size
                for result in batch_results:
                    local_index = result.get('index', 0)
                    global_idx = global_offset + local_index
                    result['index'] = global_idx
                    # Ensure document field is populated for parity with Jina Remote API
                    if 'document' not in result or not result['document']:
                        orig_doc = documents[global_idx]
                        if isinstance(orig_doc, dict):
                            result['document'] = orig_doc
                        else:
                            result['document'] = {'text': orig_doc}
            
            all_results.extend(batch_results)

        # Sort all results by relevance score (descending)
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        # Apply top_n limit if specified
        if top_n is not None and top_n < len(all_results):
            all_results = all_results[:top_n]

        return all_results

    def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32,
        top_n: Optional[int] = None,
        chunk_id_key: str = 'chunk_id',
        text_key: str = 'chunk'
    ) -> List[Dict[str, Any]]:
        """Rerank a list of chunk dictionaries against a query.
        
        This is a convenience method for reranking chunks that come from
        DirectoryTextProcessor.process_with_index(). It extracts the text
        from each chunk, reranks them, and returns results with the original
        chunk metadata preserved.
        
        Args:
            query: The search query
            chunks: List of chunk dictionaries (e.g., from process_with_index)
            batch_size: Number of documents per API call (16, 32, or 64)
            top_n: Maximum number of results to return (None = return all)
            chunk_id_key: Key for chunk ID in the input dicts (default: 'chunk_id')
            text_key: Key for chunk text in the input dicts (default: 'chunk')
            
        Returns:
            List of result dictionaries with added relevance_score,
            sorted by score descending. Preserves original chunk metadata.
        """
        if not chunks:
            return []

        # Extract texts and track original indices
        texts = [chunk[text_key] for chunk in chunks]

        # Rerank the texts
        results = self.rerank(query, texts, batch_size=batch_size, top_n=top_n)

        # Merge results with original chunk data
        reranked_chunks = []
        for result in results:
            original_index = result['index']
            chunk_data = chunks[original_index].copy()
            chunk_data['relevance_score'] = result['relevance_score']
            chunk_data['rank'] = len(reranked_chunks) + 1
            # Add the document text returned by the server for verification
            chunk_data['server_document'] = result.get('document', {}).get('text', '')
            reranked_chunks.append(chunk_data)

        return reranked_chunks

    def check_server(self) -> bool:
        """Check if the rerank server is running and accessible.
        
        Returns:
            True if server responds, False otherwise
        """
        try:
            # Try to get models list (some servers support this)
            # or just check if the endpoint exists
            response = requests.get(
                f'http://{self.host}:{self.port}/v1/models',
                timeout=5
            )
            return response.status_code < 500
        except requests.exceptions.RequestException:
            return False
