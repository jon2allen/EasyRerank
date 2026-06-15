"""Remote Reranker - Uses a remote Cloud API (Jina AI, OpenRouter, etc.) for reranking.

This module provides a class to rerank documents using a remote reranker API.
Supports both pure-text and mixed text/image document lists (vision-language models).

Requirements:
    - API Key (can be passed in or set via JINA_API_KEY / OPENROUTER_API_KEY env var)

Usage:
    from remote_reranker import RemoteReranker

    # Text-only reranking
    reranker = RemoteReranker()
    results = reranker.rerank(
        query="What is the capital of France?",
        documents=["Paris is the capital...", "Berlin is..."],
        batch_size=32
    )

    # Vision/image reranking (nvidia/llama-nemotron-rerank-vl-1b-v2:free)
    reranker = RemoteReranker(
        api_key="sk-or-...",
        model="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        base_url="https://openrouter.ai/api/v1/rerank"
    )
    results = reranker.rerank(
        query="a photograph of a cat",
        documents=[
            {"image": "https://example.com/cat.jpg"},
            {"text": "A fluffy cat on a windowsill."},
            {"text": "A street map of Berlin."},
        ]
    )

    # Results are sorted by relevance score (descending)
    for result in results:
        doc = result['document']
        source = doc.get('image') or doc.get('text', '')
        print(f"Score: {result['relevance_score']}, Source: {source}")

Reference:
    See rerank_api.md for API details.
"""

import os
import sys
import requests
from typing import List, Dict, Any, Optional, Union


class RemoteReranker:
    """Reranks documents using Jina AI's remote Cloud API.
    
    This class communicates with the remote Jina AI API to perform document reranking.
    It supports authentication via a passed-in API key or the JINA_API_KEY environment variable.
    Its interface mirrors LocalReranker for full compatibility.
    
    Attributes:
        api_key: The Jina AI API key.
        model: Model identifier (default: 'jina-reranker-v3')
        timeout: Request timeout in seconds (default: 120)
        base_url: Full base URL for the API (default: 'https://api.jina.ai/v1/rerank')
    """

    # Valid batch sizes (Jina Cloud supports up to 64 documents for jina-reranker-v3)
    VALID_BATCH_SIZES = {16, 32, 64}
    
    # Maximum batch size
    MAX_BATCH_SIZE = 64

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = 'jina-reranker-v3',
        timeout: int = 120,
        base_url: str = 'https://api.jina.ai/v1/rerank'
    ):
        """Initialize the Remote Reranker.
        
        Args:
            api_key: The Jina AI API key. If None, it will look up JINA_API_KEY in the environment.
            model: Model identifier (default: 'jina-reranker-v3')
            timeout: Request timeout in seconds (default: 120)
            base_url: Base URL for the API (default: 'https://api.jina.ai/v1/rerank')
            
        Raises:
            ValueError: If no API key is provided or found in the environment.
        """
        self.api_key = api_key or os.getenv('JINA_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Jina AI API key must be provided either as an argument or via the "
                "JINA_API_KEY environment variable."
            )
        self.model = model
        self.timeout = timeout
        self.base_url = base_url

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
        documents: List[Union[str, Dict[str, Any]]],
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make a single API call to the remote rerank endpoint.

        Supports both text-only and mixed text/image document payloads.
        If any document is a dict (e.g. {"image": url} or {"text": "..."}),
        the entire payload is sent as a list of dicts — plain strings are
        automatically wrapped as {"text": s}.  Pure string lists are sent
        as-is for backward compatibility with Cohere and Jina APIs.

        Args:
            query: The search query
            documents: List of document texts (str) or image/text dicts (dict).
                       Dicts must have either an "image" key (URL or base64 data URI)
                       or a "text" key.  Mixed lists are supported.
                       Maximum MAX_BATCH_SIZE documents per call.
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

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        # If any document is a dict (vision/image mode), format all as dicts.
        # Plain strings are wrapped as {"text": s} so the API receives a
        # homogeneous list of objects.  Pure string lists are left unchanged
        # for backward compatibility (Cohere, Jina accept plain string arrays).
        has_dict_docs = any(isinstance(d, dict) for d in documents)
        if has_dict_docs:
            formatted_docs: List[Any] = [
                d if isinstance(d, dict) else {"text": d}
                for d in documents
            ]
        else:
            formatted_docs = documents  # type: ignore[assignment]

        payload: Dict[str, Any] = {
            'model': self.model,
            'query': query,
            'documents': formatted_docs
        }

        if top_n is not None:
            payload['top_n'] = top_n

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Include server response in error message for debugging
            try:
                error_details = response.json()
                error_msg = f"Remote Rerank API call failed: {e}. Server response: {error_details}"
            except Exception:
                error_msg = f"Remote Rerank API call failed: {e}. Server response: {response.text[:500]}"
            raise requests.exceptions.RequestException(error_msg)
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"Remote Rerank API call failed: {e}"
            )

    def rerank(
        self,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        batch_size: int = 32,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank a list of documents against a query.

        Supports both text-only and mixed text/image document lists.
        See _call_rerank_api for full details on document format.

        This method splits the documents into batches and makes multiple
        API calls if needed, then combines and re-sorts the results by
        relevance score.

        Args:
            query: The search query
            documents: List of document texts (str) or image/text dicts.
                       Example text-only:  ["Paris is the capital...", "Berlin..."]
                       Example vision:     [{"image": "https://..."}, {"text": "..."}]
                       Example mixed:      [{"image": "https://..."}, "plain text"]
            batch_size: Number of documents per API call (16, 32, or 64)
            top_n: Maximum number of results to return (None = return all)

        Returns:
            List of result dictionaries, each containing:
                - index: Original document index
                - relevance_score: Score from the reranker (0.0 to 1.0)
                - document: {'text': ...} or {'image': ...} depending on input
            Sorted by relevance_score in descending order.

        Raises:
            ValueError: If batch_size is invalid or no documents provided
        """
        if not documents:
            return []

        batch_size = self._validate_batch_size(batch_size)

        # Split into batches
        batches = self._batch_documents(documents, batch_size)

        # Collect all results
        all_results: List[Dict[str, Any]] = []

        # Jina Remote API returns index absolute to the current batch.
        # When compiling multiple batches, we need to map the batch-specific index 
        # back to the absolute index in the original `documents` list.
        for batch_idx, batch in enumerate(batches):
            response = self._call_rerank_api(query, batch, top_n=None)
            batch_results = response.get('results', [])
            
            # Map batch-relative index to original global index
            global_offset = batch_idx * batch_size
            for result in batch_results:
                local_index = result.get('index', 0)
                result['index'] = global_offset + local_index
                
            all_results.extend(batch_results)

        # Sort all results by relevance score (descending)
        all_results.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)

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
        
        This is a convenience method for reranking chunks. It extracts the text
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

        # Extract texts
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

    def check_connection(self) -> bool:
        """Check if the remote API is accessible and the API key is valid.
        
        Returns:
            True if connection succeeds and authorization is valid, False otherwise.
        """
        try:
            # We can check connection by making a small request to the endpoint.
            # Rerank a tiny document with a tiny query.
            self._call_rerank_api(
                query="test",
                documents=["test"],
                top_n=1
            )
            return True
        except Exception:
            return False
