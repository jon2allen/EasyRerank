"""EasyRanker - A premium meta-wrapper for routing and executing rerank tasks.

This module provides EasyRanker, which routes between LocalReranker and
RemoteReranker automatically, manages both in-memory and directory documents,
and offers an app.run() style execution pipeline with result caching.
"""

import os
import sys
import re
import requests
from typing import List, Dict, Any, Optional, Union

from .local_reranker import LocalReranker
from .remote_reranker import RemoteReranker
from .directory_text_processor import DirectoryTextProcessor


class EasyRanker:
    """A premium routing and coordination wrapper for local/remote reranking.
    
    Acts as a meta-class that:
      1. Routes to LocalReranker or RemoteReranker automatically or explicitly.
      2. Handles both directory-based document loading and in-memory lists of texts.
      3. Provides an 'app.run()'-style execution with printouts and result caching.
    """

    # Valid chunking modes
    CHUNKING_MODES = {'sentences', 'lines', 'paragraphs'}

    def __init__(
        self,
        documents: Optional[Union[str, List[str]]] = None,
        backend: str = 'auto',  # 'auto', 'local', 'remote'
        api_key: Optional[str] = None,
        host: str = 'localhost',
        port: int = 8080,
        model: Optional[str] = None,
        timeout: int = 120,
        chunk_size: int = 1,
        max_sentence_length: Optional[int] = 1500,
        chunking_mode: str = 'sentences'  # 'sentences' | 'lines' | 'paragraphs'
    ):
        """Initialize EasyRanker.
        
        Args:
            documents: Either a directory path containing .txt files (str) or a list of text strings.
            backend: Reranker backend to use ('local', 'remote', or 'auto').
            api_key: Optional Jina AI API key for remote mode.
            host: Server hostname for local mode.
            port: Server port for local mode.
            model: Model identifier override.
            timeout: Network request timeout.
            chunk_size: Sentences/lines/paragraphs per chunk when processing a directory.
            max_sentence_length: Max character length for sentence/line/paragraph chunks.
            chunking_mode: Text segmentation mode - 'sentences', 'lines', or 'paragraphs' (default: 'sentences').
        """
        if chunking_mode not in self.CHUNKING_MODES:
            raise ValueError(
                f"chunking_mode must be one of {self.CHUNKING_MODES}, got '{chunking_mode}'"
            )
        
        self.documents_source = documents
        self.backend = backend.lower()
        self.api_key = api_key
        self.host = host
        self.port = port
        self.model_override = model
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_sentence_length = max_sentence_length
        self.chunking_mode = chunking_mode

        self.latest_results: List[Dict[str, Any]] = []
        self.backend_instance = self._initialize_backend()

    def _get_api_key(self) -> Optional[str]:
        """Tries to retrieve Jina API Key from argument, env, or local file."""
        if self.api_key:
            return self.api_key

        # 1. Check environment
        key = os.getenv("JINA_API_KEY")
        if key:
            return key

        # 2. Check local api_key file
        api_key_path = os.path.join(os.getcwd(), 'api_key')
        if os.path.exists(api_key_path):
            try:
                with open(api_key_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # Parse JINA_API_KEY = "key" format or raw key
                    match = re.search(r'["\'](jina_[a-zA-Z0-9_]+)["\']', content)
                    if match:
                        return match.group(1)
                    else:
                        parts = content.split("=")
                        return parts[-1].replace('"', '').replace("'", "").strip()
            except Exception:
                pass
        return None

    def _initialize_backend(self) -> Union[LocalReranker, RemoteReranker]:
        """Auto-detects and returns the appropriate reranker backend."""
        # 1. Explicit remote
        if self.backend == 'remote':
            key = self._get_api_key()
            if not key:
                raise ValueError("Remote backend explicitly requested, but no API key was found.")
            return RemoteReranker(
                api_key=key,
                model=self.model_override or 'jina-reranker-v3',
                timeout=self.timeout
            )

        # 2. Explicit local
        if self.backend == 'local':
            return LocalReranker(
                host=self.host,
                port=self.port,
                model=self.model_override,
                timeout=self.timeout
            )

        # 3. Auto routing
        if self.backend == 'auto':
            # Check if local server is running and get its models list
            local_models = []
            server_running = False
            try:
                response = requests.get(f'http://{self.host}:{self.port}/v1/models', timeout=5)
                if response.status_code < 500:
                    server_running = True
                    try:
                        data = response.json()
                        local_models = [m.get('id', '') for m in data.get('data', [])]
                    except Exception:
                        pass
            except requests.exceptions.RequestException:
                pass

            if server_running:
                chosen_local_model = self.model_override
                
                # If no explicit model is overridden, search for a local reranker model
                if not chosen_local_model and local_models:
                    # Filter models containing "rerank" (case-insensitive)
                    rerank_models = [m for m in local_models if 'rerank' in m.lower()]
                    if rerank_models:
                        chosen_local_model = rerank_models[0]
                        print(f"DEBUG: Auto-detected local rerank model: '{chosen_local_model}'")
                
                return LocalReranker(
                    host=self.host,
                    port=self.port,
                    model=chosen_local_model,
                    timeout=self.timeout
                )

            # Local server not found, check for API key for remote
            key = self._get_api_key()
            if key:
                return RemoteReranker(
                    api_key=key,
                    model=self.model_override or 'jina-reranker-v3',
                    timeout=self.timeout
                )

            raise ValueError(
                "Auto backend failed: Local llama.cpp server is not running, "
                "and no remote Jina AI API key was found (env or local file)."
            )

        raise ValueError(f"Unknown backend type: {self.backend}")

    def rerank(
        self,
        query: str,
        documents: Optional[Union[str, List[str]]] = None,
        top_n: Optional[int] = None,
        batch_size: int = 64,
        verbose: bool = True,
        max_sentence_length: Optional[int] = None,
        chunking_mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Run the reranking pipeline.
        
        Like an app.run(), this:
          1. Sets up/overrides the documents.
          2. Performs the reranking using the selected backend.
          3. Caches the output in self.latest_results.
          4. Displays a beautifully formatted ASCII printout if verbose.
          5. Returns the ranked result list.
          
        Args:
            query: The search query string.
            documents: Override documents (directory path or list of strings).
            top_n: Maximum number of results to return.
            batch_size: Number of documents per API batch call.
            verbose: Whether to print formatted results.
            max_sentence_length: Override max character length per chunk.
            chunking_mode: Override chunking mode ('sentences', 'lines', 'paragraphs').
        """
        # Use query-level override if provided, otherwise default to instance-level
        effective_chunking_mode = chunking_mode if chunking_mode is not None else self.chunking_mode
        
        doc_source = documents if documents is not None else self.documents_source
        if not doc_source:
            raise ValueError("No documents provided to rerank (pass to __init__ or rerank()).")

        results: List[Dict[str, Any]] = []
        is_directory_mode = False

        if isinstance(doc_source, str):
            # Directory mode
            if not os.path.exists(doc_source) or not os.path.isdir(doc_source):
                raise ValueError(f"Documents directory path does not exist or is not a directory: {doc_source}")

            is_directory_mode = True
            processor = DirectoryTextProcessor(doc_source)
            
            # Use query-level override if provided, otherwise default to instance-level setting
            max_len = max_sentence_length if max_sentence_length is not None else self.max_sentence_length
            
            iterator = processor.process_with_index(
                filenames=None, 
                chunk_size=self.chunk_size,
                max_sentence_length=max_len,
                chunking_mode=effective_chunking_mode
            )
            chunks = list(iterator)

            # Route to backend chunk processor
            results = self.backend_instance.rerank_chunks(
                query=query,
                chunks=chunks,
                batch_size=batch_size,
                top_n=top_n,
                text_key='chunk'
            )
        else:
            # In-memory list mode
            results = self.backend_instance.rerank(
                query=query,
                documents=doc_source,
                batch_size=batch_size,
                top_n=top_n
            )

        # Cache latest results
        self.latest_results = results

        # Beautiful verbose printing
        if verbose:
            backend_name = "Remote Cloud API" if isinstance(self.backend_instance, RemoteReranker) else "Local Server"
            model_name = self.backend_instance.model or "Default"
            mode_desc = "Directory Chunks" if is_directory_mode else "In-Memory Documents"

            print("=" * 80)
            print(f"EASYRANKER RESULTS FOR QUERY: \"{query}\"")
            print(f"Backend: {backend_name} | Model: {model_name} | Mode: {mode_desc}")
            print("=" * 80)
            print(" Rank |  Score | Source Reference & Snippet")
            print("------+--------+--------------------------------------------------------------")

            if not results:
                print("      |        | No results returned.")
            else:
                for idx, res in enumerate(results, 1):
                    score = res.get('relevance_score', 0.0)
                    
                    if is_directory_mode:
                        filename = res.get('filename', 'Unknown')
                        chunk_id = res.get('chunk_id', 0)
                        text = res.get('chunk', '')
                        ref_line = f"File: {filename} (ID: {chunk_id})"
                    else:
                        ref_line = f"List Index: {res.get('index', 0)}"
                        # Check either structure from remote or local format
                        text = res.get('document', {}).get('text', '') if isinstance(res.get('document'), dict) else doc_source[res.get('index', 0)]

                    # Clean snippet
                    snippet = text.replace('\n', ' ').strip()
                    if len(snippet) > 60:
                        snippet = snippet[:57] + "..."

                    print(f"  {idx:2d}  | {score:.4f} | {ref_line}")
                    print(f"      |        | \"{snippet}\"")
                    if idx < len(results):
                        print("------+--------+--------------------------------------------------------------")

            print("=" * 80)
            print(f"Total results: {len(results)}")
            print("=" * 80)

        return results

    def get_latest_output(self) -> List[Dict[str, Any]]:
        """Convenience method to retrieve the results from the most recent run."""
        return self.latest_results
