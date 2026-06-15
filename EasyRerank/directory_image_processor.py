"""Directory Image Processor - Processes image files in a directory.

This module provides a class to scan a directory for image files, load them,
and perform batched pre-filtering and reranking operations on them.
"""

import os
import sys
import base64
import mimetypes
from typing import List, Optional, Tuple, Dict, Any


class DirectoryImageProcessor:
    """Processes image files in a directory.

    This class provides methods to discover, validate, and batch images
    to be sent to multimodal reranking models.
    
    Attributes:
        directory: The directory path containing the images.
        extra_extensions: List of additional custom extensions to allow.
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

    def __init__(self, directory: str, extra_extensions: Optional[List[str]] = None):
        """Initialize the directory image processor.
        
        Args:
            directory: Path to the directory containing image files.
            extra_extensions: Optional list of additional custom image extensions.
            
        Raises:
            ValueError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """
        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Path is not a directory: {directory}")
        
        self.directory = directory
        self.extra_extensions: List[str] = []
        if extra_extensions:
            for ext in extra_extensions:
                ext_str = str(ext).lower()
                if not ext_str.startswith('.'):
                    ext_str = '.' + ext_str
                self.extra_extensions.append(ext_str)

    def _is_supported_image(self, filename: str) -> bool:
        """Check if the filename has a supported image extension."""
        filename_lower = filename.lower()
        ext = os.path.splitext(filename_lower)[1]
        
        if ext in self.SUPPORTED_EXTENSIONS:
            return True
        if self.extra_extensions and ext in self.extra_extensions:
            return True
            
        return False

    def list_images(self) -> List[str]:
        """List all supported image files in the directory.
        
        Returns:
            List of image filenames (relative to the directory) sorted alphabetically.
        """
        files = []
        for filename in os.listdir(self.directory):
            if self._is_supported_image(filename):
                if os.path.isfile(os.path.join(self.directory, filename)):
                    files.append(filename)
        return sorted(files)

    def process_images_with_batched_top_n(
        self,
        query: str,
        reranker: Any,
        filenames: Optional[List[str]] = None,
        top_n: int = 3,
        max_limit: int = 64,
        max_payload_bytes: int = 30 * 1024 * 1024
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Process image files in batches, collecting top_n images from each batch.
        
        This method groups images into batches such that each batch contains at most
        64 images OR at most 30MB of base64-encoded image data, whichever comes first.
        It then queries the reranker to score the images in the batch, and selects
        the top_n images from that batch.
        
        Args:
            query: The search query.
            reranker: An initialized RemoteReranker instance with vision capabilities.
            filenames: Specific image filenames to process (None = all images in directory).
            top_n: Number of top images to keep from each batch (default: 3).
            max_limit: Maximum total images to collect (default: 64).
            max_payload_bytes: Max payload size in bytes per batch (default: 30MB).
            
        Returns:
            Tuple of (collected_top_images, reached_limit):
                - collected_top_images: List of image dicts with keys:
                  'filename', 'image_path', 'relevance_score', 'batch_origin'
                - reached_limit: True if max_limit was reached before processing all files.
        """
        files = filenames if filenames is not None else self.list_images()
        if not files:
            return [], False
            
        collected_top_images: List[Dict[str, Any]] = []
        reached_limit = False
        
        current_batch_files: List[str] = []
        current_batch_uris: List[str] = []
        current_batch_bytes = 0
        batch_num = 0
        
        def process_batch(batch_files: List[str], batch_uris: List[str], b_num: int) -> List[Dict[str, Any]]:
            if not batch_files:
                return []
            
            # Format documents as dicts with "image" key
            docs = [{"image": uri} for uri in batch_uris]
            
            try:
                # Query reranker
                results = reranker.rerank(query=query, documents=docs, top_n=None)
            except Exception as e:
                print(f"ERROR: Failed to rerank batch {b_num}: {e}", file=sys.stderr)
                return []
                
            # Map results to original filenames
            batch_results = []
            for result in results:
                idx = result.get('index', 0)
                if idx < len(batch_files):
                    fname = batch_files[idx]
                    batch_results.append({
                         'filename': fname,
                         'image_path': os.path.join(self.directory, fname),
                         'relevance_score': result.get('relevance_score', 0.0),
                         'batch_origin': b_num
                    })
            
            # Return top_n sorted by relevance score
            return sorted(batch_results, key=lambda x: x['relevance_score'], reverse=True)[:top_n]
            
        for filename in files:
            if reached_limit:
                break
                
            filepath = os.path.join(self.directory, filename)
            try:
                with open(filepath, 'rb') as f:
                    b64_data = base64.b64encode(f.read()).decode('utf-8')
                ext = os.path.splitext(filename)[1].lower()
                mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                        "gif": "image/gif", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")
                data_uri = f"data:{mime};base64,{b64_data}"
            except Exception as e:
                print(f"WARNING: Skipping unreadable file {filename}: {e}", file=sys.stderr)
                continue
                
            file_bytes = len(data_uri)
            
            # Check if adding this file exceeds batch constraints:
            # 1. 64 images count limit
            # 2. 30MB payload bytes limit
            if len(current_batch_files) >= 64 or (current_batch_bytes + file_bytes) > max_payload_bytes:
                # Process the full batch
                if current_batch_files:
                    batch_num += 1
                    top_from_batch = process_batch(current_batch_files, current_batch_uris, batch_num)
                    
                    # Add to collected
                    remaining_space = max_limit - len(collected_top_images)
                    if remaining_space <= 0:
                        reached_limit = True
                        break
                    if len(top_from_batch) > remaining_space:
                        top_from_batch = top_from_batch[:remaining_space]
                        reached_limit = True
                    
                    collected_top_images.extend(top_from_batch)
                    if len(collected_top_images) >= max_limit:
                        reached_limit = True
                        break
                        
                # Reset batch for the current image
                current_batch_files = [filename]
                current_batch_uris = [data_uri]
                current_batch_bytes = file_bytes
            else:
                current_batch_files.append(filename)
                current_batch_uris.append(data_uri)
                current_batch_bytes += file_bytes
                
        # Process the final partial batch
        if current_batch_files and not reached_limit:
            batch_num += 1
            top_from_batch = process_batch(current_batch_files, current_batch_uris, batch_num)
            
            remaining_space = max_limit - len(collected_top_images)
            if len(top_from_batch) > remaining_space:
                top_from_batch = top_from_batch[:remaining_space]
                reached_limit = True
                
            collected_top_images.extend(top_from_batch)
            if len(collected_top_images) >= max_limit:
                reached_limit = True
                
        return collected_top_images, reached_limit
