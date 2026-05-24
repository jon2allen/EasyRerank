"""Directory Text Processor - Processes .txt files in a directory using TextParser.

This module provides a class to iterate over .txt files in a directory and
generate text chunks using the TextParser class.

Usage:
    from directory_text_processor import DirectoryTextProcessor

    # Initialize with a directory
    processor = DirectoryTextProcessor('/path/to/text/files')

    # List all .txt files
    files = processor.list_files()

    # Process a single file and get chunks
    for chunk in processor.process_file('example.txt'):
        print(chunk)

    # Process multiple specific files
    for file_chunks in processor.process_files(['file1.txt', 'file2.txt']):
        for chunk in file_chunks:
            print(chunk)

    # Process all files in directory
    for file_chunks in processor.process_all():
        for chunk in file_chunks:
            print(chunk)

    # Process with index for later lookup
    iterator = processor.process_with_index(['file1.txt', 'file2.txt'])
    for item in iterator:
        print(f"Chunk {item['chunk_id']} from {item['filename']}")
    # Access index: iterator.index[20]['filename']
    
    # Process with batched top_n collection for reranking
    top_chunks, reached_limit = processor.process_with_batched_top_n(
        filenames=['file1.txt', 'file2.txt'],
        top_n=3,
        max_limit=64
    )
    # Then pass top_chunks to LocalReranker for final reranking
    for item in iterator:
        print(f"Chunk {item['chunk_id']} from {item['filename']}")
    # Later, look up chunk 20's source file
    filename = iterator.index[20]['filename']
"""

import os
from typing import Generator, List, Optional, Tuple, Dict, Any
from .text_parser import TextParser


class DirectoryTextProcessor:
    class ChunkIterator:
        """Iterator that yields chunks and builds an index simultaneously.
        
        This inner class allows iteration over chunks from multiple files
        while maintaining an index mapping chunk_id to its source file and content.
        
        Attributes:
            index: Dictionary mapping chunk_id to {'filename': str, 'chunk': str}
        """

        def __init__(self, processor, filenames, chunk_size, max_sentence_length=None):
            """Initialize the chunk iterator.
            
            Args:
                processor: Parent DirectoryTextProcessor instance
                filenames: List of filenames to process
                chunk_size: Number of sentences per chunk
                max_sentence_length: Optional max sentence character length
            """
            self.processor = processor
            self.filenames = filenames
            self.chunk_size = chunk_size
            self.max_sentence_length = max_sentence_length
            self.index: Dict[int, Dict[str, Any]] = {}
            self._chunk_id = 0

        def __iter__(self):
            """Iterate over chunks, building the index as we go.
            
            Yields:
                dict: {'chunk_id': int, 'filename': str, 'chunk': str}
            """
            for filename in self.filenames:
                self.processor._get_file_path(filename)
                for chunk in self.processor.process_file(filename, self.chunk_size, self.max_sentence_length):
                    result = {
                        'chunk_id': self._chunk_id,
                        'filename': filename,
                        'chunk': chunk
                    }
                    self.index[self._chunk_id] = {
                        'filename': filename,
                        'chunk': chunk
                    }
                    yield result
                    self._chunk_id += 1

    """Processes .txt files in a directory using TextParser to generate chunks.
    
    This class provides methods to list files, process individual or multiple
    .txt files, and iterate over all files in a directory, yielding text chunks.
    
    Attributes:
        directory: The directory path containing .txt files
    """

    def __init__(self, directory: str):
        """Initialize the directory text processor.
        
        Args:
            directory: Path to the directory containing .txt files
            
        Raises:
            ValueError: If the directory does not exist
            NotADirectoryError: If the path is not a directory
        """
        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Path is not a directory: {directory}")
        
        self.directory = directory

    def list_files(self) -> List[str]:
        """List all .txt files in the directory.
        
        Returns:
            List of .txt filenames (relative to the directory)
        """
        files = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.txt'):
                files.append(filename)
        return sorted(files)

    def _get_file_path(self, filename: str) -> str:
        """Get the full path for a file in the directory.
        
        Args:
            filename: The filename (should be a .txt file)
            
        Returns:
            Full path to the file
            
        Raises:
            ValueError: If the file is not a .txt file
            FileNotFoundError: If the file does not exist in the directory
        """
        if not filename.endswith('.txt'):
            raise ValueError(f"File must end with .txt: {filename}")
        
        filepath = os.path.join(self.directory, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filename} in {self.directory}")
        
        return filepath

    def _read_file(self, filename: str) -> str:
        """Read the content of a .txt file.
        
        Args:
            filename: The filename to read
            
        Returns:
            The text content of the file
        """
        filepath = self._get_file_path(filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def process_file(
        self,
        filename: str,
        chunk_size: int = 1,
        max_sentence_length: Optional[int] = None
    ) -> Generator[str, None, None]:
        """Process a single .txt file and yield sentence chunks.
        
        Args:
            filename: The .txt file to process
            chunk_size: Number of sentences per chunk (default: 1)
            max_sentence_length: Optional maximum character length for sentences.
                               Longer sentences will be split (default: None)
            
        Yields:
            str: A chunk of text (1 or more sentences or sentence fragments)
            
        Raises:
            ValueError: If the file is not a .txt file
            FileNotFoundError: If the file does not exist
        """
        text = self._read_file(filename)
        parser = TextParser(text)
        
        for chunk in parser.sentence_chunks(
            chunk_size=chunk_size,
            max_sentence_length=max_sentence_length
        ):
            yield chunk

    def process_files(
        self,
        filenames: List[str],
        chunk_size: int = 1
    ) -> Generator[Tuple[str, Generator[str, None, None]], None, None]:
        """Process multiple specific .txt files and yield chunks for each.
        
        Args:
            filenames: List of .txt filenames to process
            chunk_size: Number of sentences per chunk (default: 1)
            
        Yields:
            Tuple of (filename, generator for chunks) for each file
            
        Raises:
            ValueError: If any file is not a .txt file
            FileNotFoundError: If any file does not exist
        """
        for filename in filenames:
            # Validate file exists and is a .txt
            self._get_file_path(filename)
            
            def chunk_generator():
                for chunk in self.process_file(filename, chunk_size=chunk_size):
                    yield chunk
            
            yield (filename, chunk_generator())

    def process_all(
        self,
        chunk_size: int = 1
    ) -> Generator[Tuple[str, Generator[str, None, None]], None, None]:
        """Process all .txt files in the directory and yield chunks for each.
        
        Args:
            chunk_size: Number of sentences per chunk (default: 1)
            
        Yields:
            Tuple of (filename, generator for chunks) for each .txt file
        """
        files = self.list_files()
        yield from self.process_files(files, chunk_size=chunk_size)

    def process_file_paragraphs(
        self,
        filename: str
    ) -> Generator[str, None, None]:
        """Process a single .txt file and yield paragraphs.
        
        Args:
            filename: The .txt file to process
            
        Yields:
            str: A paragraph from the file
        """
        text = self._read_file(filename)
        parser = TextParser(text)
        
        for paragraph in parser.paragraphs():
            yield paragraph

    def process_file_sections(
        self,
        filename: str,
        section_delimiter: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Process a single .txt file and yield sections.
        
        Args:
            filename: The .txt file to process
            section_delimiter: String that separates sections (default: None)
            
        Yields:
            str: A section from the file
            
        Raises:
            ValueError: If section_delimiter is None
        """
        text = self._read_file(filename)
        if section_delimiter is None:
            raise ValueError("section_delimiter must be provided for sections")
        
        parser = TextParser(text, section_delimiter=section_delimiter)
        
        for section in parser.sections():
            yield section

    def process_with_index(
        self,
        filenames: Optional[List[str]] = None,
        chunk_size: int = 1,
        max_sentence_length: Optional[int] = None
    ) -> 'ChunkIterator':
        """Process files, yielding chunks while building an index.
        
        This method returns a ChunkIterator that yields chunks from the specified
        files (or all .txt files if None) while simultaneously building an index
        mapping chunk_id to {'filename': str, 'chunk': str}.
        
        The index is built incrementally as you iterate, so if you only iterate
        over 2 files, the index only contains those 2 files' chunks.
        
        Args:
            filenames: Specific .txt filenames to process (None = all .txt files)
            chunk_size: Number of sentences per chunk (default: 1)
            max_sentence_length: Optional max sentence character length (default: None)
            
        Returns:
            ChunkIterator: An iterator that yields chunk dicts and builds .index
            
        Usage:
            iterator = processor.process_with_index(['file1.txt', 'file2.txt'])
            for item in iterator:
                print(f"Chunk {item['chunk_id']} from {item['filename']}: {item['chunk']}")
            
            # Access the index at any point
            chunk_20_file = iterator.index[20]['filename']
        """
        files = filenames if filenames is not None else self.list_files()
        return self.ChunkIterator(self, files, chunk_size, max_sentence_length)

    def process_with_batched_top_n(
        self,
        filenames: Optional[List[str]] = None,
        chunk_size: int = 1,
        top_n: int = 3,
        max_limit: int = 64,
        batch_size: int = 64,
        max_sentence_length: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Process files in batches, collecting top_n chunks from each batch.
        
        This method processes files and groups chunks into batches. For each
        batch of chunks, it selects the top_n chunks (by length) and appends
        them to a collected list. If the collected list reaches max_limit,
        processing stops and an error flag is returned.
        
        Args:
            filenames: Specific .txt filenames to process (None = all .txt files)
            chunk_size: Number of sentences per chunk (default: 1)
            top_n: Number of top chunks to keep from each batch (default: 3)
            max_limit: Maximum total chunks to collect (default: 64)
            batch_size: Number of chunks per batch before selecting top_n (default: 64)
            max_sentence_length: Optional max chars per sentence. Longer sentences
                               will be split (default: None, no limit)
            
        Returns:
            Tuple of (collected_top_chunks, reached_limit):
                - collected_top_chunks: List of chunk dicts with keys 
                  'chunk_id', 'filename', 'chunk' from all batches
                - reached_limit: True if max_limit was reached before 
                  processing all files
        """
        files = filenames if filenames is not None else self.list_files()
        
        collected_top_chunks: List[Dict[str, Any]] = []
        reached_limit = False
        current_batch: List[Dict[str, Any]] = []
        chunk_id = 0
        
        # Iterate through all files and chunks
        batch_num = 0
        for filename in files:
            if reached_limit:
                break
                
            filepath = self._get_file_path(filename)
            text = self._read_file(filename)
            parser = TextParser(text)
            
            for chunk in parser.sentence_chunks(
                chunk_size=chunk_size,
                max_sentence_length=max_sentence_length
            ):
                if reached_limit:
                    break
                    
                current_batch.append({
                    'chunk_id': chunk_id,
                    'filename': filename,
                    'chunk': chunk
                })
                chunk_id += 1
                
                # When batch is full, select top_n and check limit
                if len(current_batch) >= batch_size:
                    batch_num += 1
                    # Sort by length (longest first) and take top_n
                    top_from_batch = sorted(
                        current_batch, 
                        key=lambda x: len(x['chunk']), 
                        reverse=True
                    )[:top_n]
                    
                    # Add batch origin to selected chunks
                    for chunk in top_from_batch:
                        chunk['batch_origin'] = batch_num
                    
                    # Show details of selected chunks
                    print(f"  Batch {batch_num}: Processed {len(current_batch)} chunks, "
                          f"selected top {len(top_from_batch)} (by length)")
                    for j, selected in enumerate(top_from_batch, 1):
                        print(f"    Top {j}: ID={selected['chunk_id']}, "
                              f"File={selected['filename']}, "
                              f"Length={len(selected['chunk'])} chars")
                    
                    # Check if adding these would exceed max_limit
                    remaining_space = max_limit - len(collected_top_chunks)
                    if remaining_space <= 0:
                        reached_limit = True
                        print(f"  -> REACHED max_limit of {max_limit}, stopping.")
                        break
                    
                    # Add only what fits
                    if len(top_from_batch) > remaining_space:
                        top_from_batch = top_from_batch[:remaining_space]
                        reached_limit = True
                        print(f"  -> Partial add: {len(top_from_batch)} chunks (reached limit)")
                    
                    collected_top_chunks.extend(top_from_batch)
                    
                    if len(collected_top_chunks) >= max_limit:
                        collected_top_chunks = collected_top_chunks[:max_limit]
                        reached_limit = True
                        break
                    
                    current_batch = []
        
        # Process remaining chunks in the last partial batch
        if current_batch and not reached_limit:
            batch_num += 1
            top_from_batch = sorted(
                current_batch,
                key=lambda x: len(x['chunk']),
                reverse=True
            )[:top_n]
            
            # Add batch origin to selected chunks
            for chunk in top_from_batch:
                chunk['batch_origin'] = batch_num
            
            print(f"  Batch {batch_num} (final): Processed {len(current_batch)} chunks, "
                  f"selected top {len(top_from_batch)} (by length)")
            for j, selected in enumerate(top_from_batch, 1):
                print(f"    Top {j}: ID={selected['chunk_id']}, "
                      f"File={selected['filename']}, "
                      f"Length={len(selected['chunk'])} chars")
            
            remaining_space = max_limit - len(collected_top_chunks)
            if len(top_from_batch) > remaining_space:
                top_from_batch = top_from_batch[:remaining_space]
                reached_limit = True
                print(f"  -> Partial add: {len(top_from_batch)} chunks (reached limit)")
            
            collected_top_chunks.extend(top_from_batch)
            
            if len(collected_top_chunks) >= max_limit:
                collected_top_chunks = collected_top_chunks[:max_limit]
                reached_limit = True
        
        if reached_limit:
            print(
                f"WARNING: Reached max_limit of {max_limit} collected chunks. "
                f"Collected {len(collected_top_chunks)} chunks. "
                f"Some files may not have been fully processed."
            )
        
        return collected_top_chunks, reached_limit
