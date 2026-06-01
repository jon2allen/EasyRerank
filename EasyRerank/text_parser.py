"""
Text Parser - A simple library for iterating over text paragraphs, sections, lines, and sentences.

This library provides generator-based functions and classes to break up text
documents into paragraphs, sections, lines, or sentence chunks, yielding one element at a time.

Usage:
    from text_parser import TextParser, generate_paragraphs, generate_sentence_chunks, generate_lines

    # Class-based usage
    parser = TextParser(your_text, paragraph_delimiter='\n\n')
    for paragraph in parser.paragraphs():
        print(paragraph)

    # Line-based parsing
    parser = TextParser(your_text)
    for line in parser.lines():
        print(line)

    # Simple function usage
    for paragraph in generate_paragraphs(your_text):
        process(paragraph)

    # Line-based parsing with function
    for line in generate_lines(your_text):
        print(line)

    # Section-based parsing
    parser = TextParser(your_text, paragraph_delimiter='\n\n', section_delimiter='===')
    for section in parser.sections():
        print(f"SECTION: {section}")

    # Sentence chunks (2-3 sentences at a time)
    for chunk in parser.sentence_chunks(chunk_size=2):
        print(chunk)
        print("---")

    # Or using standalone function
    for chunk in generate_sentence_chunks(your_text, chunk_size=3):
        process(chunk)
"""

import re
from typing import Generator, List, Optional


class TextParser:
    """A text parser that yields paragraphs and sections using generators.
    
    This class allows you to iterate over a text document's paragraphs or
    sections one at a time using Python generators with yield.
    
    Attributes:
        text: The source text to parse
        paragraph_delimiter: String used to split paragraphs (default: '\\n\\n')
        section_delimiter: Optional string for splitting sections
        strip_whitespace: Whether to strip whitespace from results (default: True)
    """

    def __init__(
        self,
        text: str,
        paragraph_delimiter: str = '\n\n',
        section_delimiter: Optional[str] = None,
        strip_whitespace: bool = True
    ):
        """Initialize the text parser.
        
        Args:
            text: The text content to parse
            paragraph_delimiter: String that separates paragraphs (default: '\\n\\n')
            section_delimiter: Optional string for section separation
            strip_whitespace: Whether to strip whitespace from yields (default: True)
        """
        self.text = text
        self.paragraph_delimiter = paragraph_delimiter
        self.section_delimiter = section_delimiter
        self.strip_whitespace = strip_whitespace

    def paragraphs(self, max_length: Optional[int] = None) -> Generator[str, None, None]:
        """Generator that yields paragraphs one at a time.
        
        Each call to next() or iteration produces a new paragraph.
        Empty paragraphs are skipped.
        Optionally splits long paragraphs that exceed max_length.
        
        Args:
            max_length: Optional maximum character length for paragraphs.
                      Longer paragraphs will be split into smaller fragments (default: None)
        
        Yields:
            str: The next paragraph from the text
        
        Example:
            >>> parser = TextParser("Para 1\\n\\nPara 2")
            >>> for p in parser.paragraphs():
            ...     print(p)
            Para 1
            Para 2
        """
        for para in self.text.split(self.paragraph_delimiter):
            cleaned = para.strip() if self.strip_whitespace else para
            if cleaned or not self.strip_whitespace:
                # Split long paragraphs if max_length is set
                if max_length is not None and len(cleaned) > max_length:
                    for fragment in self._split_text(cleaned, max_length):
                        yield fragment
                else:
                    yield cleaned

    def sections(self) -> Generator[str, None, None]:
        """Generator that yields sections (if section_delimiter is configured).
        
        Each call to next() or iteration produces a new section.
        Empty sections are skipped.
        
        Raises:
            ValueError: If section_delimiter is not set
        
        Yields:
            str: The next section from the text
        
        Example:
            >>> parser = TextParser("Sec1\\n===\\nSec2", section_delimiter='===')
            >>> list(parser.sections())
            ['Sec1', 'Sec2']
        """
        if not self.section_delimiter:
            raise ValueError("section_delimiter must be configured for sections()")
        
        for section in self.text.split(self.section_delimiter):
            cleaned = section.strip() if self.strip_whitespace else section
            if cleaned or not self.strip_whitespace:
                yield cleaned

    def __iter__(self):
        """Allow direct iteration over paragraphs."""
        return self.paragraphs()

    def lines(self, max_length: Optional[int] = None) -> Generator[str, None, None]:
        """Generator that yields lines one at a time.
        
        Splits the text by single newline characters and yields each line.
        Empty lines are skipped by default (unless strip_whitespace=False).
        Optionally splits long lines that exceed max_length.
        
        Args:
            max_length: Optional maximum character length for lines.
                      Longer lines will be split into smaller fragments (default: None)
        
        Yields:
            str: The next line from the text
        
        Example:
            >>> parser = TextParser("Line 1\\nLine 2\\nLine 3")
            >>> for line in parser.lines():
            ...     print(line)
            Line 1
            Line 2
            Line 3
        """
        for line in self.text.split('\n'):
            cleaned = line.strip() if self.strip_whitespace else line
            if cleaned or not self.strip_whitespace:
                # Split long lines if max_length is set
                if max_length is not None and len(cleaned) > max_length:
                    for fragment in self._split_text(cleaned, max_length):
                        yield fragment
                else:
                    yield cleaned
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into fragments of maximum length.
        
        Args:
            text: The text to split
            max_length: Maximum character length per fragment
            
        Returns:
            List of text fragments, each <= max_length characters
        """
        fragments = []
        current = text
        while len(current) > max_length:
            # Find the last space before max_length to avoid breaking words
            split_pos = current[:max_length].rfind(' ')
            if split_pos <= 0:  # No space found, hard split
                split_pos = max_length
            fragments.append(current[:split_pos])
            current = current[split_pos:].lstrip()
        if current:
            fragments.append(current)
        return fragments

    def sentence_chunks(
        self,
        chunk_size: int = 1,
        max_sentence_length: Optional[int] = None
    ) -> Generator[str, None, None]:
        """Generator that yields chunks of sentences.
        
        Splits the text into sentences and yields them in chunks of chunk_size.
        Optionally splits long sentences that exceed max_sentence_length.
        
        Args:
            chunk_size: Number of sentences per chunk (default: 1)
            max_sentence_length: Optional maximum character length for a sentence.
                               If a sentence exceeds this, it will be split into smaller
                               fragments (default: None, no limit)
        
        Yields:
            str: A chunk containing chunk_size sentences (or fragments) joined together
        
        Example:
            >>> text = "Hello world. How are you? I am fine."
            >>> parser = TextParser(text)
            >>> list(parser.sentence_chunks(chunk_size=2))
            ['Hello world. How are you?', 'I am fine.']
            
            >>> # With max length
            >>> text = "This is a very long sentence. Short one."
            >>> list(parser.sentence_chunks(chunk_size=1, max_sentence_length=15))
            ['This is a very', 'long sentence.', 'Short one.']
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        
        # Split into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', self.text)
        
        # Filter and clean
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Split long sentences if max_sentence_length is set
        if max_sentence_length is not None:
            split_sentences = []
            for s in sentences:
                if len(s) <= max_sentence_length:
                    split_sentences.append(s)
                else:
                    # Split by newline first (for document headers)
                    parts = s.split('\n')
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        if len(part) <= max_sentence_length:
                            split_sentences.append(part)
                        else:
                            # Split by period if there are any (fallback)
                            if '.' in part:
                                sub_parts = [p.strip() for p in part.split('.') if p.strip()]
                                for sub in sub_parts:
                                    if len(sub) <= max_sentence_length:
                                        split_sentences.append(sub + '.')
                                    else:
                                        # Still too long, split by word
                                        words = sub.split()
                                        for k in range(0, len(words), max_sentence_length // 10):
                                            fragment = ' '.join(words[k:k + max_sentence_length // 10])
                                            if fragment:
                                                split_sentences.append(fragment)
                            else:
                                # No periods, split by word
                                words = part.split()
                                for k in range(0, len(words), max_sentence_length // 10):
                                    fragment = ' '.join(words[k:k + max_sentence_length // 10])
                                    if fragment:
                                        split_sentences.append(fragment)
            sentences = split_sentences
        
        # Yield chunks
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            if chunk:
                yield ' '.join(chunk)


def generate_paragraphs(
    text: str,
    delimiter: str = '\n\n',
    strip_whitespace: bool = True
) -> Generator[str, None, None]:
    """Simple generator function that yields paragraphs from text.
    
    This is a functional alternative to using the TextParser class.
    Each iteration produces a new paragraph.
    
    Args:
        text: The text content to parse
        delimiter: String that separates paragraphs (default: '\\n\\n')
        strip_whitespace: Whether to strip whitespace from yields (default: True)
    
    Yields:
        str: The next paragraph from the text
    
    Example:
        >>> text = "First para\\n\\nSecond para"
        >>> for p in generate_paragraphs(text):
        ...     print(p)
        First para
        Second para
    """
    for para in text.split(delimiter):
        cleaned = para.strip() if strip_whitespace else para
        if cleaned or not strip_whitespace:
            yield cleaned


def generate_sections(
    text: str,
    delimiter: str,
    strip_whitespace: bool = True
) -> Generator[str, None, None]:
    """Simple generator function that yields sections from text.
    
    Args:
        text: The text content to parse
        delimiter: String that separates sections
        strip_whitespace: Whether to strip whitespace from yields (default: True)
    
    Yields:
        str: The next section from the text
    
    Example:
        >>> text = "Section 1\\n---\\nSection 2"
        >>> for s in generate_sections(text, '---'):
        ...     print(s)
        Section 1
        Section 2
    """
    for section in text.split(delimiter):
        cleaned = section.strip() if strip_whitespace else section
        if cleaned or not strip_whitespace:
            yield cleaned


def generate_lines(
    text: str,
    strip_whitespace: bool = True
) -> Generator[str, None, None]:
    """Simple generator function that yields lines from text.
    
    Splits text by single newline characters and yields each line.
    
    Args:
        text: The text content to parse
        strip_whitespace: Whether to strip whitespace from yields (default: True)
    
    Yields:
        str: The next line from the text
    
    Example:
        >>> text = "Line 1\\nLine 2\\nLine 3"
        >>> for line in generate_lines(text):
        ...     print(line)
        Line 1
        Line 2
        Line 3
    """
    for line in text.split('\n'):
        cleaned = line.strip() if strip_whitespace else line
        if cleaned or not strip_whitespace:
            yield cleaned


def generate_sentence_chunks(
    text: str,
    chunk_size: int = 1,
    max_sentence_length: Optional[int] = None
) -> Generator[str, None, None]:
    """Simple generator function that yields chunks of sentences from text.
    
    Splits text into sentences and yields them in chunks of chunk_size.
    Optionally splits long sentences that exceed max_sentence_length.
    
    Args:
        text: The text content to parse
        chunk_size: Number of sentences per chunk (default: 1)
        max_sentence_length: Optional maximum character length for a sentence.
                               If a sentence exceeds this, it will be split into smaller
                               fragments (default: None, no limit)
    
    Yields:
        str: A chunk containing chunk_size sentences (or fragments) joined together
    
    Example:
        >>> text = "Hello. How are you? I am fine."
        >>> list(generate_sentence_chunks(text, chunk_size=2))
        ['Hello. How are you?', 'I am fine.']
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    
    # Split into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter and clean
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Split long sentences if max_sentence_length is set
    if max_sentence_length is not None:
        split_sentences = []
        for s in sentences:
            if len(s) <= max_sentence_length:
                split_sentences.append(s)
            else:
                # Split by newline first (for document headers)
                parts = s.split('\n')
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if len(part) <= max_sentence_length:
                        split_sentences.append(part)
                    else:
                        # Split by period if there are any (fallback)
                        if '.' in part:
                            sub_parts = [p.strip() for p in part.split('.') if p.strip()]
                            for sub in sub_parts:
                                if len(sub) <= max_sentence_length:
                                    split_sentences.append(sub + '.')
                                else:
                                    # Still too long, split by word
                                    words = sub.split()
                                    for k in range(0, len(words), max_sentence_length // 10):
                                        fragment = ' '.join(words[k:k + max_sentence_length // 10])
                                        if fragment:
                                            split_sentences.append(fragment)
                        else:
                            # No periods, split by word
                            words = part.split()
                            for k in range(0, len(words), max_sentence_length // 10):
                                fragment = ' '.join(words[k:k + max_sentence_length // 10])
                                if fragment:
                                    split_sentences.append(fragment)
        sentences = split_sentences
    
    # Yield chunks
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i + chunk_size]
        if chunk:
            yield ' '.join(chunk)
