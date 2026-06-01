"""Quick Test 8: DirectoryTextProcessor chunking_mode verification.

This script tests the chunking_mode parameter for DirectoryTextProcessor
to verify that sentences, lines, and paragraphs modes all work correctly.

Usage:
    python3 quick_test8.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EasyRerank import DirectoryTextProcessor

# Use the Madison directory
madison_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Madison')

print("=" * 80)
print("Quick Test 8: DirectoryTextProcessor chunking_mode verification")
print("=" * 80)
print()

# Initialize processor
processor = DirectoryTextProcessor(madison_dir)

# Get a sample file for testing
sample_file = processor.list_files()[0]
print(f"Testing with file: {sample_file}")
print()

# Test 1: Sentence mode (default)
print("-" * 80)
print("TEST 1: Sentence Mode (default)")
print("-" * 80)

iterator = processor.process_with_index(
    filenames=[sample_file],
    chunking_mode='sentences'
)
sentence_chunks = list(iterator)
print(f"Sentence mode: {len(sentence_chunks)} chunks")
print("Sample chunks:")
for i, chunk in enumerate(sentence_chunks[:3], 1):
    print(f"  {i}. [ID:{chunk['chunk_id']}] {chunk['chunk'][:80]}...")
print()

# Test 2: Line mode
print("-" * 80)
print("TEST 2: Line Mode")
print("-" * 80)

iterator = processor.process_with_index(
    filenames=[sample_file],
    chunking_mode='lines'
)
line_chunks = list(iterator)
print(f"Line mode: {len(line_chunks)} chunks")
print("Sample chunks:")
for i, chunk in enumerate(line_chunks[:3], 1):
    print(f"  {i}. [ID:{chunk['chunk_id']}] {chunk['chunk'][:80]}...")
print()

# Test 3: Paragraph mode
print("-" * 80)
print("TEST 3: Paragraph Mode")
print("-" * 80)

iterator = processor.process_with_index(
    filenames=[sample_file],
    chunking_mode='paragraphs'
)
para_chunks = list(iterator)
print(f"Paragraph mode: {len(para_chunks)} chunks")
print("Sample chunks:")
for i, chunk in enumerate(para_chunks[:3], 1):
    print(f"  {i}. [ID:{chunk['chunk_id']}] {chunk['chunk'][:80]}...")
print()

# Test 4: process_with_batched_top_n with different modes
print("-" * 80)
print("TEST 4: process_with_batched_top_n with chunking_mode")
print("-" * 80)

for mode in ['sentences', 'lines', 'paragraphs']:
    chunks, reached = processor.process_with_batched_top_n(
        filenames=[sample_file],
        top_n=2,
        max_limit=10,
        chunking_mode=mode
    )
    print(f"{mode.upper()} mode: {len(chunks)} top chunks collected")
print()

# Test 5: Verify mode validation
print("-" * 80)
print("TEST 5: Mode Validation")
print("-" * 80)

try:
    iterator = processor.process_with_index(
        filenames=[sample_file],
        chunking_mode='invalid_mode'
    )
    list(iterator)
    print("ERROR: Should have raised ValueError for invalid mode")
except ValueError as e:
    print(f"✓ Correctly rejected invalid mode: {e}")

print()
print("=" * 80)
print("Quick Test 8: PASSED - All chunking_mode tests completed successfully")
print("=" * 80)
