"""Quick test program for DirectoryTextProcessor with Madison's actual inaugural speeches."""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EasyRerank import DirectoryTextProcessor

# Use the actual Madison directory
madison_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Madison')

# Process the two inaugural addresses with 2 sentences per chunk
print("Processing Madison's inaugural addresses with chunk_size=2:\n")
processor = DirectoryTextProcessor(madison_dir)

# Get the two inaugural speech files
files = [
    '01_1809-03-04_First_Inaugural_Address.txt',
    '12_1813-03-04_Second_Inaugural_Address.txt'
]

iterator = processor.process_with_index(filenames=files, chunk_size=2)

for item in iterator:
    print(f"chunk_id: {item['chunk_id']}")
    print(f"filename: {item['filename']}")
    print(f"chunk: {item['chunk']}")
    print()

print("=" * 60)
print("Final index:")
for chunk_id, data in iterator.index.items():
    print(f"  {chunk_id}: {data['filename']}")
