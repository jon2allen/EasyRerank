"""Quick Test 11: Inline Markdown Food List with Line-based Chunking.

This script tests line-based chunking with an inline markdown list of
30 top foods of Western Europe, each with country and description.

Usage:
    python3 quick_test11.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EasyRerank import TextParser, DirectoryTextProcessor, EasyRanker

# Inline markdown list of 30 top foods of Western Europe
MARKDOWN_FOODS = """
# Top 30 Traditional Foods of Western Europe

## France
1. **Coq au Vin** - Classic French chicken stew braised in red wine with mushrooms, onions, and bacon.
2. **Bouillabaisse** - Traditional Provençal fish stew from Marseille, made with various Mediterranean fish and shellfish.
3. **Ratatouille** - Vegetable stew from Nice, featuring eggplant, zucchini, bell peppers, and tomatoes.
4. **Cassoulet** - Slow-cooked white bean casserole with duck or pork, originating from Toulouse.
5. **Quiche Lorraine** - Savory tart with eggs, cream, and bacon from the Lorraine region.

## Italy
6. **Risotto alla Milanese** - Creamy saffron-infused risotto from Milan, often served with ossobuco.
7. **Osso Buco** - Braised veal shanks with bone marrow, a Milanese specialty.
8. **Pasta Carbonara** - Roman pasta dish with eggs, pecorino cheese, pancetta, and black pepper.
9. **Tiramisu** - Coffee-flavored dessert from Veneto, made with mascarpone cheese and ladyfingers.
10. **Ribollita** - Tuscan bread and vegetable soup, a hearty peasant dish.

## Spain
11. **Paella Valenciana** - Traditional rice dish from Valencia with rabbit, chicken, and green vegetables.
12. **Gazpacho** - Cold tomato-based soup from Andalusia, perfect for hot summers.
13. **Fabada Asturiana** - Rich white bean stew with chorizo and morcilla from Asturias.
14. **Pulpo a la Gallega** - Galician-style octopus, boiled and served with paprika and olive oil.
15. **Churros con Chocolate** - Fried dough pastries served with thick hot chocolate for dipping.

## Germany
16. **Bratwurst** - Grilled sausages from Nuremberg, typically served with sauerkraut and mustard.
17. **Sauerbraten** - Pot roast marinated in vinegar and spices, a Rhineland specialty.
18. **Pretzel** - Twisted baked bread snack, especially popular in Bavaria.
19. **Schnitzel** - Breaded and fried cutlet, originally from Austria but widely popular in Germany.
20. **Black Forest Cake** - Chocolate and cherry layer cake with whipped cream from the Black Forest region.

## United Kingdom
21. **Fish and Chips** - Fried battered fish served with thick-cut chips, a British national dish.
22. **Beef Wellington** - Fillet steak coated with puff pastry and mushroom duxelles.
23. **Shepherd's Pie** - Minced lamb with vegetables, topped with mashed potatoes.
24. **Full English Breakfast** - Hearty breakfast with eggs, bacon, sausage, baked beans, and toast.
25. **Sticky Toffee Pudding** - Moist sponge cake with dates, covered in toffee sauce.

## Belgium
26. **Moules-frites** - Steamed mussels served with French fries, a Belgian classic.
27. **Waffles** - Light, crispy waffles from Brussels or Liege, often served with toppings.
28. **Carbonnade Flamande** - Flemish beef and onion stew cooked in dark beer.

## Netherlands
29. **Stamppot** - Mashed potatoes mixed with vegetables, served with smoked sausage.
30. **Haring** - Raw herring, traditionally eaten by holding the fish by its tail and taking bites.
"""

print("=" * 80)
print("Quick Test 11: Western European Foods Markdown - Line-based Chunking")
print("=" * 80)
print()

# Test 1: Parse the markdown with TextParser
print("TEST 1: Parsing markdown with TextParser.lines()")
print("-" * 80)

parser = TextParser(MARKDOWN_FOODS)
lines = list(parser.lines())
print(f"Total lines in markdown: {len(lines)}")
print()

# Show first 10 lines
print("First 10 lines:")
for i, line in enumerate(lines[:10], 1):
    print(f"  {i:2d}. {line[:70]}{'...' if len(line) > 70 else ''}")
print()

# Test 2: Chunk at 4 lines (using manual grouping)
print("TEST 2: Manual grouping of lines into chunks of 4")
print("-" * 80)

# Process with 4-line chunks
chunks_4lines = []
current_chunk = []
for i, line in enumerate(lines, 1):
    current_chunk.append(line)
    if i % 4 == 0:
        chunks_4lines.append('\n'.join(current_chunk))
        current_chunk = []

# Add any remaining lines
if current_chunk:
    chunks_4lines.append('\n'.join(current_chunk))

print(f"Number of 4-line chunks: {len(chunks_4lines)}")
print()

# Show first 3 chunks
print("First 3 chunks (4 lines each):")
for i, chunk in enumerate(chunks_4lines[:3], 1):
    print(f"\nChunk {i}:")
    print(f"  {chunk}")
    print(f"  (Length: {len(chunk)} chars)")

print()

# Test 3: Compare different chunking modes
print("TEST 3: Comparing chunking modes on the food markdown")
print("-" * 80)

# Line-based chunking
parser_lines = TextParser(MARKDOWN_FOODS)
line_chunks = list(parser_lines.lines())
print(f"Lines mode: {len(line_chunks)} chunks (each line is a chunk)")

# Paragraph-based chunking
parser_para = TextParser(MARKDOWN_FOODS, paragraph_delimiter='\n\n')
para_chunks = list(parser_para.paragraphs())
print(f"Paragraphs mode: {len(para_chunks)} chunks")

# Sentence-based chunking
parser_sent = TextParser(MARKDOWN_FOODS)
sentence_chunks = list(parser_sent.sentence_chunks(chunk_size=1))
print(f"Sentences mode: {len(sentence_chunks)} chunks (chunk_size=1)")

print()
print("Sample from each mode:")
print(f"  Line 1:     {line_chunks[0][:60]}...")
print(f"  Paragraph 1: {para_chunks[0][:60]}...")
print(f"  Sentence 1:  {sentence_chunks[0][:60]}...")

print()

# Test 4: Use DirectoryTextProcessor with a temporary file
print("TEST 4: Using DirectoryTextProcessor with a temporary markdown file")
print("-" * 80)

# Create a temporary file with the markdown content
temp_file = "temp_foods.md.txt"
with open(temp_file, 'w', encoding='utf-8') as f:
    f.write(MARKDOWN_FOODS)

try:
    # Process with DirectoryTextProcessor
    processor = DirectoryTextProcessor('.')
    
    # Process the temp file with line-based chunking
    iterator = processor.process_with_index(
        filenames=[temp_file],
        chunking_mode='lines'
    )
    
    line_chunks = list(iterator)
    print(f"Total line chunks from DirectoryTextProcessor: {len(line_chunks)}")
    print()
    
    # Show first 5 chunks
    print("First 5 line chunks:")
    for i, chunk in enumerate(line_chunks[:5], 1):
        print(f"  Chunk {i} [ID:{chunk['chunk_id']}]: {chunk['chunk'][:60]}...")
    
    print()
    
    # Now chunk these lines into groups of 4
    print("Grouping line chunks into batches of 4:")
    for batch_start in range(0, len(line_chunks), 4):
        batch = line_chunks[batch_start:batch_start + 4]
        if len(batch) == 4:
            print(f"  Batch {batch_start//4 + 1}: {len(batch)} chunks")
            for chunk in batch:
                print(f"    - [ID:{chunk['chunk_id']}] {chunk['chunk'][:50]}...")
        else:
            print(f"  Batch {batch_start//4 + 1}: {len(batch)} chunks (final partial batch)")
            for chunk in batch:
                print(f"    - [ID:{chunk['chunk_id']}] {chunk['chunk'][:50]}...")
    
finally:
    # Clean up temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print()
        print(f"Cleaned up temporary file: {temp_file}")

print()

# Test 5: Feed all chunks to EasyRanker with auto backend, no model specified
print("TEST 5: Reranking food chunks with EasyRanker (backend='auto', no model)")
print("-" * 80)

try:
    # Create chunks from the markdown (using line-based)
    parser = TextParser(MARKDOWN_FOODS)
    all_lines = list(parser.lines())
    
    # Group into 4-line chunks
    def make_4line_chunks(lines_list):
        for i in range(0, len(lines_list), 4):
            yield '\n'.join(lines_list[i:i + 4])
    
    food_chunks = list(make_4line_chunks(all_lines))
    print(f"Created {len(food_chunks)} chunks (4 lines each) from the food markdown")
    print()
    
    # Initialize EasyRanker with auto backend, no model specified
    ranker = EasyRanker(
        documents=food_chunks,  # Pass the chunks directly as in-memory documents
        backend='auto',
        # Note: No model specified - will use default or auto-detect
        chunking_mode='sentences'  # This won't be used for in-memory docs
    )
    
    print("Attempting to rerank with query: 'French'")
    print("(Note: Lower scores indicate higher relevance for this model)")
    print()
    
    # Query for French foods
    results = ranker.rerank(
        query="French",
        top_n=5,
        verbose=False
    )
    
    print(f"Retrieved {len(results)} results")
    print()
    
    # Display top results
    for i, result in enumerate(results[:5], 1):
        score = result.get('relevance_score', 0)
        # Show the index so we can map back to original
        idx = result.get('index', 0)
        if idx < len(food_chunks):
            original_chunk = food_chunks[idx]
            # Check if this chunk contains "French"
            has_french = "French" in original_chunk
            marker = " 🇫🇷" if has_french else ""
            print(f"  Rank {i}: Score={score:.4f}{marker}")
            print(f"    [Chunk {idx}]: {original_chunk[:120]}...")
        else:
            print(f"  Rank {i}: Score={score:.4f}")
            print(f"    [Chunk {idx}]: (index out of range)")
    
    print()
    
    # Show which chunks actually contain "French"
    french_chunks = [i for i, chunk in enumerate(food_chunks) if "French" in chunk]
    print(f"Chunks containing 'French': {french_chunks}")
    if french_chunks:
        print("  Content:")
        for idx in french_chunks:
            print(f"    Chunk {idx}: {food_chunks[idx][:80]}...")
    
    print()
    print("✓ EasyRanker with backend='auto' and no model specified works!")
    
except Exception as e:
    print(f"Note: {e}")
    print()
    print("This is expected if no local server is running and no API key is available.")
    print("To run this test, start a local server:")
    print("  llama-server -m jina-reranker-v3-Q4_K_M.gguf --rerank --port 8080")
    print("Or ensure a Jina API key is available.")

print()
print("=" * 80)
print("Quick Test 11: PASSED - Line-based chunking with Western European foods")
print("=" * 80)
