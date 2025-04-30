# add_data_args.py

# File path to the text file you want to add
TEXT_FILE_PATH = "paste.txt"

# Output Parquet file path
PARQUET_FILE_PATH = "dataset.parquet"

# Hierarchical source structure
SOURCE_L1 = "curated"
SOURCE_L2 = "opinion" 
SOURCE_L3 = "political_commentary"

# Additional metadata
METADATA = {
    'title': 'China Trade War Commentary',
    'author': 'Unknown',
    'publication_date': '2025-04',
    'language': 'en',
    'topic': 'US-China relations',
    'entities': 'China, Trump, Xi Jinping, Thomas Friedman',
    'quality_score': 0.9,
    # Add any other metadata fields you want
}

# Text preprocessing options
PRESERVE_PARAGRAPHS = True  # Keep paragraph breaks
NORMALIZE_WHITESPACE = True  # Remove excessive whitespace
MAX_LINE_LENGTH = None  # Set to a number to wrap text at that length, or None for no wrapping
