# add_data.py
import pyarrow as pa
import pyarrow.parquet as pq
import uuid
import re
import os
from add_data_args import *

def preprocess_text(text, preserve_paragraphs=True, normalize_whitespace=True, max_line_length=None):
    """Preprocess text for inclusion in the dataset"""
    
    # Replace multiple newlines with double newlines (preserves paragraph structure)
    if preserve_paragraphs:
        text = re.sub(r'\n{3,}', '\n\n', text)
    else:
        # Or collapse all whitespace into single spaces if not preserving paragraphs
        text = re.sub(r'\s+', ' ', text)
    
    # Trim excessive whitespace
    if normalize_whitespace:
        text = re.sub(r' {2,}', ' ', text)
    
    # Wrap text at specified line length if needed
    if max_line_length and max_line_length > 0:
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_line_length and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        
        if current_line:
            lines.append(' '.join(current_line))
        
        text = '\n'.join(lines)
    
    return text

def add_to_parquet(text, file_path, source_l1, source_l2, source_l3, metadata=None):
    """Add text with hierarchical source structure and metadata to Parquet file"""
    
    # Generate a unique ID
    text_id = str(uuid.uuid4())
    
    # Calculate word count
    word_count = len(text.split())
    
    # Create data dictionary
    data = {
        'id': [text_id],
        'text': [text],
        'source_l1': [source_l1],
        'source_l2': [source_l2], 
        'source_l3': [source_l3],
        'word_count': [word_count],
    }
    
    # Add any additional metadata
    if metadata:
        for key, value in metadata.items():
            data[key] = [value]
    
    # Convert to PyArrow Table
    table = pa.Table.from_pydict(data)
    
    try:
        # Try to append to existing file
        existing_table = pq.read_table(file_path)
        combined_table = pa.concat_tables([existing_table, table])
        pq.write_table(combined_table, file_path, compression='snappy')
        print(f"Added entry {text_id} to existing file {file_path}")
    except (FileNotFoundError, OSError):
        # Create new file if it doesn't exist
        pq.write_table(table, file_path, compression='snappy')
        print(f"Created new file {file_path} with entry {text_id}")
        
    return text_id

def main():
    # Check if text file exists
    if not os.path.exists(TEXT_FILE_PATH):
        print(f"Error: Text file '{TEXT_FILE_PATH}' not found.")
        return
    
    # Read the text file
    with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as file:
        text_content = file.read()
    
    # Preprocess the text
    processed_text = preprocess_text(
        text_content,
        preserve_paragraphs=PRESERVE_PARAGRAPHS,
        normalize_whitespace=NORMALIZE_WHITESPACE,
        max_line_length=MAX_LINE_LENGTH
    )
    
    # Add to Parquet file
    text_id = add_to_parquet(
        processed_text,
        PARQUET_FILE_PATH,
        SOURCE_L1,
        SOURCE_L2,
        SOURCE_L3,
        METADATA
    )
    
    print(f"Successfully added text with ID: {text_id}")
    print(f"Word count: {len(processed_text.split())}")
    print(f"Source path: {SOURCE_L1}/{SOURCE_L2}/{SOURCE_L3}")

if __name__ == "__main__":
    main()
