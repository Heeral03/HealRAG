import re
from pathlib import Path

def chunk_text(text: str, chunk_size_words: int = 300, overlap_words: int = 50) -> list[str]:
    """
    Split text into word-based chunks with a specified overlap.
    """
    # Normalize whitespaces
    normalized_text = re.sub(r'\s+', ' ', text).strip()
    words = normalized_text.split(' ')
    
    if len(words) <= chunk_size_words:
        return [normalized_text]
        
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        # Advance the start pointer
        start += (chunk_size_words - overlap_words)
        
        # Avoid infinite loop if overlap >= chunk_size
        if chunk_size_words <= overlap_words:
            break
            
    return chunks

def chunk_directory(corpus_dir: Path, chunk_size_words: int = 300, overlap_words: int = 50) -> list[dict]:
    """
    Read all text files in corpus_dir and split them into chunks.
    Returns a list of dictionaries with text, source, and chunk index.
    """
    all_chunks = []
    file_paths = sorted(list(corpus_dir.glob("*.txt")))
    
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        chunks = chunk_text(content, chunk_size_words, overlap_words)
        for i, chunk_txt in enumerate(chunks):
            all_chunks.append({
                "text": chunk_txt,
                "source": file_path.name,
                "chunk_index": i
            })
            
    return all_chunks

if __name__ == "__main__":
    import config
    print("Testing chunker on seeded corpus...")
    chunks = chunk_directory(config.CORPUS_DIR, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print(f"Sample chunk from {chunks[0]['source']}:\n{chunks[0]['text'][:200]}...")
