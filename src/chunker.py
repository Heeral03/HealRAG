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
    Attaches full document metadata from sidecar JSON files if available.
    Returns a list of dictionaries with text, source, chunk_index, and document metadata.
    """
    import json
    all_chunks = []
    file_paths = sorted(list(corpus_dir.glob("*.txt")))
    
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for sidecar JSON metadata
        json_path = file_path.with_suffix(".json")
        doc_metadata = {}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    doc_metadata = json.load(jf)
            except Exception as e:
                print(f"Warning: Failed to load metadata for {file_path.name}: {e}")
            
        chunks = chunk_text(content, chunk_size_words, overlap_words)
        for i, chunk_txt in enumerate(chunks):
            chunk_data = {
                "text": chunk_txt,
                "source": file_path.name,
                "chunk_index": i,
                "source_url": doc_metadata.get("source_url", ""),
                "publisher": doc_metadata.get("publisher", "Unknown"),
                "jurisdiction": doc_metadata.get("jurisdiction", "Global"),
                "document_type": doc_metadata.get("document_type", "regulator_guidance"),
                "publication_date": doc_metadata.get("publication_date", ""),
                "effective_from": doc_metadata.get("effective_from", ""),
                "version": doc_metadata.get("version", ""),
                "article": doc_metadata.get("article", ""),
                "retrieved_at": doc_metadata.get("retrieved_at", "2026-09-03"),
                "content_hash": doc_metadata.get("content_hash", ""),
                "hierarchy_rank": doc_metadata.get("hierarchy_rank", 2),
                "source_hierarchy": doc_metadata.get("source_hierarchy", "primary law > regulator guidance > official technical standard > secondary commentary")
            }
            all_chunks.append(chunk_data)
            
    return all_chunks

if __name__ == "__main__":
    import config
    print("Testing chunker on seeded corpus...")
    chunks = chunk_directory(config.CORPUS_DIR, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print(f"Sample chunk from {chunks[0]['source']}:\n{chunks[0]['text'][:200]}...")
