import re
from pathlib import Path

def recursive_split_text(text: str, max_words: int = 300, overlap_words: int = 50) -> list[str]:
    """
    Recursively splits a text block if it exceeds max_words using natural text separators:
    Paragraphs (\n\n) -> Lines (\n) -> Sentences (. ! ?) -> Words.
    Applies word overlap when merging or splitting.
    """
    words = text.strip().split()
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []

    # Priority separators
    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        splits = [s for s in text.split(sep) if s.strip()]
        if len(splits) > 1:
            chunks = []
            current_words = []
            current_str = []
            
            for piece in splits:
                piece_words = piece.strip().split()
                if len(current_words) + len(piece_words) <= max_words:
                    current_words.extend(piece_words)
                    current_str.append(piece)
                else:
                    if current_str:
                        chunks.append(sep.join(current_str).strip())
                    # Check if single piece is still too large
                    if len(piece_words) > max_words:
                        sub_chunks = recursive_split_text(piece, max_words, overlap_words)
                        chunks.extend(sub_chunks)
                        current_words = []
                        current_str = []
                    else:
                        # Apply overlap from previous piece if applicable
                        overlap_tail = current_words[-overlap_words:] if len(current_words) >= overlap_words else current_words
                        current_words = overlap_tail + piece_words
                        current_str = [" ".join(overlap_tail), piece] if overlap_tail else [piece]

            if current_str and " ".join(current_words) not in chunks:
                chunks.append(sep.join(current_str).strip())

            if chunks:
                return [c for c in chunks if c.strip()]

    # Fallback to word-level sliding window if no separator works
    words = text.split()
    chunks = []
    start = 0
    step = max(1, max_words - overlap_words)
    while start < len(words):
        chunks.append(" ".join(words[start:start + max_words]))
        start += step
    return chunks


def chunk_text(text: str, chunk_size_words: int = 300, overlap_words: int = 50) -> list[str]:
    r"""
    Structure-Aware & Recursive Chunker:
    1. First identifies statutory/structural markers (Article \d+, Chapter [IVXLCDM]+, Section \d+).
    2. Splits along statutory boundaries.
    3. Recursively splits any structural section exceeding chunk_size_words.
    """
    # Regex pattern for statutory structural markers
    structure_pattern = r'(?=(?:Chapter\s+[IVXLCDM\d]+|Article\s+\d+|Section\s+\d+))'
    sections = re.split(structure_pattern, text, flags=re.IGNORECASE)
    sections = [s.strip() for s in sections if s and len(s.strip()) > 10]

    if not sections:
        sections = [text]

    final_chunks = []
    for sec in sections:
        sec_words = sec.split()
        if len(sec_words) <= chunk_size_words:
            final_chunks.append(sec)
        else:
            sub_chunks = recursive_split_text(sec, max_words=chunk_size_words, overlap_words=overlap_words)
            final_chunks.extend(sub_chunks)

    return final_chunks if final_chunks else [text.strip()]

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
