import re
from typing import Dict, List

class KnowledgeRefiner:
    """
    Decomposes retrieved document chunks into individual sentences and filters out
    irrelevant surrounding noise (Knowledge Stripping per the CRAG architecture).
    Retains only key supporting sentences to prevent prompt distraction.
    """

    def __init__(self, min_sentence_score: float = 0.25):
        self.min_sentence_score = min_sentence_score

    def decompose_sentences(self, text: str) -> List[str]:
        """Splits a document chunk into clean sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def refine_chunks(
        self, query: str, chunks: List[Dict]
    ) -> List[Dict]:
        """
        Processes a list of chunks, extracting and returning only high-relevance sentences.
        Returns refined chunks containing condensed text.
        """
        query_terms = set(
            term.lower()
            for term in query.replace("?", "").replace(".", "").split()
            if len(term) > 3
        )

        refined_chunks = []
        for chunk in chunks:
            raw_text = chunk.get("text", "")
            sentences = self.decompose_sentences(raw_text)

            selected_sentences = []
            for stmt in sentences:
                stmt_words = set(stmt.lower().split())
                if not query_terms:
                    overlap = 1.0
                else:
                    overlap = len(query_terms.intersection(stmt_words)) / len(query_terms)

                # Keep sentence if keyword overlap exceeds threshold or chunk overall score is high
                if overlap >= self.min_sentence_score or chunk.get("similarity_score", 0.0) >= 0.70:
                    selected_sentences.append(stmt)

            # If noise stripping eliminated everything, retain the top 2 sentences as fallback
            if not selected_sentences:
                selected_sentences = sentences[:2]

            refined_text = " ".join(selected_sentences)
            
            chunk_copy = dict(chunk)
            chunk_copy["text"] = refined_text
            chunk_copy["original_length_chars"] = len(raw_text)
            chunk_copy["refined_length_chars"] = len(refined_text)
            chunk_copy["noise_reduced"] = len(raw_text) > len(refined_text)
            refined_chunks.append(chunk_copy)

        return refined_chunks


if __name__ == "__main__":
    refiner = KnowledgeRefiner()
    query = "What is the LOINC code for body weight?"
    sample_chunk = {
        "source": "doc_patient_001.txt",
        "text": (
            "Patient PAT-0001 Export. Active: true. "
            "Heart rate LOINC 8867-4 is 72 beats/min. "
            "Body weight LOINC 29463-7 is 68 kg. "
            "Consent status permit secondary use. Address: 123 Main St."
        ),
        "similarity_score": 0.65,
    }
    refined = refiner.refine_chunks(query, [sample_chunk])
    print("Original Text:", sample_chunk["text"])
    print("Refined Text: ", refined[0]["text"])
    print("Noise Reduced:", refined[0]["noise_reduced"])
