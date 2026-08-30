from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load .env directly
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import config

class Generator:
    def __init__(self):
        self.api_key = config.GROQ_API_KEY
        if self.api_key and len(self.api_key.strip()) > 0:
            print("Configuring Groq Client...")
            self.client = Groq(api_key=self.api_key.strip())
            self.model_name = config.GROQ_MODEL
            self.use_mock = False
        else:
            print("WARNING: GROQ_API_KEY not found in environment. Falling back to Mock LLM.")
            self.use_mock = True
            
    def _generate_mock_response(self, prompt: str, retrieved_chunks: list[dict]) -> str:
        """
        Generate a structured mock response by extracting details from context chunks.
        """
        response_lines = [
            "[LOCAL MOCK LLM MODE - NO GROQ_API_KEY FOUND]",
            "Based on the retrieved context, here is the synthesized answer:\n"
        ]
        
        sources = set()
        details = []
        for i, chunk in enumerate(retrieved_chunks):
            sources.add(chunk["source"])
            clean_text = chunk["text"].replace("\n", " ").strip()
            sentences = [s.strip() for s in clean_text.split(".") if s.strip()]
            first_sentences = ". ".join(sentences[:2])
            details.append(f"- From {chunk['source']}: {first_sentences}.")
            
        response_lines.extend(details)
        response_lines.append(f"\nSources: {', '.join(sorted(list(sources)))}")
        return "\n".join(response_lines)

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        """
        Construct a context-stuffed prompt and generate an answer using Groq API (llama-3.3-70b-versatile).
        """
        context_block = ""
        for idx, chunk in enumerate(retrieved_chunks):
            context_block += f"--- Document Chunk {idx+1} [Source: {chunk['source']}] ---\n"
            context_block += f"{chunk['text']}\n\n"
            
        prompt = (
            "You are an expert digital health standards and healthcare regulatory assistant.\n"
            "Your goal is to answer the user's query with extreme precision based ONLY on the provided context chunks.\n\n"
            "### Strict Guidelines:\n"
            "1. Base your answer strictly on the provided context chunks below.\n"
            "2. If the context chunks contain enough information, provide a structured, detailed answer with section headings, bullet points, and exact inline citations (e.g., [Source: doc_001_gdpr_art9_para1.txt]).\n"
            "3. Connect concepts explicitly to the exact legal/technical sections mentioned in the text (e.g., Article numbers, FHIR resource attributes, profile requirements).\n"
            "4. If the context ONLY partially answers the query, answer what is present and explicitly state what specific aspect is missing from the corpus.\n"
            "5. If the context does not contain the answer or is unrelated, clearly state: 'The provided document corpus does not contain information to answer this question.' Do NOT hallucinate outside facts.\n\n"
            f"=== Context Chunks ===\n{context_block}"
            f"=== User Query ===\n{query}\n\n"
            "=== Structured Answer ==="
        )
        
        if self.use_mock:
            return self._generate_mock_response(prompt, retrieved_chunks)
            
        try:
            print(f"Sending request to Groq API ({self.model_name})...")
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional assistant specialized in digital health regulations, FHIR standards, and healthcare interoperability."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                temperature=0.2,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq API: {e}. Falling back to mock generator.")
            return self._generate_mock_response(prompt, retrieved_chunks)

if __name__ == "__main__":
    print("Testing Generator with Groq...")
    mock_chunks = [
        {
            "text": "FHIR Patient Resource represents administrative information. Key fields: name (HumanName), birthDate (Date), gender (administrative value).",
            "source": "doc_020_fhir_patient.txt",
            "chunk_index": 0
        }
    ]
    gen = Generator()
    test_query = "What fields are in the FHIR Patient resource?"
    response = gen.generate(test_query, mock_chunks)
    print("\n--- Answer Output ---")
    print(response)
