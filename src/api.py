import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from embedder import build_index
from crag_pipeline import CRAGPipeline
from retriever import Retriever
from generator import Generator

app = FastAPI(
    title="HealRAG API",
    description="Production Digital Health Corrective RAG (CRAG) Service for European Regulatory & Interoperability Standards",
    version="1.0.0",
)

# Global singleton pipeline instance
pipeline_instance: Optional[CRAGPipeline] = None

from embedder import build_index
from seeder import generate_synthetic_corpus
from chunker import chunk_directory

@app.on_event("startup")
def startup_event():
    global pipeline_instance
    print("[HealRAG API] Initializing Vector DB and CRAG Pipeline on startup...")
    if not config.FAISS_INDEX_PATH.exists() or not config.METADATA_PATH.exists():
        print("[HealRAG API] FAISS index missing. Seeding corpus and building FAISS index...")
        generate_synthetic_corpus()
        chunks = chunk_directory(config.CORPUS_DIR, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        build_index(chunks)
    pipeline_instance = CRAGPipeline()
    print("[HealRAG API] CRAG Service initialized successfully.")

# Pydantic Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., example="What is GDPR Article 9?", description="User regulatory or technical health query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of document chunks to retrieve")
    vanilla_mode: bool = Field(default=False, description="If True, bypasses CRAG Evaluator and runs Vanilla RAG baseline")

class ChunkResponse(BaseModel):
    source: str
    text: str
    similarity_score: float
    eval_status: Optional[str] = None
    eval_score: Optional[float] = None

class QueryResponse(BaseModel):
    query: str
    mode: str
    eval_action: Optional[str] = None
    confidence_score: Optional[float] = None
    pipeline_log: List[str]
    latency_sec: float
    final_chunks: List[ChunkResponse]
    response: str

@app.get("/", tags=["Health Check"])
def root():
    return {
        "service": "HealRAG API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health Check"])
def health_check():
    db_ready = config.FAISS_INDEX_PATH.exists()
    return {
        "status": "healthy" if db_ready else "initializing",
        "faiss_index_exists": db_ready,
        "llm_model": config.GROQ_MODEL
    }

@app.post("/query", response_model=QueryResponse, tags=["CRAG Pipeline"])
def execute_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    t0 = time.time()
    try:
        if req.vanilla_mode:
            retriever = Retriever()
            generator = Generator()
            chunks = retriever.retrieve(req.query, top_k=req.top_k)
            ans = generator.generate(req.query, chunks)
            latency = time.time() - t0

            formatted_chunks = [
                ChunkResponse(
                    source=c.get("source", "Unknown"),
                    text=c.get("text", ""),
                    similarity_score=c.get("similarity_score", 0.0)
                )
                for c in chunks
            ]

            return QueryResponse(
                query=req.query,
                mode="Vanilla RAG Baseline",
                eval_action="NONE",
                confidence_score=chunks[0].get("similarity_score", 0.0) if chunks else 0.0,
                pipeline_log=["Executed direct vector retrieval without CRAG Evaluator."],
                latency_sec=round(latency, 3),
                final_chunks=formatted_chunks,
                response=ans
            )
        else:
            global pipeline_instance
            if pipeline_instance is None:
                pipeline_instance = CRAGPipeline()

            res = pipeline_instance.run(req.query, top_k=req.top_k)
            latency = time.time() - t0

            formatted_chunks = [
                ChunkResponse(
                    source=c.get("source", "Unknown"),
                    text=c.get("text", ""),
                    similarity_score=c.get("similarity_score", 0.0),
                    eval_status=c.get("eval_status"),
                    eval_score=c.get("eval_score")
                )
                for c in res["final_chunks"]
            ]

            return QueryResponse(
                query=req.query,
                mode="Corrective RAG (CRAG)",
                eval_action=res["eval_action"],
                confidence_score=res["confidence_score"],
                pipeline_log=res["pipeline_log"],
                latency_sec=round(latency, 3),
                final_chunks=formatted_chunks,
                response=res["response"]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
