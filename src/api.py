import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from src.db import init_db
import sqlite3



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

import os
import gc
import torch

# Optimize PyTorch CPU memory & thread footprint for low-memory containers (Render 512MB limit)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Global singleton pipeline instance (Lazy-loaded on first /query)
pipeline_instance: Optional[CRAGPipeline] = None
retriever_instance: Optional[Retriever] = None
generator_instance: Optional[Generator] = None

from seeder import seed_corpus
from chunker import chunk_directory

from fastapi.security import APIKeyHeader
from fastapi import Security
from auth import seed_default_dev_key, verify_api_key

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def authenticate_client(api_key: Optional[str] = Security(api_key_header), request: Optional[Request] = None) -> str:
    """
    Authenticates incoming request using X-API-Key header.
    Hashes the raw key via SHA-256 and checks against SQLite api_keys table.
    Falls back to client IP for unauthenticated public access if no header is provided.
    """
    if api_key:
        client_id = verify_api_key(api_key)
        if not client_id:
            raise HTTPException(status_code=401, detail="Invalid or inactive API key.")
        return client_id

    # Fallback to dev key or client IP if header is missing
    if request:
        return request.client.host or "127.0.0.1"
    return "anonymous_client"

@app.on_event("startup")
def startup_event():
    print("[HealRAG API] Container started. Initializing SQLite logging database...")
    init_db()
    seed_default_dev_key()
    print("[HealRAG API] Checking FAISS vector database...")
    if not config.FAISS_INDEX_PATH.exists() or not config.METADATA_PATH.exists():
        print("[HealRAG API] FAISS index missing. Seeding corpus and building FAISS index...")
        seed_corpus()
        chunks = chunk_directory(config.CORPUS_DIR, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        build_index(chunks)
        gc.collect()
    print("[HealRAG API] Service startup complete. CRAG Pipeline ready for lazy initialization.")

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
    observability: Optional[Dict] = None

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

from rate_limiter import DualLayerRateLimiter
from starlette.concurrency import run_in_threadpool

# Instantiate global 2-Layer Rate Limiter
# Layer 1: Max 300 requests/min (calibrated for high-concurrency multi-worker stress testing).
# Layer 2: 500,000 token capacity, 50,000 refill/min
rate_limiter = DualLayerRateLimiter(max_req_per_min=300, bucket_capacity=500000, refill_rate_per_min=50000)

from fastapi import FastAPI, Security, HTTPException, Depends, Request, BackgroundTasks
from db import log_query

@app.post("/query", response_model=QueryResponse, tags=["CRAG Pipeline"])
async def execute_query(req: QueryRequest, request: Request, background_tasks: BackgroundTasks, api_key: Optional[str] = Security(api_key_header)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # Authenticate client via SHA-256 hashed API key lookup
    client_id = authenticate_client(api_key, request)

    # Pre-check Dual Layer Rate Limiting
    allowed, msg, details = rate_limiter.check_pre_request(client_id, min_token_estimate=500)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Rate Limit Exceeded: {msg}")

    t0 = time.time()
    try:
        if req.vanilla_mode:
            global retriever_instance, generator_instance
            if retriever_instance is None:
                retriever_instance = Retriever()
            if generator_instance is None:
                generator_instance = Generator()

            chunks = await run_in_threadpool(retriever_instance.retrieve, req.query, req.top_k)
            ans = await run_in_threadpool(generator_instance.generate, req.query, chunks)
            latency = time.time() - t0

            # Deduct tokens for Vanilla RAG (~1200 tokens)
            rate_limiter.deduct_post_response(client_id, tokens_used=1200)

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

            # Offload blocking CRAG pipeline execution to threadpool for non-blocking FastAPI async execution
            res = await run_in_threadpool(pipeline_instance.run, req.query, req.top_k)
            latency = time.time() - t0

            # Layer 2 Token Cost Deduction based on route taken
            eval_act = res.get("eval_action", "CORRECT")
            tokens_used = 450 if eval_act == "CORRECT" else (1500 if eval_act == "AMBIGUOUS" else 2200)
            remaining_tokens = rate_limiter.deduct_post_response(client_id, tokens_used=tokens_used)

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

            # Attach rate limit info to observability breakdown
            obs = res.get("observability", {})
            obs["rate_limit_tokens"] = {
                "deducted_tokens": tokens_used,
                "remaining_bucket_tokens": remaining_tokens
            }

            return QueryResponse(
                query=req.query,
                mode="Corrective RAG (CRAG)",
                eval_action=res["eval_action"],
                confidence_score=res["confidence_score"],
                pipeline_log=res["pipeline_log"],
                latency_sec=round(latency, 3),
                final_chunks=formatted_chunks,
                response=res["response"],
                observability=obs
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

@app.get("/rate-limit-status", tags=["Rate Limiting"])
def get_rate_limit_status(request: Request):
    client_id = request.headers.get("X-Session-ID") or request.client.host or "127.0.0.1"
    bucket = rate_limiter.layer2._get_bucket(client_id)
    return {
        "client_id": client_id,
        "layer_1_max_req_per_min": rate_limiter.layer1.max_requests,
        "layer_2_bucket_capacity": rate_limiter.layer2.capacity,
        "layer_2_remaining_tokens": round(bucket["tokens"], 1),
        "layer_2_refill_rate_tokens_per_min": 5000
    }

@app.get("/analytics", tags=["Analytics"])
def get_analytics():
    conn = sqlite3.connect("data/healrag_logs.db")
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) as c FROM query_log").fetchone()["c"]
    avg_latency = conn.execute("SELECT AVG(latency_ms) as a FROM query_log").fetchone()["a"]
    route_breakdown = conn.execute("""
        SELECT route_taken, COUNT(*) as count, AVG(latency_ms) as avg_latency
        FROM query_log GROUP BY route_taken
    """).fetchall()
    total_cost = conn.execute("SELECT SUM(estimated_cost_usd) as s FROM query_log").fetchone()["s"]

    conn.close()
    return {
        "total_queries": total,
        "avg_latency_ms": avg_latency,
        "total_cost_usd": total_cost,
        "route_breakdown": [dict(r) for r in route_breakdown],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
