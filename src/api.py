import sys
import time
import os
import gc
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Request, Security, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import torch

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from db import init_db, get_db_connection
from embedder import build_index
from crag_pipeline import CRAGPipeline
from retriever import Retriever
from generator import Generator
from seeder import seed_corpus
from chunker import chunk_directory
from rate_limiter import DualLayerRateLimiter
from starlette.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from auth import seed_default_dev_key, verify_api_key, register_api_key, revoke_api_key

# Optimize PyTorch CPU memory & thread footprint for low-memory containers (Render 512MB limit)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

app = FastAPI(
    title="HealRAG API",
    description="Production Digital Health Corrective RAG (CRAG) Service for European Regulatory & Interoperability Standards",
    version="1.0.0",
)

# CORS Middleware Configuration
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Global singleton pipeline instance (Lazy-loaded on first /query)
pipeline_instance: Optional[CRAGPipeline] = None
retriever_instance: Optional[Retriever] = None
generator_instance: Optional[Generator] = None

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def authenticate_client(
    api_key: Optional[str] = Security(api_key_header),
    request: Optional[Request] = None,
    required_scope: Optional[str] = "read-only"
) -> str:
    """
    Authenticates incoming request using X-API-Key header.
    Checks SHA-256 key hash, active status, expiration timestamp, and required scope.
    Falls back to client IP for public query access if no header is provided.
    """
    if api_key:
        info = verify_api_key(api_key, required_scope=required_scope)
        if not info:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid, expired, inactive, or unauthorized API key for scope '{required_scope}'."
            )
        return info["client_id"]

    if required_scope == "admin":
        raise HTTPException(status_code=401, detail="Admin authorization header (X-API-Key) required.")

    if request and hasattr(request, "client") and request.client:
        return request.client.host or "127.0.0.1"
    return "127.0.0.1"

@app.on_event("startup")
def startup_event():
    print("[HealRAG API] Container started. Initializing SQLite logging database...")
    init_db()
    seed_default_dev_key()
    print("[HealRAG API] Checking Qdrant vector database...")
    if not config.QDRANT_DIR.exists() or not config.METADATA_PATH.exists():
        print("[HealRAG API] Qdrant index missing. Seeding corpus and building Qdrant index...")
        from seeder import seed_corpus
        from chunker import chunk_directory
        from embedder import build_index
        seed_corpus(config.CORPUS_DIR)
        chunks = chunk_directory(config.CORPUS_DIR, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        build_index(chunks)
        print("[HealRAG API] Qdrant index build complete.")
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

class RevokeKeyRequest(BaseModel):
    client_id: str = Field(..., description="Client ID whose key should be revoked")

class GenerateKeyRequest(BaseModel):
    client_id: str = Field(..., description="Unique client identifier")
    scope: str = Field(default="read-only", description="Key scope: 'read-only' or 'admin'")
    expires_in_days: Optional[int] = Field(default=None, description="Lifespan in days")

class KeyOperationResponse(BaseModel):
    status: str
    client_id: str
    message: str
    raw_api_key: Optional[str] = None
    scope: Optional[str] = None

# Instantiate global 2-Layer Rate Limiter
rate_limiter = DualLayerRateLimiter(max_req_per_min=300, bucket_capacity=500000, refill_rate_per_min=50000)

STATIC_DIR = config.BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=FileResponse, tags=["Web Interface"])
def root():
    index_path = config.BASE_DIR / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "service": "HealRAG API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health Check"])
def health_check():
    db_ready = config.QDRANT_DIR.exists()
    return {
        "status": "healthy" if db_ready else "initializing",
        "qdrant_index_exists": db_ready,
        "llm_model": config.GROQ_MODEL
    }

@app.post("/query", response_model=QueryResponse, tags=["CRAG Pipeline"])
async def execute_query(
    req: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Security(api_key_header)
):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # Authenticate client with read-only scope requirement
    client_id = authenticate_client(api_key, request, required_scope="read-only")

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

            res = await run_in_threadpool(pipeline_instance.run, req.query, req.top_k)
            latency = time.time() - t0

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

@app.post("/auth/generate", response_model=KeyOperationResponse, tags=["Authentication & Security"])
def generate_key_endpoint(
    req: GenerateKeyRequest,
    request: Request,
    api_key: Optional[str] = Security(api_key_header)
):
    """Admin-only endpoint to generate new API keys with scope and expiration."""
    auth_client = authenticate_client(api_key, request, required_scope="admin")
    if req.scope not in ["read-only", "admin"]:
        raise HTTPException(status_code=400, detail="Scope must be either 'read-only' or 'admin'.")

    cid, raw_key = register_api_key(
        client_id=req.client_id,
        expires_in_days=req.expires_in_days,
        scope=req.scope
    )
    return KeyOperationResponse(
        status="success",
        client_id=cid,
        message=f"Issued new '{req.scope}' key for '{cid}'. Save raw_api_key now; it will not be displayed again.",
        raw_api_key=raw_key,
        scope=req.scope
    )

@app.post("/auth/revoke", response_model=KeyOperationResponse, tags=["Authentication & Security"])
def revoke_key_endpoint(
    req: RevokeKeyRequest,
    request: Request,
    api_key: Optional[str] = Security(api_key_header)
):
    """Admin-only endpoint to revoke an API key by client_id."""
    auth_client = authenticate_client(api_key, request, required_scope="admin")
    revoked = revoke_api_key(req.client_id)
    if not revoked:
        raise HTTPException(status_code=404, detail=f"No active API key found for client_id '{req.client_id}'.")
    return KeyOperationResponse(
        status="success",
        client_id=req.client_id,
        message=f"API key for client_id '{req.client_id}' successfully revoked."
    )

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
    conn, is_pg = get_db_connection()
    try:
        cur = conn.cursor()
        if is_pg:
            cur.execute("SELECT COUNT(*) FROM query_log")
            total = cur.fetchone()[0] or 0
            cur.execute("SELECT AVG(latency_ms) FROM query_log")
            avg_latency = cur.fetchone()[0] or 0.0
            cur.execute("""
                SELECT route_taken, COUNT(*) as count, AVG(latency_ms) as avg_latency
                FROM query_log GROUP BY route_taken
            """)
            route_breakdown = [{"route_taken": r[0], "count": r[1], "avg_latency": r[2]} for r in cur.fetchall()]
            cur.execute("SELECT SUM(estimated_cost_usd) FROM query_log")
            total_cost = cur.fetchone()[0] or 0.0
        else:
            total = conn.execute("SELECT COUNT(*) as c FROM query_log").fetchone()["c"]
            avg_latency = conn.execute("SELECT AVG(latency_ms) as a FROM query_log").fetchone()["a"]
            route_breakdown = [dict(r) for r in conn.execute("""
                SELECT route_taken, COUNT(*) as count, AVG(latency_ms) as avg_latency
                FROM query_log GROUP BY route_taken
            """).fetchall()]
            total_cost = conn.execute("SELECT SUM(estimated_cost_usd) as s FROM query_log").fetchone()["s"]
    finally:
        conn.close()

    return {
        "total_queries": total,
        "avg_latency_ms": avg_latency,
        "total_cost_usd": total_cost,
        "route_breakdown": route_breakdown,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
