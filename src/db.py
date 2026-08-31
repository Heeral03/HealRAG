import sqlite3
from pathlib import Path
from datetime import datetime

# Point to standard data directory relative to repository root
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "healrag_logs.db"

def init_db():
    """Call this once at startup to create the table if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            trust_grade TEXT NOT NULL,
            evaluator_score REAL,
            route_taken TEXT NOT NULL,
            winning_chunk_index INTEGER,
            provenance_source TEXT,
            latency_ms REAL NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            estimated_cost_usd REAL,
            answer_text TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_query(record: dict):
    """Insert one query record. Call this after every pipeline run."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO query_log (
            query_text, trust_grade, evaluator_score, route_taken,
            winning_chunk_index, provenance_source, latency_ms,
            prompt_tokens, completion_tokens, estimated_cost_usd, answer_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["query_text"], record["trust_grade"], record["evaluator_score"],
        record["route_taken"], record.get("winning_chunk_index"),
        record.get("provenance_source"), record["latency_ms"],
        record.get("prompt_tokens"), record.get("completion_tokens"),
        record.get("estimated_cost_usd"), record.get("answer_text"),
    ))
    conn.commit()
    conn.close()