import os
import sqlite3
from pathlib import Path
from typing import Tuple, Any

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "healrag_logs.db"

def get_db_connection() -> Tuple[Any, bool]:
    """
    Returns a database connection tuple: (connection_object, is_postgres_bool).
    Prioritizes PostgreSQL via POSTGRES_URI or DATABASE_URL environment variables.
    Falls back seamlessly to local SQLite if Postgres is unavailable or unconfigured.
    """
    pg_uri = os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if pg_uri and PSYCOPG2_AVAILABLE:
        try:
            conn = psycopg2.connect(pg_uri)
            return conn, True
        except Exception as e:
            print(f"[HealRAG DB Warning] Failed to connect to PostgreSQL ({e}). Falling back to SQLite.")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn, False

def init_db():
    """Call this once at startup to create DB tables (PostgreSQL or SQLite)."""
    conn, is_pg = get_db_connection()
    try:
        cur = conn.cursor()
        if is_pg:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_log (
                    id SERIAL PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    trust_grade VARCHAR(100) NOT NULL,
                    evaluator_score DOUBLE PRECISION,
                    route_taken VARCHAR(100) NOT NULL,
                    winning_chunk_index INTEGER,
                    provenance_source TEXT,
                    latency_ms DOUBLE PRECISION NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    estimated_cost_usd DOUBLE PRECISION,
                    answer_text TEXT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id SERIAL PRIMARY KEY,
                    client_id VARCHAR(255) UNIQUE NOT NULL,
                    key_hash VARCHAR(255) UNIQUE NOT NULL,
                    scope VARCHAR(50) DEFAULT 'read-only',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active INTEGER DEFAULT 1
                );
            """)
        else:
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT UNIQUE NOT NULL,
                    key_hash TEXT UNIQUE NOT NULL,
                    scope TEXT DEFAULT 'read-only',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME,
                    is_active INTEGER DEFAULT 1
                )
            """)
            existing_columns = [col[1] for col in cur.execute("PRAGMA table_info(api_keys)").fetchall()]
            if "expires_at" not in existing_columns:
                cur.execute("ALTER TABLE api_keys ADD COLUMN expires_at DATETIME")
            if "scope" not in existing_columns:
                cur.execute("ALTER TABLE api_keys ADD COLUMN scope TEXT DEFAULT 'read-only'")
        conn.commit()
    finally:
        conn.close()

def log_query(record: dict):
    """Insert one query record into query_log table (PostgreSQL or SQLite)."""
    conn, is_pg = get_db_connection()
    try:
        cur = conn.cursor()
        ph = "%s" if is_pg else "?"
        query = f"""
            INSERT INTO query_log (
                query_text, trust_grade, evaluator_score, route_taken,
                winning_chunk_index, provenance_source, latency_ms,
                prompt_tokens, completion_tokens, estimated_cost_usd, answer_text
            ) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
        """
        params = (
            record["query_text"], record["trust_grade"], record.get("evaluator_score"),
            record["route_taken"], record.get("winning_chunk_index"),
            record.get("provenance_source"), record["latency_ms"],
            record.get("prompt_tokens"), record.get("completion_tokens"),
            record.get("estimated_cost_usd"), record.get("answer_text"),
        )
        cur.execute(query, params)
        conn.commit()
    finally:
        conn.close()