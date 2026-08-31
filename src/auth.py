import hashlib
import secrets
import sqlite3
from typing import Optional, Tuple
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "healrag_logs.db"

def hash_api_key(api_key: str) -> str:
    """Computes SHA-256 cryptographic hash of a raw API key string."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

def generate_api_key(prefix: str = "sk_live_") -> str:
    """Generates a cryptographically secure raw API key string."""
    return prefix + secrets.token_urlsafe(32)

def register_api_key(client_id: str, custom_key: Optional[str] = None) -> Tuple[str, str]:
    """
    Registers a new API key for a client.
    Stores ONLY the SHA-256 key_hash in SQLite, returning (client_id, raw_api_key).
    """
    raw_key = custom_key or generate_api_key()
    key_hash = hash_api_key(raw_key)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            INSERT INTO api_keys (client_id, key_hash)
            VALUES (?, ?)
            ON CONFLICT(client_id) DO UPDATE SET key_hash=excluded.key_hash, is_active=1
        """, (client_id, key_hash))
        conn.commit()
    finally:
        conn.close()

    return client_id, raw_key

def verify_api_key(raw_key: str) -> Optional[str]:
    """
    Verifies an incoming raw API key string against stored SHA-256 hashes in SQLite.
    Returns client_id if valid and active, otherwise None.
    """
    if not raw_key or not isinstance(raw_key, str):
        return None

    incoming_hash = hash_api_key(raw_key)

    if not DB_PATH.exists():
        return None

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("""
            SELECT client_id, is_active FROM api_keys
            WHERE key_hash = ? AND is_active = 1
        """, (incoming_hash,)).fetchone()

        if row:
            return row["client_id"]
        return None
    finally:
        conn.close()

def seed_default_dev_key() -> str:
    """Seeds a default dev/demo API key on startup if api_keys table is empty."""
    default_dev_key = "sk_live_healrag_demo_2026"
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        count = conn.execute("SELECT COUNT(*) as c FROM api_keys").fetchone()[0]
        if count == 0:
            register_api_key(client_id="demo_developer", custom_key=default_dev_key)
            print(f"[HealRAG Auth] Seeded default dev key into SQLite ('demo_developer'). Hash: {hash_api_key(default_dev_key)[:12]}...")
    finally:
        conn.close()
    return default_dev_key
