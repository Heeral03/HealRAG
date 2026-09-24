import hashlib
import secrets
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from db import get_db_connection

def hash_api_key(api_key: str) -> str:
    """Computes SHA-256 cryptographic hash of a raw API key string."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

def generate_api_key(prefix: str = "sk_live_") -> str:
    """Generates a cryptographically secure raw API key string."""
    return prefix + secrets.token_urlsafe(32)

def register_api_key(
    client_id: str,
    custom_key: Optional[str] = None,
    expires_in_days: Optional[int] = None,
    scope: str = "read-only"
) -> Tuple[str, str]:
    """
    Registers a new API key for a client with scope and optional expiration.
    Stores ONLY the SHA-256 key_hash in PostgreSQL / SQLite, returning (client_id, raw_api_key).
    """
    raw_key = custom_key or generate_api_key()
    key_hash = hash_api_key(raw_key)

    expires_at = None
    if expires_in_days is not None:
        expires_at = (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=expires_in_days)).strftime("%Y-%m-%d %H:%M:%S")

    conn, is_pg = get_db_connection()
    try:
        cur = conn.cursor()
        if is_pg:
            cur.execute("""
                INSERT INTO api_keys (client_id, key_hash, scope, expires_at, is_active)
                VALUES (%s, %s, %s, %s, 1)
                ON CONFLICT(client_id) DO UPDATE SET
                    key_hash=EXCLUDED.key_hash,
                    scope=EXCLUDED.scope,
                    expires_at=EXCLUDED.expires_at,
                    is_active=1
            """, (client_id, key_hash, scope, expires_at))
        else:
            cur.execute("""
                INSERT INTO api_keys (client_id, key_hash, scope, expires_at, is_active)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(client_id) DO UPDATE SET
                    key_hash=excluded.key_hash,
                    scope=excluded.scope,
                    expires_at=excluded.expires_at,
                    is_active=1
            """, (client_id, key_hash, scope, expires_at))
        conn.commit()
    finally:
        conn.close()

    return client_id, raw_key

def verify_api_key(raw_key: str, required_scope: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Verifies an incoming raw API key string against stored SHA-256 hashes in PostgreSQL / SQLite.
    Checks active status, expiration timestamp, and permission scope.
    Returns dict {"client_id": str, "scope": str} if valid and active, otherwise None.
    """
    if not raw_key or not isinstance(raw_key, str):
        return None

    incoming_hash = hash_api_key(raw_key)

    conn, is_pg = get_db_connection()
    try:
        cur = conn.cursor()
        ph = "%s" if is_pg else "?"
        query = f"""
            SELECT client_id, scope, expires_at, is_active FROM api_keys
            WHERE key_hash = {ph} AND is_active = 1
        """
        cur.execute(query, (incoming_hash,))
        row = cur.fetchone()

        if not row:
            return None

        # Format row mapping depending on DB adapter
        if is_pg:
            client_id, client_scope, expires_at_val, is_active = row[0], row[1], row[2], row[3]
            expires_at_str = expires_at_val.strftime("%Y-%m-%d %H:%M:%S") if isinstance(expires_at_val, datetime) else expires_at_val
        else:
            client_id, client_scope, expires_at_str = row["client_id"], row["scope"], row["expires_at"]

        # Expiry Check
        if expires_at_str:
            try:
                exp_dt = datetime.strptime(str(expires_at_str), "%Y-%m-%d %H:%M:%S")
                if datetime.now(timezone.utc).replace(tzinfo=None) > exp_dt:
                    print(f"[HealRAG Auth] Key for client '{client_id}' expired at {expires_at_str}.")
                    return None
            except ValueError:
                pass

        # Scope Check ('admin' satisfies all scope requirements)
        client_scope = client_scope or "read-only"
        if required_scope == "admin" and client_scope != "admin":
            print(f"[HealRAG Auth] Permission denied for client '{client_id}': requires scope 'admin', got '{client_scope}'.")
            return None

        return {"client_id": client_id, "scope": client_scope}
    finally:
        conn.close()

def revoke_api_key(client_id: str) -> bool:
    """
    Revokes an existing API key for a client_id by setting is_active = 0.
    Returns True if key was found and revoked, False otherwise.
    """
    conn, is_pg = get_db_connection()
    try:
        cur = conn.cursor()
        ph = "%s" if is_pg else "?"
        cur.execute(f"UPDATE api_keys SET is_active = 0 WHERE client_id = {ph}", (client_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()

def seed_default_dev_key() -> str:
    """
    Seeds a random dev/boot API key on startup if no active keys exist.
    Prints the raw generated key directly to server logs. Never uses hardcoded demo keys.
    """
    conn, is_pg = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM api_keys WHERE is_active = 1")
        row = cur.fetchone()
        active_count = row[0] if is_pg else row[0]

        if active_count == 0:
            conn.close() # Close before calling register_api_key which opens its own connection
            random_boot_key = generate_api_key(prefix="sk_live_")
            register_api_key(client_id="demo_developer", custom_key=random_boot_key, scope="admin")
            print(f"\n=========================================================")
            print(f"[HealRAG Auth] BOOT API KEY GENERATED: {random_boot_key}")
            print(f"[HealRAG Auth] Client ID: demo_developer | Scope: admin")
            print(f"=========================================================\n")
            return random_boot_key
        else:
            cur.execute("SELECT client_id FROM api_keys WHERE is_active = 1")
            row = cur.fetchone()
            cid = row[0] if is_pg else row["client_id"]
            db_type = "PostgreSQL" if is_pg else "SQLite"
            print(f"[HealRAG Auth] {db_type} contains existing active API key for client '{cid}'.")
            return "[EXISTING_KEY_ACTIVE]"
    finally:
        try:
            conn.close()
        except Exception:
            pass
