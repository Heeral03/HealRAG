import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from db import get_db_connection, init_db, log_query
from auth import register_api_key, verify_api_key, revoke_api_key

class TestPostgreSQLAdapter(unittest.TestCase):
    @patch("db.PSYCOPG2_AVAILABLE", True)
    @patch("db.psycopg2.connect")
    @patch.dict("os.environ", {"POSTGRES_URI": "postgresql://user:pass@localhost:5432/healrag_db"})
    def test_postgres_connection_fallback_and_queries(self, mock_connect):
        """Verify get_db_connection uses psycopg2 when POSTGRES_URI is provided."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        conn, is_pg = get_db_connection()
        self.assertTrue(is_pg)
        self.assertEqual(conn, mock_conn)
        mock_connect.assert_called_once_with("postgresql://user:pass@localhost:5432/healrag_db")

    @patch("db.PSYCOPG2_AVAILABLE", True)
    @patch("db.psycopg2.connect")
    @patch.dict("os.environ", {"POSTGRES_URI": "postgresql://user:pass@localhost:5432/healrag_db"})
    def test_postgres_init_db_and_log_query(self, mock_connect):
        """Verify PostgreSQL DDL execution and %s placeholder formatting in log_query."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn

        init_db()
        self.assertTrue(mock_cur.execute.called)

        # Verify log_query uses %s placeholders for Postgres
        log_query({
            "query_text": "What is GDPR Article 9?",
            "trust_grade": "A",
            "evaluator_score": 0.85,
            "route_taken": "CORRECT",
            "winning_chunk_index": 0,
            "provenance_source": "gdpr.txt",
            "latency_ms": 120.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "estimated_cost_usd": 0.0001,
            "answer_text": "Special categories of data..."
        })
        args, _ = mock_cur.execute.call_args
        self.assertIn("%s", args[0])

if __name__ == "__main__":
    unittest.main()
