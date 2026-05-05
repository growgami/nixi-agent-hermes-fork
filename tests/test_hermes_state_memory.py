"""Tests for hermes_state.py — thread_memory and user_memory CRUD."""

import time
import pytest
from pathlib import Path

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_memory.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


# =========================================================================
# thread_memory
# =========================================================================

class TestThreadMemory:
    def test_set_and_get_roundtrip(self, db):
        """set_thread_memory + get_thread_memory roundtrip."""
        db.set_thread_memory("session-1", "mood", "curious")
        result = db.get_thread_memory("session-1")
        assert result == [("mood", "curious")]

    def test_multiple_keys_for_same_session(self, db):
        """Multiple keys under the same session_key are all returned."""
        db.set_thread_memory("session-1", "mood", "curious")
        db.set_thread_memory("session-1", "topic", "python")
        db.set_thread_memory("session-1", "style", "concise")
        result = db.get_thread_memory("session-1")
        assert len(result) == 3
        assert ("mood", "curious") in result
        assert ("topic", "python") in result
        assert ("style", "concise") in result

    def test_get_nonexistent_session(self, db):
        """get_thread_memory returns [] for nonexistent session_key."""
        result = db.get_thread_memory("nonexistent")
        assert result == []

    def test_upsert_overwrites_existing_key(self, db):
        """set_thread_memory overwrites existing key (INSERT OR REPLACE)."""
        db.set_thread_memory("session-1", "mood", "happy")
        db.set_thread_memory("session-1", "mood", "reflective")
        result = db.get_thread_memory("session-1")
        assert result == [("mood", "reflective")]

    def test_delete_single_key(self, db):
        """delete_thread_memory_key removes only the specified key."""
        db.set_thread_memory("session-1", "mood", "curious")
        db.set_thread_memory("session-1", "topic", "python")
        db.delete_thread_memory_key("session-1", "mood")
        result = db.get_thread_memory("session-1")
        assert result == [("topic", "python")]

    def test_delete_all_for_session(self, db):
        """delete_thread_memory removes all keys for a session_key."""
        db.set_thread_memory("session-1", "mood", "curious")
        db.set_thread_memory("session-1", "topic", "python")
        db.set_thread_memory("session-2", "mood", "happy")
        db.delete_thread_memory("session-1")
        assert db.get_thread_memory("session-1") == []
        assert db.get_thread_memory("session-2") == [("mood", "happy")]

    def test_updated_at_is_set(self, db):
        """updated_at is populated on write."""
        db.set_thread_memory("session-1", "mood", "curious")
        with db._lock:
            row = db._conn.execute(
                "SELECT updated_at FROM thread_memory WHERE session_key = ? AND key = ?",
                ("session-1", "mood"),
            ).fetchone()
        assert row is not None
        assert row["updated_at"] is not None


# =========================================================================
# user_memory
# =========================================================================

class TestUserMemory:
    def test_set_and_get_roundtrip(self, db):
        """set_user_memory + get_user_memory roundtrip."""
        db.set_user_memory("user-1", "preference", "dark-mode")
        result = db.get_user_memory("user-1")
        assert result == [("preference", "dark-mode")]

    def test_get_excludes_expired(self, db):
        """get_user_memory excludes expired entries but keeps non-expiring ones."""
        db.set_user_memory("user-1", "active", "yes")
        # Use negative TTL to guarantee immediate expiry
        db.set_user_memory("user-1", "expired", "no", ttl_hours=-1)
        # No sleep needed — expires_at is in the past
        result = db.get_user_memory("user-1")
        assert result == [("active", "yes")]

    def test_lazy_deletes_expired(self, db):
        """get_user_memory physically deletes expired rows (lazy cleanup)."""
        db.set_user_memory("user-1", "expired", "no", ttl_hours=-1)
        # Call triggers lazy delete — no sleep needed, already expired
        db.get_user_memory("user-1")
        # Verify the row is physically gone
        with db._lock:
            row = db._conn.execute(
                "SELECT COUNT(*) FROM user_memory WHERE user_id = ? AND key = ?",
                ("user-1", "expired"),
            ).fetchone()
        assert row[0] == 0

    def test_set_with_ttl(self, db):
        """set_user_memory with ttl_hours sets expires_at."""
        db.set_user_memory("user-1", "temp", "data", ttl_hours=1.0)
        with db._lock:
            row = db._conn.execute(
                "SELECT expires_at FROM user_memory WHERE user_id = ? AND key = ?",
                ("user-1", "temp"),
            ).fetchone()
        assert row is not None
        assert row["expires_at"] is not None

    def test_set_without_ttl(self, db):
        """set_user_memory without ttl_hours leaves expires_at as NULL."""
        db.set_user_memory("user-1", "permanent", "data")
        with db._lock:
            row = db._conn.execute(
                "SELECT expires_at FROM user_memory WHERE user_id = ? AND key = ?",
                ("user-1", "permanent"),
            ).fetchone()
        assert row is not None
        assert row["expires_at"] is None

    def test_upsert_overwrites_existing_key(self, db):
        """set_user_memory overwrites existing key (INSERT OR REPLACE)."""
        db.set_user_memory("user-1", "preference", "dark-mode")
        db.set_user_memory("user-1", "preference", "light-mode")
        result = db.get_user_memory("user-1")
        assert result == [("preference", "light-mode")]

    def test_delete_specific_key(self, db):
        """delete_user_memory removes only the specified key."""
        db.set_user_memory("user-1", "preference", "dark-mode")
        db.set_user_memory("user-1", "language", "en")
        db.delete_user_memory("user-1", "preference")
        result = db.get_user_memory("user-1")
        assert result == [("language", "en")]

    def test_delete_all_user_memory(self, db):
        """delete_all_user_memory removes all keys for a user."""
        db.set_user_memory("user-1", "preference", "dark-mode")
        db.set_user_memory("user-1", "language", "en")
        db.set_user_memory("user-2", "preference", "light-mode")
        db.delete_all_user_memory("user-1")
        assert db.get_user_memory("user-1") == []
        assert db.get_user_memory("user-2") == [("preference", "light-mode")]

    def test_get_nonexistent_user(self, db):
        """get_user_memory returns [] for nonexistent user_id."""
        result = db.get_user_memory("nonexistent")
        assert result == []

    def test_updated_at_is_set(self, db):
        """updated_at is populated on write."""
        db.set_user_memory("user-1", "pref", "dark")
        with db._lock:
            row = db._conn.execute(
                "SELECT updated_at FROM user_memory WHERE user_id = ? AND key = ?",
                ("user-1", "pref"),
            ).fetchone()
        assert row is not None
        assert row["updated_at"] is not None


# =========================================================================
# Schema migration
# =========================================================================

class TestMemorySchemaMigration:
    def test_fresh_db_has_memory_tables(self, db):
        """Fresh database should have thread_memory and user_memory tables."""
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "thread_memory" in tables
        assert "user_memory" in tables

    def test_fresh_db_has_memory_indexes(self, db):
        """Fresh database should have memory table indexes."""
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        assert "idx_thread_memory_session" in indexes
        assert "idx_user_memory_expires" in indexes

    def test_migration_from_v8(self, tmp_path):
        """Migrating from v8 should create memory tables and indexes."""
        import sqlite3

        db_path = tmp_path / "migrate_v8.db"
        conn = sqlite3.connect(str(db_path))
        # Create a v8-compatible schema with all columns up to v8.
        # SCHEMA_SQL runs with IF NOT EXISTS, so it needs all referenced
        # columns to exist. Create a minimal v8 schema that passes validation.
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (8);
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                api_call_count INTEGER DEFAULT 0,
                FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT
            );
            CREATE TABLE IF NOT EXISTS state_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
            CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique ON sessions(title) WHERE title IS NOT NULL;
        """)
        conn.commit()
        conn.close()

        # Opening with SessionDB should run the v9 migration
        session_db = SessionDB(db_path=db_path)
        try:
            cursor = session_db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = {row[0] for row in cursor.fetchall()}
            assert "thread_memory" in tables
            assert "user_memory" in tables

            cursor = session_db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
            )
            indexes = {row[0] for row in cursor.fetchall()}
            assert "idx_thread_memory_session" in indexes
            assert "idx_user_memory_expires" in indexes

            # Verify schema version is now 9
            cursor = session_db._conn.execute("SELECT version FROM schema_version")
            assert cursor.fetchone()[0] == 9
        finally:
            session_db.close()

    def test_schema_version_9(self, db):
        """Current schema version should be 9."""
        cursor = db._conn.execute("SELECT version FROM schema_version")
        assert cursor.fetchone()[0] == 9