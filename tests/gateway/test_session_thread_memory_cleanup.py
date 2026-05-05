"""Integration tests for thread-memory cleanup on session reset.

When a session auto-resets (idle timeout, daily reset, or suspended),
thread-memory rows for the old session_key must be deleted so stale
conversation context doesn't bleed into the new session.  User-memory
rows must persist across resets.
"""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_idle_config():
    """Config with a 5-minute idle reset policy so tests don't need 24h waits."""
    return GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=5),
    )


def _make_store(tmp_path, *, db=None, config=None):
    """Build a SessionStore with loading disabled and an optional real DB."""
    cfg = config or _short_idle_config()
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=cfg)
    store._loaded = True
    if db is not None:
        store._db = db
    return store


def _source(platform=Platform.TELEGRAM, chat_id="12345",
            chat_type="dm", user_id="u1", user_name="alice"):
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name=user_name,
    )


def _make_idle_entry(store, session_key, idle_minutes):
    """Rewind updated_at on an existing entry to simulate idle expiry."""
    from gateway.session import _now
    entry = store._entries[session_key]
    entry.updated_at = _now() - timedelta(minutes=idle_minutes + 10)
    store._save()


# ---------------------------------------------------------------------------
# Tests — real SQLite DB
# ---------------------------------------------------------------------------

class TestThreadMemoryCleanupOnReset:
    """Thread memory is deleted when a session auto-resets via idle timeout."""

    @pytest.fixture
    def db(self, tmp_path):
        from hermes_state import SessionDB
        d = SessionDB(db_path=tmp_path / "state.db")
        yield d
        d.close()

    @pytest.fixture
    def store(self, tmp_path, db):
        return _make_store(tmp_path, db=db)

    def test_thread_memory_deleted_on_idle_reset(self, store, db):
        """Thread memory rows are removed when an idle timeout triggers a reset."""
        source = _source(user_id="user-x")
        # First call creates session
        entry = store.get_or_create_session(source)
        session_key = entry.session_key

        # Write thread memory for this session
        db.set_thread_memory(session_key, "topic", "python debugging")
        db.set_thread_memory(session_key, "mood", "curious")
        assert len(db.get_thread_memory(session_key)) == 2

        # Also write user memory — must survive the reset
        db.set_user_memory("user-x", "name_preference", "Alice")
        assert len(db.get_user_memory("user-x")) == 1

        # Force the session to look idle so _should_reset returns "idle"
        _make_idle_entry(store, session_key, idle_minutes=10)

        # Now call get_or_create_session again — should trigger reset
        new_entry = store.get_or_create_session(source)

        # Thread memory for old session_key should be gone
        assert db.get_thread_memory(session_key) == []

        # User memory must survive
        assert db.get_user_memory("user-x") == [("name_preference", "Alice")]

    def test_thread_memory_deleted_on_suspended_reset(self, store, db):
        """Thread memory rows are removed when a suspended session resets."""
        source = _source(user_id="user-y")
        entry = store.get_or_create_session(source)
        session_key = entry.session_key

        # Write thread memory
        db.set_thread_memory(session_key, "context", "api integration")
        assert len(db.get_thread_memory(session_key)) == 1

        # Write user memory
        db.set_user_memory("user-y", "style", "verbose")
        assert len(db.get_user_memory("user-y")) == 1

        # Mark session as suspended
        store.suspend_session(session_key)

        # Call get_or_create_session — should trigger suspended reset
        new_entry = store.get_or_create_session(source)

        # Thread memory for old session_key should be gone
        assert db.get_thread_memory(session_key) == []

        # User memory must survive
        assert db.get_user_memory("user-y") == [("style", "verbose")]

    def test_no_thread_memory_cleanup_without_db(self, tmp_path):
        """When SessionDB is unavailable, session reset still works (no crash)."""
        store = _make_store(tmp_path, db=None)
        source = _source()
        entry = store.get_or_create_session(source)
        session_key = entry.session_key

        _make_idle_entry(store, session_key, idle_minutes=10)

        # Should not raise — thread memory cleanup is silently skipped
        new_entry = store.get_or_create_session(source)
        assert new_entry.session_id != entry.session_id  # session actually reset


class TestThreadMemoryCleanupFailure:
    """Cleanup failure logs debug but does not block session creation."""

    def test_cleanup_failure_does_not_block_session(self, tmp_path):
        """If delete_thread_memory raises, session creation still succeeds."""
        store = _make_store(tmp_path)
        mock_db = MagicMock()
        mock_db.end_session = MagicMock()
        mock_db.create_session = MagicMock()
        mock_db.delete_thread_memory.side_effect = Exception("DB I/O error")
        store._db = mock_db

        source = _source()
        entry1 = store.get_or_create_session(source)
        session_key = entry1.session_key

        # Make idle and force reset
        _make_idle_entry(store, session_key, idle_minutes=10)
        entry2 = store.get_or_create_session(source)

        # New session was created despite cleanup failure
        assert entry2.session_id != entry1.session_id
        mock_db.delete_thread_memory.assert_called_once_with(session_key)