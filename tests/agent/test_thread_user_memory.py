"""Tests for agent/thread_user_memory.py — ThreadUserMemoryManager."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from hermes_state import SessionDB
from agent.thread_user_memory import ThreadUserMemoryManager


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_memory.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


@pytest.fixture()
def mgr(db):
    """Create a ThreadUserMemoryManager backed by a real SessionDB."""
    return ThreadUserMemoryManager(db)


# =========================================================================
# prefetch()
# =========================================================================


class TestPrefetch:
    def test_returns_formatted_blocks(self, mgr, db):
        """prefetch() returns <thread-context> and <user-context> blocks."""
        db.set_thread_memory("sess-1", "mood", "curious")
        db.set_user_memory("user-1", "name", "Alice")

        result = mgr.prefetch("sess-1", "user-1")

        assert "<thread-context>" in result
        assert "</thread-context>" in result
        assert "<user-context>" in result
        assert "</user-context>" in result
        assert "[System note: The following is thread-scoped working memory" in result
        assert "[System note: The following is user-level memory" in result
        assert "- mood: curious" in result
        assert "- name: Alice" in result

    def test_returns_empty_string_when_both_empty(self, mgr, db):
        """prefetch() returns empty string when both memories are empty."""
        result = mgr.prefetch("sess-1", "user-1")
        assert result == ""

    def test_omits_thread_context_when_empty(self, mgr, db):
        """prefetch() omits <thread-context> block when thread memory is empty."""
        db.set_user_memory("user-1", "name", "Alice")

        result = mgr.prefetch("sess-1", "user-1")

        assert "<thread-context>" not in result
        assert "<user-context>" in result
        assert "- name: Alice" in result

    def test_omits_user_context_when_empty(self, mgr, db):
        """prefetch() omits <user-context> block when user memory is empty."""
        db.set_thread_memory("sess-1", "mood", "curious")

        result = mgr.prefetch("sess-1", "user-1")

        assert "<thread-context>" in result
        assert "<user-context>" not in result
        assert "- mood: curious" in result

    def test_thread_context_format(self, mgr, db):
        """prefetch() formats thread-context block correctly."""
        db.set_thread_memory("sess-1", "mood", "curious")
        db.set_thread_memory("sess-1", "topic", "python")

        result = mgr.prefetch("sess-1", "user-1")

        assert "<thread-context>" in result
        assert "[System note: The following is thread-scoped working memory, NOT new user input. This context is ephemeral and tied to this conversation thread.]" in result
        assert "- mood: curious" in result
        assert "- topic: python" in result
        assert "</thread-context>" in result

    def test_user_context_format(self, mgr, db):
        """prefetch() formats user-context block correctly."""
        db.set_user_memory("user-1", "name", "Alice")
        db.set_user_memory("user-1", "lang", "en")

        result = mgr.prefetch("sess-1", "user-1")

        assert "<user-context>" in result
        assert "[System note: The following is user-level memory that persists across channels and sessions. Treat as background knowledge about this user.]" in result
        assert "- name: Alice" in result
        assert "- lang: en" in result
        assert "</user-context>" in result

    def test_after_thread_set_includes_pair(self, mgr, db):
        """prefetch() includes key-value pair set via thread_set."""
        mgr.thread_set("sess-1", "preference", "concise")
        result = mgr.prefetch("sess-1", "user-1")
        assert "- preference: concise" in result


# =========================================================================
# thread_set / thread_get / thread_delete
# =========================================================================


class TestThreadCRUD:
    def test_thread_set_and_get(self, mgr):
        """thread_set + thread_get roundtrip."""
        result_json = mgr.thread_set("sess-1", "mood", "curious")
        result = json.loads(result_json)
        assert result == {"success": True, "key": "mood"}

        get_result = json.loads(mgr.thread_get("sess-1", "mood"))
        assert get_result == {"key": "mood", "value": "curious"}

    def test_thread_set_overwrites(self, mgr):
        """thread_set overwrites previous value for same key."""
        mgr.thread_set("sess-1", "mood", "happy")
        mgr.thread_set("sess-1", "mood", "reflective")

        result = json.loads(mgr.thread_get("sess-1", "mood"))
        assert result == {"key": "mood", "value": "reflective"}

    def test_thread_get_all_keys(self, mgr):
        """thread_get with key=None returns all key-value pairs."""
        mgr.thread_set("sess-1", "mood", "curious")
        mgr.thread_set("sess-1", "topic", "python")

        result = json.loads(mgr.thread_get("sess-1"))
        assert "keys" in result
        assert result["keys"]["mood"] == "curious"
        assert result["keys"]["topic"] == "python"

    def test_thread_get_missing_key(self, mgr):
        """thread_get for nonexistent key returns null value."""
        result = json.loads(mgr.thread_get("sess-1", "nonexistent"))
        assert result == {"key": "nonexistent", "value": None}

    def test_thread_delete(self, mgr):
        """thread_delete removes the specified key."""
        mgr.thread_set("sess-1", "mood", "curious")
        mgr.thread_set("sess-1", "topic", "python")

        result = json.loads(mgr.thread_delete("sess-1", "mood"))
        assert result == {"success": True, "key": "mood"}

        # mood is gone, topic remains
        remaining = json.loads(mgr.thread_get("sess-1"))
        assert "mood" not in remaining["keys"]
        assert remaining["keys"]["topic"] == "python"


# =========================================================================
# user_set / user_get / user_delete
# =========================================================================


class TestUserCRUD:
    def test_user_set_and_get(self, mgr):
        """user_set + user_get roundtrip."""
        result_json = mgr.user_set("user-1", "preference", "dark-mode")
        result = json.loads(result_json)
        assert result == {"success": True, "key": "preference"}

        get_result = json.loads(mgr.user_get("user-1", "preference"))
        assert get_result == {"key": "preference", "value": "dark-mode"}

    def test_user_set_with_ttl(self, mgr):
        """user_set with ttl_hours stores and retrieves value."""
        result_json = mgr.user_set("user-1", "temp", "data", ttl_hours=1.0)
        result = json.loads(result_json)
        assert result == {"success": True, "key": "temp"}

        get_result = json.loads(mgr.user_get("user-1", "temp"))
        assert get_result == {"key": "temp", "value": "data"}

    def test_user_set_without_ttl(self, mgr):
        """user_set without ttl_hours stores and retrieves value."""
        mgr.user_set("user-1", "permanent", "data")

        result = json.loads(mgr.user_get("user-1", "permanent"))
        assert result == {"key": "permanent", "value": "data"}

    def test_user_delete(self, mgr):
        """user_delete removes the specified key."""
        mgr.user_set("user-1", "preference", "dark-mode")
        result = json.loads(mgr.user_delete("user-1", "preference"))
        assert result == {"success": True, "key": "preference"}

        # key is gone
        get_result = json.loads(mgr.user_get("user-1", "preference"))
        assert get_result == {"key": "preference", "value": None}


# =========================================================================
# delete_thread_memory
# =========================================================================


class TestDeleteThreadMemory:
    def test_delete_thread_memory(self, mgr, db):
        """delete_thread_memory delegates to db.delete_thread_memory()."""
        mgr.thread_set("sess-1", "mood", "curious")
        mgr.thread_set("sess-1", "topic", "python")

        mgr.delete_thread_memory("sess-1")

        assert db.get_thread_memory("sess-1") == []


# =========================================================================
# handle_tool_call()
# =========================================================================


class TestHandleToolCall:
    def test_thread_mem_set(self, mgr):
        """handle_tool_call dispatches thread_mem.set."""
        result = json.loads(
            mgr.handle_tool_call("thread_mem.set", {
                "key": "mood",
                "value": "curious",
                "_gateway_session_key": "sess-1",
                "_user_id": "user-1",
            })
        )
        assert result == {"success": True, "key": "mood"}

    def test_thread_mem_get(self, mgr):
        """handle_tool_call dispatches thread_mem.get."""
        mgr.thread_set("sess-1", "mood", "curious")
        result = json.loads(
            mgr.handle_tool_call("thread_mem.get", {
                "key": "mood",
                "_gateway_session_key": "sess-1",
                "_user_id": "user-1",
            })
        )
        assert result == {"key": "mood", "value": "curious"}

    def test_thread_mem_get_all(self, mgr):
        """handle_tool_call dispatches thread_mem.get without key."""
        mgr.thread_set("sess-1", "mood", "curious")
        result = json.loads(
            mgr.handle_tool_call("thread_mem.get", {
                "_gateway_session_key": "sess-1",
                "_user_id": "user-1",
            })
        )
        assert result["keys"]["mood"] == "curious"

    def test_thread_mem_delete(self, mgr):
        """handle_tool_call dispatches thread_mem.delete."""
        mgr.thread_set("sess-1", "mood", "curious")
        result = json.loads(
            mgr.handle_tool_call("thread_mem.delete", {
                "key": "mood",
                "_gateway_session_key": "sess-1",
                "_user_id": "user-1",
            })
        )
        assert result == {"success": True, "key": "mood"}

    def test_user_mem_set(self, mgr):
        """handle_tool_call dispatches user_mem.set."""
        result = json.loads(
            mgr.handle_tool_call("user_mem.set", {
                "key": "preference",
                "value": "dark-mode",
                "_gateway_session_key": "sess-1",
                "_user_id": "user-1",
            })
        )
        assert result == {"success": True, "key": "preference"}

    def test_user_mem_get(self, mgr):
        """handle_tool_call dispatches user_mem.get."""
        mgr.user_set("user-1", "preference", "dark-mode")
        result = json.loads(
            mgr.handle_tool_call("user_mem.get", {
                "key": "preference",
                "_gateway_session_key": "sess-1",
                "_user_id": "user-1",
            })
        )
        assert result == {"key": "preference", "value": "dark-mode"}

    def test_user_mem_delete(self, mgr):
        """handle_tool_call dispatches user_mem.delete."""
        mgr.user_set("user-1", "preference", "dark-mode")
        result = json.loads(
            mgr.handle_tool_call("user_mem.delete", {
                "key": "preference",
                "_gateway_session_key": "sess-1",
                "_user_id": "user-1",
            })
        )
        assert result == {"success": True, "key": "preference"}

    def test_thread_mem_requires_session_key(self, mgr):
        """handle_tool_call returns error when session_key is empty for thread_mem."""
        result = json.loads(
            mgr.handle_tool_call("thread_mem.set", {
                "key": "mood",
                "value": "curious",
                "_gateway_session_key": "",
                "_user_id": "user-1",
            })
        )
        assert "error" in result
        assert "thread_mem" in result["error"]

    def test_thread_mem_missing_session_key(self, mgr):
        """handle_tool_call returns error when _gateway_session_key is absent."""
        result = json.loads(
            mgr.handle_tool_call("thread_mem.set", {
                "key": "mood",
                "value": "curious",
                "_user_id": "user-1",
            })
        )
        assert "error" in result

    def test_user_mem_requires_user_id(self, mgr):
        """handle_tool_call returns error when user_id is empty for user_mem."""
        result = json.loads(
            mgr.handle_tool_call("user_mem.set", {
                "key": "preference",
                "value": "dark-mode",
                "_gateway_session_key": "sess-1",
                "_user_id": "",
            })
        )
        assert "error" in result
        assert "user_mem" in result["error"] or "user" in result["error"].lower()

    def test_user_mem_missing_user_id(self, mgr):
        """handle_tool_call returns error when _user_id is absent."""
        result = json.loads(
            mgr.handle_tool_call("user_mem.set", {
                "key": "preference",
                "value": "dark-mode",
                "_gateway_session_key": "sess-1",
            })
        )
        assert "error" in result

    def test_unknown_function_name(self, mgr):
        """handle_tool_call returns error for unknown function names."""
        result = json.loads(
            mgr.handle_tool_call("unknown.func", {
                "_gateway_session_key": "sess-1",
                "_user_id": "user-1",
            })
        )
        assert "error" in result


# =========================================================================
# Error handling
# =========================================================================


class TestErrorHandling:
    def test_thread_set_db_error(self, mgr, db):
        """thread_set returns error JSON on database failure."""
        # Force a DB error by closing the connection
        db.close()

        result = json.loads(mgr.thread_set("sess-1", "mood", "curious"))
        assert "error" in result

    def test_thread_get_db_error(self, mgr, db):
        """thread_get returns error JSON on database failure."""
        db.close()

        result = json.loads(mgr.thread_get("sess-1", "mood"))
        assert "error" in result

    def test_user_set_db_error(self, mgr, db):
        """user_set returns error JSON on database failure."""
        db.close()

        result = json.loads(mgr.user_set("user-1", "pref", "dark"))
        assert "error" in result

    def test_user_get_db_error(self, mgr, db):
        """user_get returns error JSON on database failure."""
        db.close()

        result = json.loads(mgr.user_get("user-1", "pref"))
        assert "error" in result

    def test_prefetch_db_error(self, mgr, db):
        """prefetch returns empty string on database failure (graceful)."""
        db.close()

        # prefetch should not raise — it catches exceptions and returns ""
        result = mgr.prefetch("sess-1", "user-1")
        assert result == ""

    def test_delete_thread_memory_db_error(self, mgr, db):
        """delete_thread_memory does not raise on database failure."""
        db.close()

        # Should not raise, even with a closed DB
        mgr.delete_thread_memory("sess-1")

    def test_handle_tool_call_db_error(self, mgr, db):
        """handle_tool_call returns error JSON on database failure."""
        db.close()

        result = json.loads(
            mgr.handle_tool_call("thread_mem.set", {
                "key": "mood",
                "value": "curious",
                "_gateway_session_key": "sess-1",
                "_user_id": "user-1",
            })
        )
        assert "error" in result