"""End-to-end verification: registry, dispatch, and full flow for thread/user memory.

Verifies:
1. All 6 tools appear in the registry after discover_builtin_tools().
2. Both dispatch paths (sequential + concurrent via _invoke_tool) route correctly.
3. handle_function_call() returns stub error for thread_mem/user_mem.
4. End-to-end flow: set → get → prefetch → delete works with real SessionDB.
"""

import json
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB
from agent.thread_user_memory import ThreadUserMemoryManager
from tools.registry import registry


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _ensure_tools_imported():
    """Import tool modules so registry is populated."""
    import tools.thread_mem  # noqa: F401
    import tools.user_mem  # noqa: F401


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_e2e.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


@pytest.fixture()
def mgr(db):
    """Create a ThreadUserMemoryManager backed by a real SessionDB."""
    return ThreadUserMemoryManager(db)


# =========================================================================
# 1. Tool registration
# =========================================================================


class TestToolRegistration:
    """Verify all 6 new tool names appear in the registry."""

    EXPECTED_TOOLS = {
        "thread_mem.set",
        "thread_mem.get",
        "thread_mem.delete",
        "user_mem.set",
        "user_mem.get",
        "user_mem.delete",
    }

    def test_all_six_tools_registered(self):
        """All 6 thread_mem/user_mem tools are in the registry."""
        all_names = set(registry.get_all_tool_names())
        missing = self.EXPECTED_TOOLS - all_names
        assert not missing, f"Missing tools: {missing}"

    def test_thread_mem_set_has_valid_entry(self):
        """registry.get_entry('thread_mem.set') returns a valid ToolEntry."""
        entry = registry.get_entry("thread_mem.set")
        assert entry is not None
        assert entry.handler is not None
        assert entry.check_fn is not None

    def test_user_mem_set_has_valid_entry(self):
        """registry.get_entry('user_mem.set') returns a valid ToolEntry."""
        entry = registry.get_entry("user_mem.set")
        assert entry is not None
        assert entry.handler is not None
        assert entry.check_fn is not None

    def test_toolset_names(self):
        """Tools map to 'thread_mem' and 'user_mem' toolsets."""
        for name in ("thread_mem.set", "thread_mem.get", "thread_mem.delete"):
            assert registry.get_toolset_for_tool(name) == "thread_mem"
        for name in ("user_mem.set", "user_mem.get", "user_mem.delete"):
            assert registry.get_toolset_for_tool(name) == "user_mem"


# =========================================================================
# 2. Tool dispatch paths
# =========================================================================


class TestDispatchPaths:
    """Verify thread_mem/user_mem calls route through the correct paths."""

    @pytest.fixture
    def mock_agent(self):
        """Build a minimal AIAgent-like object for _invoke_tool testing."""
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        agent._thread_user_memory_manager = None
        agent._memory_manager = None
        agent._memory_store = MagicMock()
        agent._todo_store = MagicMock()
        agent._session_db = None
        agent.session_id = "test-session"
        agent.valid_tool_names = set()
        agent.quiet_mode = False
        agent.clarify_callback = None
        agent._gateway_session_key = "gw-sess-1"
        agent._user_id = "user-1"
        return agent

    def test_sequential_path_thread_mem_set(self, mock_agent, mgr):
        """_execute_tool_calls_sequential elif branch routes thread_mem.set to manager."""
        mock_agent._thread_user_memory_manager = mgr
        # Directly call the sequential dispatch path by invoking _invoke_tool
        # which contains the elif branch for thread_mem/user_mem
        result = mock_agent._invoke_tool(
            "thread_mem.set",
            {"key": "mood", "value": "curious"},
            effective_task_id="",
        )
        parsed = json.loads(result)
        assert parsed.get("success") is True
        # Verify data persisted
        get_result = json.loads(
            mock_agent._invoke_tool(
                "thread_mem.get",
                {"key": "mood"},
                effective_task_id="",
            )
        )
        assert get_result["value"] == "curious"

    def test_concurrent_path_user_mem_get(self, mock_agent, mgr):
        """_invoke_tool elif branch routes user_mem.get to manager."""
        # First set a value via the manager directly
        mgr.user_set("user-1", "preference", "dark-mode")
        # Now test the concurrent dispatch path
        mock_agent._thread_user_memory_manager = mgr
        result = mock_agent._invoke_tool(
            "user_mem.get",
            {"key": "preference"},
            effective_task_id="",
        )
        parsed = json.loads(result)
        assert parsed["key"] == "preference"
        assert parsed["value"] == "dark-mode"

    def test_handle_function_call_stub_error_thread_mem(self):
        """handle_function_call returns stub error for thread_mem.set."""
        from model_tools import handle_function_call
        result = json.loads(
            handle_function_call("thread_mem.set", {"key": "k", "value": "v"})
        )
        assert "error" in result
        assert "agent loop" in result["error"].lower() or "must be handled" in result["error"].lower()

    def test_handle_function_call_stub_error_user_mem(self):
        """handle_function_call returns stub error for user_mem.get."""
        from model_tools import handle_function_call
        result = json.loads(
            handle_function_call("user_mem.get", {"key": "k"})
        )
        assert "error" in result


# =========================================================================
# 3. End-to-end flow
# =========================================================================


class TestEndToEndFlow:
    """Full flow: set → get → prefetch → delete with real SessionDB."""

    def test_thread_memory_full_flow(self, mgr, db):
        """Thread memory: set → get → prefetch → delete_thread_memory."""
        session_key = "e2e-session-1"

        # Set
        set_result = json.loads(mgr.thread_set(session_key, "topic", "python"))
        assert set_result == {"success": True, "key": "topic"}

        mgr.thread_set(session_key, "mood", "curious")

        # Get
        get_result = json.loads(mgr.thread_get(session_key, "topic"))
        assert get_result == {"key": "topic", "value": "python"}

        # Prefetch
        prefetch_output = mgr.prefetch(session_key, "user-noone")
        assert "<thread-context>" in prefetch_output
        assert "- topic: python" in prefetch_output
        assert "- mood: curious" in prefetch_output

        # Delete all thread memory
        mgr.delete_thread_memory(session_key)
        assert db.get_thread_memory(session_key) == []

        # Prefetch after cleanup returns no thread-context
        prefetch_after = mgr.prefetch(session_key, "user-noone")
        assert "<thread-context>" not in prefetch_after

    def test_user_memory_full_flow(self, mgr, db):
        """User memory: set → get → prefetch → delete."""
        user_id = "user-e2e"

        # Set
        set_result = json.loads(mgr.user_set(user_id, "name", "Alice"))
        assert set_result == {"success": True, "key": "name"}

        # Set with TTL
        mgr.user_set(user_id, "temp_pref", "yes", ttl_hours=24)

        # Get
        get_result = json.loads(mgr.user_get(user_id, "name"))
        assert get_result == {"key": "name", "value": "Alice"}

        # Prefetch
        prefetch_output = mgr.prefetch("no-session", user_id)
        assert "<user-context>" in prefetch_output
        assert "- name: Alice" in prefetch_output
        assert "- temp_pref: yes" in prefetch_output

        # Delete specific key
        del_result = json.loads(mgr.user_delete(user_id, "temp_pref"))
        assert del_result == {"success": True, "key": "temp_pref"}

        # Verify remaining
        remaining = json.loads(mgr.user_get(user_id, "name"))
        assert remaining["value"] == "Alice"

    def test_combined_prefetch(self, mgr, db):
        """Prefetch includes both thread and user context when both have data."""
        session_key = "e2e-combined"
        user_id = "user-combined"

        mgr.thread_set(session_key, "task", "debugging")
        mgr.user_set(user_id, "timezone", "UTC")

        result = mgr.prefetch(session_key, user_id)

        assert "<thread-context>" in result
        assert "<user-context>" in result
        assert "- task: debugging" in result
        assert "- timezone: UTC" in result

    def test_session_reset_cleans_thread_not_user(self, db):
        """Session reset deletes thread memory but preserves user memory."""
        session_key = "sess-reset-test"
        user_id = "user-reset-test"

        db.set_thread_memory(session_key, "context", "api work")
        db.set_user_memory(user_id, "preference", "verbose")

        # Simulate session reset: delete_thread_memory
        db.delete_thread_memory(session_key)

        assert db.get_thread_memory(session_key) == []
        assert db.get_user_memory(user_id) == [("preference", "verbose")]