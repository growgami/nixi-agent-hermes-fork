"""Tests for tools/thread_mem.py — tool registration for thread_mem.set/get/delete."""

import json

import pytest


from tools.registry import registry


@pytest.fixture(autouse=True)
def _import_thread_mem():
    """Ensure thread_mem module is imported (registers tools)."""
    import tools.thread_mem  # noqa: F401


class TestThreadMemSchema:
    """Validate schema shape for each thread_mem tool."""

    def test_thread_mem_set_schema(self):
        entry = registry.get_entry("thread_mem.set")
        assert entry is not None, "thread_mem.set not registered"
        schema = entry.schema
        assert schema["name"] == "thread_mem.set"
        params = schema["parameters"]
        assert params["type"] == "object"
        assert "key" in params["properties"]
        assert "value" in params["properties"]
        assert "key" in params.get("required", [])
        assert "value" in params.get("required", [])

    def test_thread_mem_get_schema(self):
        entry = registry.get_entry("thread_mem.get")
        assert entry is not None
        schema = entry.schema
        assert schema["name"] == "thread_mem.get"
        params = schema["parameters"]
        assert "key" in params["properties"]
        assert "key" in params.get("required", [])

    def test_thread_mem_delete_schema(self):
        entry = registry.get_entry("thread_mem.delete")
        assert entry is not None
        schema = entry.schema
        assert schema["name"] == "thread_mem.delete"
        params = schema["parameters"]
        assert "key" in params["properties"]
        assert "key" in params.get("required", [])


class TestThreadMemHandler:
    """Test handler lambdas dispatch correctly to ThreadUserMemoryManager."""

    def _make_mock_manager(self):
        """Create a mock ThreadUserMemoryManager that returns JSON via handle_tool_call."""
        from unittest.mock import MagicMock
        mgr = MagicMock()
        mgr.handle_tool_call.return_value = json.dumps({"success": True, "key": "k1"})
        return mgr

    def test_thread_mem_set_dispatches(self):
        mgr = self._make_mock_manager()
        entry = registry.get_entry("thread_mem.set")
        result = entry.handler(
            {"key": "k1", "value": "v1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="sess-abc",
            _user_id="user-1",
        )
        mgr.handle_tool_call.assert_called_once_with(
            "thread_mem.set",
            {"key": "k1", "value": "v1", "_gateway_session_key": "sess-abc", "_user_id": "user-1"},
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_thread_mem_get_dispatches(self):
        mgr = self._make_mock_manager()
        entry = registry.get_entry("thread_mem.get")
        result = entry.handler(
            {"key": "k1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="sess-abc",
            _user_id="user-1",
        )
        mgr.handle_tool_call.assert_called_once_with(
            "thread_mem.get",
            {"key": "k1", "_gateway_session_key": "sess-abc", "_user_id": "user-1"},
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_thread_mem_delete_dispatches(self):
        mgr = self._make_mock_manager()
        entry = registry.get_entry("thread_mem.delete")
        result = entry.handler(
            {"key": "k1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="sess-abc",
            _user_id="user-1",
        )
        mgr.handle_tool_call.assert_called_once_with(
            "thread_mem.delete",
            {"key": "k1", "_gateway_session_key": "sess-abc", "_user_id": "user-1"},
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_thread_mem_set_missing_session_key(self):
        """Empty session_key passes through to handle_tool_call which returns error."""
        mgr = self._make_mock_manager()
        # Simulate the actual error from handle_tool_call for empty session_key
        mgr.handle_tool_call.return_value = json.dumps(
            {"error": "thread_mem is only available in gateway sessions"},
        )
        entry = registry.get_entry("thread_mem.set")
        result = entry.handler(
            {"key": "k1", "value": "v1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="",
            _user_id="user-1",
        )
        parsed = json.loads(result)
        assert "error" in parsed

    def test_thread_mem_get_missing_session_key(self):
        mgr = self._make_mock_manager()
        mgr.handle_tool_call.return_value = json.dumps(
            {"error": "thread_mem is only available in gateway sessions"},
        )
        entry = registry.get_entry("thread_mem.get")
        result = entry.handler(
            {"key": "k1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="",
            _user_id="user-1",
        )
        parsed = json.loads(result)
        assert "error" in parsed

    def test_thread_mem_delete_missing_session_key(self):
        mgr = self._make_mock_manager()
        mgr.handle_tool_call.return_value = json.dumps(
            {"error": "thread_mem is only available in gateway sessions"},
        )
        entry = registry.get_entry("thread_mem.delete")
        result = entry.handler(
            {"key": "k1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="",
            _user_id="user-1",
        )
        parsed = json.loads(result)
        assert "error" in parsed

    def test_thread_mem_no_manager_returns_error(self):
        """Without a manager, handler returns error JSON."""
        entry = registry.get_entry("thread_mem.set")
        result = entry.handler(
            {"key": "k1", "value": "v1"},
        )
        parsed = json.loads(result)
        assert "error" in parsed


class TestThreadMemCheckRequirements:
    """check_requirements always returns True (no external deps)."""

    def test_check_requirements(self):
        from tools.thread_mem import check_requirements
        assert check_requirements() is True


class TestRegistryContainsThreadMemTools:
    """Verify all 3 thread_mem tools are in the registry after import."""

    def test_registry_has_all_thread_mem_tools(self):
        import tools.thread_mem  # noqa: F401
        names = registry.get_all_tool_names()
        assert "thread_mem.set" in names
        assert "thread_mem.get" in names
        assert "thread_mem.delete" in names