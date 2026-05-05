"""Tests for tools/user_mem.py — tool registration for user_mem.set/get/delete."""

import json

import pytest


from tools.registry import registry


@pytest.fixture(autouse=True)
def _import_user_mem():
    """Ensure user_mem module is imported (registers tools)."""
    import tools.user_mem  # noqa: F401


class TestUserMemSchema:
    """Validate schema shape for each user_mem tool."""

    def test_user_mem_set_schema(self):
        entry = registry.get_entry("user_mem.set")
        assert entry is not None, "user_mem.set not registered"
        schema = entry.schema
        assert schema["name"] == "user_mem.set"
        params = schema["parameters"]
        assert params["type"] == "object"
        assert "key" in params["properties"]
        assert "value" in params["properties"]
        assert "ttl_hours" in params["properties"]
        assert "key" in params.get("required", [])
        assert "value" in params.get("required", [])
        # ttl_hours is optional — not in required
        assert "ttl_hours" not in params.get("required", [])

    def test_user_mem_get_schema(self):
        entry = registry.get_entry("user_mem.get")
        assert entry is not None
        schema = entry.schema
        assert schema["name"] == "user_mem.get"
        params = schema["parameters"]
        assert "key" in params["properties"]
        assert "key" in params.get("required", [])

    def test_user_mem_delete_schema(self):
        entry = registry.get_entry("user_mem.delete")
        assert entry is not None
        schema = entry.schema
        assert schema["name"] == "user_mem.delete"
        params = schema["parameters"]
        assert "key" in params["properties"]
        assert "key" in params.get("required", [])


class TestUserMemHandler:
    """Test handler lambdas dispatch correctly to ThreadUserMemoryManager."""

    def _make_mock_manager(self):
        """Create a mock ThreadUserMemoryManager that returns JSON via handle_tool_call."""
        from unittest.mock import MagicMock
        mgr = MagicMock()
        mgr.handle_tool_call.return_value = json.dumps({"success": True, "key": "k1"})
        return mgr

    def test_user_mem_set_dispatches(self):
        mgr = self._make_mock_manager()
        entry = registry.get_entry("user_mem.set")
        result = entry.handler(
            {"key": "k1", "value": "v1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="sess-abc",
            _user_id="user-1",
        )
        mgr.handle_tool_call.assert_called_once_with(
            "user_mem.set",
            {"key": "k1", "value": "v1", "_gateway_session_key": "sess-abc", "_user_id": "user-1"},
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_user_mem_set_with_ttl_hours(self):
        mgr = self._make_mock_manager()
        entry = registry.get_entry("user_mem.set")
        result = entry.handler(
            {"key": "k1", "value": "v1", "ttl_hours": 24},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="sess-abc",
            _user_id="user-1",
        )
        mgr.handle_tool_call.assert_called_once_with(
            "user_mem.set",
            {"key": "k1", "value": "v1", "ttl_hours": 24, "_gateway_session_key": "sess-abc", "_user_id": "user-1"},
        )

    def test_user_mem_get_dispatches(self):
        mgr = self._make_mock_manager()
        entry = registry.get_entry("user_mem.get")
        result = entry.handler(
            {"key": "k1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="sess-abc",
            _user_id="user-1",
        )
        mgr.handle_tool_call.assert_called_once_with(
            "user_mem.get",
            {"key": "k1", "_gateway_session_key": "sess-abc", "_user_id": "user-1"},
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_user_mem_delete_dispatches(self):
        mgr = self._make_mock_manager()
        entry = registry.get_entry("user_mem.delete")
        result = entry.handler(
            {"key": "k1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="sess-abc",
            _user_id="user-1",
        )
        mgr.handle_tool_call.assert_called_once_with(
            "user_mem.delete",
            {"key": "k1", "_gateway_session_key": "sess-abc", "_user_id": "user-1"},
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_user_mem_set_missing_user_id(self):
        """Empty user_id passes through to handle_tool_call which returns error."""
        mgr = self._make_mock_manager()
        mgr.handle_tool_call.return_value = json.dumps(
            {"error": "user_mem is only available for identified users"},
        )
        entry = registry.get_entry("user_mem.set")
        result = entry.handler(
            {"key": "k1", "value": "v1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="sess-abc",
            _user_id="",
        )
        parsed = json.loads(result)
        assert "error" in parsed

    def test_user_mem_get_missing_user_id(self):
        mgr = self._make_mock_manager()
        mgr.handle_tool_call.return_value = json.dumps(
            {"error": "user_mem is only available for identified users"},
        )
        entry = registry.get_entry("user_mem.get")
        result = entry.handler(
            {"key": "k1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="sess-abc",
            _user_id="",
        )
        parsed = json.loads(result)
        assert "error" in parsed

    def test_user_mem_delete_missing_user_id(self):
        mgr = self._make_mock_manager()
        mgr.handle_tool_call.return_value = json.dumps(
            {"error": "user_mem is only available for identified users"},
        )
        entry = registry.get_entry("user_mem.delete")
        result = entry.handler(
            {"key": "k1"},
            _thread_user_memory_manager=mgr,
            _gateway_session_key="sess-abc",
            _user_id="",
        )
        parsed = json.loads(result)
        assert "error" in parsed

    def test_user_mem_no_manager_returns_error(self):
        """Without a manager, handler returns error JSON."""
        entry = registry.get_entry("user_mem.set")
        result = entry.handler(
            {"key": "k1", "value": "v1"},
        )
        parsed = json.loads(result)
        assert "error" in parsed


class TestUserMemCheckRequirements:
    """check_requirements always returns True (no external deps)."""

    def test_check_requirements(self):
        from tools.user_mem import check_requirements
        assert check_requirements() is True


class TestRegistryContainsUserMemTools:
    """Verify all 3 user_mem tools are in the registry after import."""

    def test_registry_has_all_user_mem_tools(self):
        import tools.user_mem  # noqa: F401
        names = registry.get_all_tool_names()
        assert "user_mem.set" in names
        assert "user_mem.get" in names
        assert "user_mem.delete" in names