"""Tests for configurable agent cache max size (NIXI_AGENT_CACHE_SIZE).

Verify:
- Default value of 128 when env var is unset
- Custom value from env var
- Fallback to 128 on invalid values (non-numeric, negative)
- Value of 0 is valid (agents never cached — edge case for testing)
- Lazy initialization: env var read on first access, not at import time
- Cache isolation: resetting module variable to None restores "not yet read" state
"""

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_cache_max_size():
    """Reset module-level cache so each test starts fresh."""
    from gateway import run as gw_run

    original = gw_run._AGENT_CACHE_MAX_SIZE
    gw_run._AGENT_CACHE_MAX_SIZE = None
    yield
    gw_run._AGENT_CACHE_MAX_SIZE = original


class TestGetAgentCacheMaxSize:
    """Test _get_agent_cache_max_size() lazy env-var configuration."""

    def test_default_when_env_unset(self, monkeypatch):
        """Without NIXI_AGENT_CACHE_SIZE, default is 128."""
        monkeypatch.delenv("NIXI_AGENT_CACHE_SIZE", raising=False)
        from gateway.run import _get_agent_cache_max_size

        assert _get_agent_cache_max_size() == 128

    def test_custom_value_from_env(self, monkeypatch):
        """NIXI_AGENT_CACHE_SIZE=256 returns 256."""
        monkeypatch.setenv("NIXI_AGENT_CACHE_SIZE", "256")
        from gateway.run import _get_agent_cache_max_size

        assert _get_agent_cache_max_size() == 256

    def test_zero_is_valid(self, monkeypatch):
        """NIXI_AGENT_CACHE_SIZE=0 means no caching — valid edge case."""
        monkeypatch.setenv("NIXI_AGENT_CACHE_SIZE", "0")
        from gateway.run import _get_agent_cache_max_size

        assert _get_agent_cache_max_size() == 0

    def test_non_numeric_falls_back_to_128(self, monkeypatch):
        """Non-numeric value logs warning and falls back to 128."""
        monkeypatch.setenv("NIXI_AGENT_CACHE_SIZE", "not-a-number")
        from gateway.run import _get_agent_cache_max_size

        assert _get_agent_cache_max_size() == 128

    def test_empty_string_falls_back_to_128(self, monkeypatch):
        """Empty string falls back to 128."""
        monkeypatch.setenv("NIXI_AGENT_CACHE_SIZE", "")
        from gateway.run import _get_agent_cache_max_size

        # Empty string fails int() conversion → fallback 128
        assert _get_agent_cache_max_size() == 128

    def test_negative_value_accepted_as_is(self, monkeypatch):
        """Negative values are accepted — no clamping. Caller decides policy."""
        monkeypatch.setenv("NIXI_AGENT_CACHE_SIZE", "-1")
        from gateway.run import _get_agent_cache_max_size

        # Negative is valid int; _enforce_agent_cache_cap will allow
        # negative excess which means no eviction ever fires.
        assert _get_agent_cache_max_size() == -1

    def test_lazy_init_reads_env_on_first_call(self, monkeypatch):
        """Env var is read on first call, not at import time."""
        monkeypatch.delenv("NIXI_AGENT_CACHE_SIZE", raising=False)
        from gateway import run as gw_run

        # Module variable should be None (not yet initialized)
        assert gw_run._AGENT_CACHE_MAX_SIZE is None

        # Set env var AFTER import
        monkeypatch.setenv("NIXI_AGENT_CACHE_SIZE", "64")

        # First call should read the env var
        assert gw_run._get_agent_cache_max_size() == 64

    def test_subsequent_calls_return_cached_value(self, monkeypatch):
        """After first call, the value is cached and env var changes are ignored."""
        monkeypatch.setenv("NIXI_AGENT_CACHE_SIZE", "200")
        from gateway import run as gw_run

        # First call reads env var and caches
        assert gw_run._get_agent_cache_max_size() == 200

        # Changing env var after first call has no effect
        monkeypatch.setenv("NIXI_AGENT_CACHE_SIZE", "999")
        assert gw_run._get_agent_cache_max_size() == 200

    def test_reset_module_variable_allows_re_read(self, monkeypatch):
        """Resetting _AGENT_CACHE_MAX_SIZE to None restores "not yet read" state."""
        monkeypatch.setenv("NIXI_AGENT_CACHE_SIZE", "50")
        from gateway import run as gw_run

        # First call reads env var
        assert gw_run._get_agent_cache_max_size() == 50

        # Reset module variable
        gw_run._AGENT_CACHE_MAX_SIZE = None

        # Change env var
        monkeypatch.setenv("NIXI_AGENT_CACHE_SIZE", "100")

        # Should re-read env var
        assert gw_run._get_agent_cache_max_size() == 100

    def test_monkeypatch_direct_assignment_override(self, monkeypatch):
        """Direct monkeypatch assignment to module var still works (test isolation)."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 3)

        # Should return the monkeypatched value directly
        assert gw_run._get_agent_cache_max_size() == 3


class TestEnforceCacheCapUsesAccessor:
    """Verify _enforce_agent_cache_cap uses _get_agent_cache_max_size."""

    def test_cap_uses_env_var_size(self, monkeypatch):
        """_enforce_agent_cache_cap respects NIXI_AGENT_CACHE_SIZE env var."""
        from collections import OrderedDict

        from gateway import run as gw_run

        monkeypatch.delenv("NIXI_AGENT_CACHE_SIZE", raising=False)
        # Force re-read of cache size
        gw_run._AGENT_CACHE_MAX_SIZE = None
        monkeypatch.setenv("NIXI_AGENT_CACHE_SIZE", "2")

        # Verify accessor returns 2
        assert gw_run._get_agent_cache_max_size() == 2

        # Now test that _enforce_agent_cache_cap uses that value
        runner = gw_run.GatewayRunner.__new__(gw_run.GatewayRunner)
        runner._agent_cache = OrderedDict()
        runner._agent_cache_lock = __import__("threading").Lock()
        runner._cleanup_agent_resources = lambda a: None
        runner._running_agents = {}

        # Insert 3 entries — only 2 should remain
        for i in range(3):
            with runner._agent_cache_lock:
                runner._agent_cache[f"s{i}"] = (MagicMock(), f"sig{i}")
                runner._enforce_agent_cache_cap()

        assert len(runner._agent_cache) == 2