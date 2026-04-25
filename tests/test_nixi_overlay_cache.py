"""Tests for NixiAdapter overlay cache — TTL-based caching of employee overlay lookups.

Verifies:
- Cache hit returns overlay without calling load_overlay()
- Cache expiry triggers re-read from disk
- Eviction on insert: expired entries evicted first, then LRU by insertion order
- No eviction on read access (O(1) lookup path)
- Empty overlays (new users with no USER.md) are cached too
- Config resolution: NIXI_OVERLAY_CACHE_TTL and NIXI_OVERLAY_CACHE_MAX from extra/env/default
"""

import asyncio
import os
import time
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_config(**overrides):
    """Create a PlatformConfig for NixiAdapter with sensible defaults."""
    extra = {
        "internal_secret": "test-secret-12345678",
        "team_id": "T_TEAM_TEST",
        "host": "127.0.0.1",
        "port": 0,
    }
    extra.update(overrides)
    return PlatformConfig(enabled=True, extra=extra)


# ─── Cache initialization ────────────────────────────────────────────────────


class TestOverlayCacheInit:
    """Tests for overlay cache attribute initialization."""

    def test_cache_initialized_as_ordered_dict(self):
        """Overlay cache should be an OrderedDict."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config())
        assert isinstance(adapter._overlay_cache, OrderedDict)

    def test_default_ttl_is_300(self):
        """Default cache TTL should be 300 seconds."""
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NIXI_OVERLAY_CACHE_TTL", None)
            adapter = NixiAdapter(_make_config())
            assert adapter._overlay_cache_ttl == 300

    def test_default_max_is_256(self):
        """Default cache max should be 256 entries."""
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NIXI_OVERLAY_CACHE_MAX", None)
            adapter = NixiAdapter(_make_config())
            assert adapter._overlay_cache_max == 256

    def test_ttl_from_config_extra(self):
        """NIXI_OVERLAY_CACHE_TTL from config extra takes precedence."""
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {"NIXI_OVERLAY_CACHE_TTL": "600"}, clear=False):
            adapter = NixiAdapter(_make_config(NIXI_OVERLAY_CACHE_TTL="120"))
            assert adapter._overlay_cache_ttl == 120

    def test_ttl_from_env_var(self):
        """NIXI_OVERLAY_CACHE_TTL from env var when config extra not set."""
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(
            enabled=True,
            extra={
                "internal_secret": "test-secret-12345678",
                "team_id": "T_TEAM_TEST",
                "host": "127.0.0.1",
                "port": 0,
            },
        )
        with patch.dict(os.environ, {"NIXI_OVERLAY_CACHE_TTL": "600"}, clear=False):
            adapter = NixiAdapter(config)
            assert adapter._overlay_cache_ttl == 600

    def test_max_from_config_extra(self):
        """NIXI_OVERLAY_CACHE_MAX from config extra takes precedence."""
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {"NIXI_OVERLAY_CACHE_MAX": "512"}, clear=False):
            adapter = NixiAdapter(_make_config(NIXI_OVERLAY_CACHE_MAX="64"))
            assert adapter._overlay_cache_max == 64

    def test_max_from_env_var(self):
        """NIXI_OVERLAY_CACHE_MAX from env var when config extra not set."""
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(
            enabled=True,
            extra={
                "internal_secret": "test-secret-12345678",
                "team_id": "T_TEAM_TEST",
                "host": "127.0.0.1",
                "port": 0,
            },
        )
        with patch.dict(os.environ, {"NIXI_OVERLAY_CACHE_MAX": "512"}, clear=False):
            adapter = NixiAdapter(config)
            assert adapter._overlay_cache_max == 512

    def test_invalid_ttl_falls_back_to_default(self):
        """Non-integer TTL falls back to 300."""
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NIXI_OVERLAY_CACHE_TTL", None)
            adapter = NixiAdapter(_make_config(NIXI_OVERLAY_CACHE_TTL="not_a_number"))
            assert adapter._overlay_cache_ttl == 300

    def test_invalid_max_falls_back_to_default(self):
        """Non-integer max falls back to 256."""
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NIXI_OVERLAY_CACHE_MAX", None)
            adapter = NixiAdapter(_make_config(NIXI_OVERLAY_CACHE_MAX="abc"))
            assert adapter._overlay_cache_max == 256


# ─── Cache hit: returns overlay without re-reading disk ───────────────────────


class TestOverlayCacheHit:
    """Tests for cache hit behavior — overlay returned from cache without disk read."""

    def test_cache_hit_returns_overlay_without_second_load_overlay_call(self):
        """After first call caches the overlay, second call should not call load_overlay again."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config())

        call_count = {"value": 0}

        def mock_load_overlay(user_id):
            call_count["value"] += 1
            if call_count["value"] > 1:
                raise RuntimeError("load_overlay should not be called more than once for cache hit")
            return "# Cached overlay content"

        with patch("nixi.gateway_adapter.load_overlay", side_effect=mock_load_overlay):
            # First call: cache miss, loads from disk
            result1 = adapter._get_overlay_with_cache("user1")
            assert result1 == "# Cached overlay content"
            assert call_count["value"] == 1

            # Second call: cache hit, should NOT call load_overlay again
            result2 = adapter._get_overlay_with_cache("user1")
            assert result2 == "# Cached overlay content"
            assert call_count["value"] == 1  # Still 1, no second call


# ─── Cache expiry ────────────────────────────────────────────────────────────


class TestOverlayCacheExpiry:
    """Tests for TTL-based cache expiry."""

    def test_expired_entry_triggers_re_read(self):
        """When TTL expires, load_overlay should be called again."""
        from nixi.gateway_adapter import NixiAdapter

        # Use a very short TTL
        adapter = NixiAdapter(_make_config(NIXI_OVERLAY_CACHE_TTL="0"))
        adapter._overlay_cache_ttl = 0  # Ensure immediate expiry

        call_count = {"value": 0}

        def mock_load_overlay(user_id):
            call_count["value"] += 1
            return f"overlay_v{call_count['value']}"

        with patch("nixi.gateway_adapter.load_overlay", side_effect=mock_load_overlay):
            # TTL=0 means entries expire immediately
            result1 = adapter._get_overlay_with_cache("user1")
            assert result1 == "overlay_v1"
            assert call_count["value"] == 1

            # Each call should re-read because TTL is 0
            result2 = adapter._get_overlay_with_cache("user1")
            assert result2 == "overlay_v2"
            assert call_count["value"] == 2

    def test_entry_within_ttl_is_served_from_cache(self):
        """When within TTL, cached entry is returned without re-reading."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config(NIXI_OVERLAY_CACHE_TTL="300"))

        call_count = {"value": 0}

        def mock_load_overlay(user_id):
            call_count["value"] += 1
            return f"overlay_v{call_count['value']}"

        with patch("nixi.gateway_adapter.load_overlay", side_effect=mock_load_overlay):
            result1 = adapter._get_overlay_with_cache("user1")
            assert result1 == "overlay_v1"
            assert call_count["value"] == 1

            # Within TTL — should return cached value
            result2 = adapter._get_overlay_with_cache("user1")
            assert result2 == "overlay_v1"  # Same result = cached
            assert call_count["value"] == 1  # No additional call


# ─── Eviction on insert ──────────────────────────────────────────────────────


class TestOverlayCacheEvictionOnInsert:
    """Tests for cache eviction when max entries exceeded on insert."""

    def test_eviction_removes_expired_entries_first(self):
        """When cache exceeds max, expired entries are evicted first."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config(NIXI_OVERLAY_CACHE_TTL="300", NIXI_OVERLAY_CACHE_MAX="3"))
        adapter._overlay_cache_ttl = 300
        adapter._overlay_cache_max = 3

        # Fill cache with 3 entries where the first has an old timestamp
        current_time = time.time()
        # user1: expired (timestamp in the past)
        adapter._overlay_cache["user1"] = ("expired_overlay", current_time - 600)
        # user2 and user3: fresh
        adapter._overlay_cache["user2"] = ("overlay2", current_time)
        adapter._overlay_cache["user3"] = ("overlay3", current_time)

        call_count = {"value": 0}

        def mock_load_overlay(user_id):
            call_count["value"] += 1
            return f"new_overlay_{user_id}"

        with patch("nixi.gateway_adapter.load_overlay", side_effect=mock_load_overlay):
            # Insert one more — should evict expired user1
            result = adapter._get_overlay_with_cache("user4")
            assert result == "new_overlay_user4"

            # user1 (expired) should be evicted, user2 and user3 still present
            assert "user1" not in adapter._overlay_cache
            assert "user2" in adapter._overlay_cache
            assert "user3" in adapter._overlay_cache
            assert "user4" in adapter._overlay_cache

    def test_eviction_lru_after_removing_expired(self):
        """If still over max after removing expired, LRU (oldest insert) is evicted."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config(NIXI_OVERLAY_CACHE_TTL="300", NIXI_OVERLAY_CACHE_MAX="2"))
        adapter._overlay_cache_ttl = 300
        adapter._overlay_cache_max = 2

        current_time = time.time()

        def mock_load_overlay(user_id):
            return f"overlay_{user_id}"

        with patch("nixi.gateway_adapter.load_overlay", side_effect=mock_load_overlay):
            # Fill cache to max (2)
            adapter._get_overlay_with_cache("user1")
            adapter._get_overlay_with_cache("user2")

            # Both are fresh; inserting user3 should evict user1 (oldest insert)
            adapter._get_overlay_with_cache("user3")

            assert "user1" not in adapter._overlay_cache  # Evicted — oldest
            assert "user2" in adapter._overlay_cache
            assert "user3" in adapter._overlay_cache

    def test_no_eviction_on_read_access(self):
        """Read access should never trigger eviction — O(1) lookup path."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config(NIXI_OVERLAY_CACHE_TTL="300", NIXI_OVERLAY_CACHE_MAX="3"))
        adapter._overlay_cache_ttl = 300
        adapter._overlay_cache_max = 3

        current_time = time.time()

        # Pre-populate cache exactly at max
        adapter._overlay_cache["user1"] = ("overlay1", current_time)
        adapter._overlay_cache["user2"] = ("overlay2", current_time)
        adapter._overlay_cache["user3"] = ("overlay3", current_time)

        # Read all entries multiple times — no eviction should happen
        with patch("nixi.gateway_adapter.load_overlay", return_value="should_not_be_called"):
            result1 = adapter._get_overlay_with_cache("user1")
            result2 = adapter._get_overlay_with_cache("user2")
            result3 = adapter._get_overlay_with_cache("user3")

            assert result1 == "overlay1"
            assert result2 == "overlay2"
            assert result3 == "overlay3"

            # All entries still present — read doesn't evict
            assert len(adapter._overlay_cache) == 3
            assert "user1" in adapter._overlay_cache
            assert "user2" in adapter._overlay_cache
            assert "user3" in adapter._overlay_cache


# ─── Empty overlay caching ──────────────────────────────────────────────────


class TestOverlayCacheEmptyOverlay:
    """Tests that empty overlays (users with no USER.md) are also cached."""

    def test_empty_overlay_is_cached(self):
        """Empty string overlay should be cached, preventing repeated disk reads."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config())

        call_count = {"value": 0}

        def mock_load_overlay(user_id):
            call_count["value"] += 1
            if call_count["value"] > 1:
                raise RuntimeError("load_overlay should not be called again for cached empty overlay")
            return ""  # Empty overlay — no USER.md file exists yet

        with patch("nixi.gateway_adapter.load_overlay", side_effect=mock_load_overlay):
            result1 = adapter._get_overlay_with_cache("newuser")
            assert result1 == ""
            assert call_count["value"] == 1

            # Second call should return cached empty string without disk read
            result2 = adapter._get_overlay_with_cache("newuser")
            assert result2 == ""
            assert call_count["value"] == 1  # No additional call

    def test_different_users_cached_independently(self):
        """Each user's overlay should be cached independently."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config())

        def mock_load_overlay(user_id):
            overlays = {
                "user1": "# Manager context",
                "user2": "# Engineer context",
                "user3": "",  # No overlay yet
            }
            return overlays.get(user_id, "")

        with patch("nixi.gateway_adapter.load_overlay", side_effect=mock_load_overlay):
            assert adapter._get_overlay_with_cache("user1") == "# Manager context"
            assert adapter._get_overlay_with_cache("user2") == "# Engineer context"
            assert adapter._get_overlay_with_cache("user3") == ""

            # All three should be in the cache
            assert len(adapter._overlay_cache) == 3
            assert "user1" in adapter._overlay_cache
            assert "user2" in adapter._overlay_cache
            assert "user3" in adapter._overlay_cache


# ─── _dispatch_event uses cache ───────────────────────────────────────────────


class TestDispatchEventUsesCache:
    """Tests that _dispatch_event uses the cached overlay."""

    @pytest.mark.asyncio
    async def test_dispatch_event_uses_overlay_cache(self):
        """_dispatch_event should use _get_overlay_with_cache instead of direct load_overlay."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config())

        load_calls = {"value": 0}

        def mock_load_overlay(user_id):
            load_calls["value"] += 1
            if load_calls["value"] > 1:
                raise RuntimeError("load_overlay called more than once for same user")
            return "# Employee context"

        with patch("nixi.gateway_adapter.load_overlay", side_effect=mock_load_overlay):
            called_events = []

            async def capture_event(event):
                called_events.append(event)
                return "response"

            adapter._message_handler = capture_event

            # First dispatch: loads from disk
            await adapter._dispatch_event(
                event_data={"event": {"text": "hello", "channel": "C1"}},
                user_id="U_CACHE_TEST",
                user_name="CacheTestUser",
            )

            await asyncio.sleep(0.1)

            assert len(called_events) == 1
            assert called_events[0].channel_prompt == "# Employee context"
            assert load_calls["value"] == 1

            # Second dispatch for same user: should use cache
            called_events.clear()

            await adapter._dispatch_event(
                event_data={"event": {"text": "hello again", "channel": "C1"}},
                user_id="U_CACHE_TEST",
                user_name="CacheTestUser",
            )

            await asyncio.sleep(0.1)

            assert len(called_events) == 1
            assert called_events[0].channel_prompt == "# Employee context"
            # load_overlay still only called once — cache served the second request
            assert load_calls["value"] == 1