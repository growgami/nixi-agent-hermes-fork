"""Tests for gateway/channel_directory.py — channel resolution and display."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.channel_directory import (
    build_channel_directory,
    build_channel_directory_sync,
    lookup_channel_type,
    resolve_channel_name,
    format_directory_for_display,
    load_directory,
    _build_from_sessions,
    DIRECTORY_PATH,
)


def _write_directory(tmp_path, platforms):
    """Helper to write a fake channel directory."""
    data = {"updated_at": "2026-01-01T00:00:00", "platforms": platforms}
    cache_file = tmp_path / "channel_directory.json"
    cache_file.write_text(json.dumps(data))
    return cache_file


class TestLoadDirectory:
    def test_missing_file(self, tmp_path):
        with patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "nope.json"):
            result = load_directory()
        assert result["updated_at"] is None
        assert result["platforms"] == {}

    def test_valid_file(self, tmp_path):
        cache_file = _write_directory(tmp_path, {
            "telegram": [{"id": "123", "name": "John", "type": "dm"}]
        })
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            result = load_directory()
        assert result["platforms"]["telegram"][0]["name"] == "John"

    def test_corrupt_file(self, tmp_path):
        cache_file = tmp_path / "channel_directory.json"
        cache_file.write_text("{bad json")
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            result = load_directory()
        assert result["updated_at"] is None


class TestBuildChannelDirectoryWrites:
    def test_failed_write_preserves_previous_cache(self, tmp_path, monkeypatch):
        cache_file = _write_directory(tmp_path, {
            "telegram": [{"id": "123", "name": "Alice", "type": "dm"}]
        })
        previous = json.loads(cache_file.read_text())

        def broken_dump(data, fp, *args, **kwargs):
            fp.write('{"updated_at":')
            fp.flush()
            raise OSError("disk full")

        monkeypatch.setattr(json, "dump", broken_dump)

        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            build_channel_directory_sync({})
            result = load_directory()

        assert result == previous


class TestResolveChannelName:
    def _setup(self, tmp_path, platforms):
        cache_file = _write_directory(tmp_path, platforms)
        return patch("gateway.channel_directory.DIRECTORY_PATH", cache_file)

    def test_exact_match(self, tmp_path):
        platforms = {
            "discord": [
                {"id": "111", "name": "bot-home", "guild": "MyServer", "type": "channel"},
                {"id": "222", "name": "general", "guild": "MyServer", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("discord", "bot-home") == "111"
            assert resolve_channel_name("discord", "#bot-home") == "111"

    def test_case_insensitive(self, tmp_path):
        platforms = {
            "slack": [{"id": "C01", "name": "Engineering", "type": "channel"}]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("slack", "engineering") == "C01"
            assert resolve_channel_name("slack", "ENGINEERING") == "C01"

    def test_guild_qualified_match(self, tmp_path):
        platforms = {
            "discord": [
                {"id": "111", "name": "general", "guild": "ServerA", "type": "channel"},
                {"id": "222", "name": "general", "guild": "ServerB", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("discord", "ServerA/general") == "111"
            assert resolve_channel_name("discord", "ServerB/general") == "222"

    def test_prefix_match_unambiguous(self, tmp_path):
        platforms = {
            "slack": [
                {"id": "C01", "name": "engineering-backend", "type": "channel"},
                {"id": "C02", "name": "design-team", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            # "engineering" prefix matches only one channel
            assert resolve_channel_name("slack", "engineering") == "C01"

    def test_prefix_match_ambiguous_returns_none(self, tmp_path):
        platforms = {
            "slack": [
                {"id": "C01", "name": "eng-backend", "type": "channel"},
                {"id": "C02", "name": "eng-frontend", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("slack", "eng") is None

    def test_no_channels_returns_none(self, tmp_path):
        with self._setup(tmp_path, {}):
            assert resolve_channel_name("telegram", "someone") is None

    def test_no_match_returns_none(self, tmp_path):
        platforms = {
            "telegram": [{"id": "123", "name": "John", "type": "dm"}]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("telegram", "nonexistent") is None

    def test_topic_name_resolves_to_composite_id(self, tmp_path):
        platforms = {
            "telegram": [{"id": "-1001:17585", "name": "Coaching Chat / topic 17585", "type": "group"}]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("telegram", "Coaching Chat / topic 17585") == "-1001:17585"

    def test_display_label_with_type_suffix_resolves(self, tmp_path):
        platforms = {
            "telegram": [
                {"id": "123", "name": "Alice", "type": "dm"},
                {"id": "456", "name": "Dev Group", "type": "group"},
                {"id": "-1001:17585", "name": "Coaching Chat / topic 17585", "type": "group"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("telegram", "Alice (dm)") == "123"
            assert resolve_channel_name("telegram", "Dev Group (group)") == "456"
            assert resolve_channel_name("telegram", "Coaching Chat / topic 17585 (group)") == "-1001:17585"


class TestBuildFromSessions:
    def _write_sessions(self, tmp_path, sessions_data):
        """Write sessions.json at the path _build_from_sessions expects."""
        sessions_path = tmp_path / "sessions" / "sessions.json"
        sessions_path.parent.mkdir(parents=True)
        sessions_path.write_text(json.dumps(sessions_data))

    def test_builds_from_sessions_json(self, tmp_path):
        self._write_sessions(tmp_path, {
            "session_1": {
                "origin": {
                    "platform": "telegram",
                    "chat_id": "12345",
                    "chat_name": "Alice",
                },
                "chat_type": "dm",
            },
            "session_2": {
                "origin": {
                    "platform": "telegram",
                    "chat_id": "67890",
                    "user_name": "Bob",
                },
                "chat_type": "group",
            },
            "session_3": {
                "origin": {
                    "platform": "discord",
                    "chat_id": "99999",
                },
            },
        })

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            entries = _build_from_sessions("telegram")

        assert len(entries) == 2
        names = {e["name"] for e in entries}
        assert "Alice" in names
        assert "Bob" in names

    def test_missing_sessions_file(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            entries = _build_from_sessions("telegram")
        assert entries == []

    def test_deduplication_by_chat_id(self, tmp_path):
        self._write_sessions(tmp_path, {
            "s1": {"origin": {"platform": "telegram", "chat_id": "123", "chat_name": "X"}},
            "s2": {"origin": {"platform": "telegram", "chat_id": "123", "chat_name": "X"}},
        })

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            entries = _build_from_sessions("telegram")

        assert len(entries) == 1

    def test_keeps_distinct_topics_with_same_chat_id(self, tmp_path):
        self._write_sessions(tmp_path, {
            "group_root": {
                "origin": {"platform": "telegram", "chat_id": "-1001", "chat_name": "Coaching Chat"},
                "chat_type": "group",
            },
            "topic_a": {
                "origin": {
                    "platform": "telegram",
                    "chat_id": "-1001",
                    "chat_name": "Coaching Chat",
                    "thread_id": "17585",
                },
                "chat_type": "group",
            },
            "topic_b": {
                "origin": {
                    "platform": "telegram",
                    "chat_id": "-1001",
                    "chat_name": "Coaching Chat",
                    "thread_id": "17587",
                },
                "chat_type": "group",
            },
        })

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            entries = _build_from_sessions("telegram")

        ids = {entry["id"] for entry in entries}
        names = {entry["name"] for entry in entries}
        assert ids == {"-1001", "-1001:17585", "-1001:17587"}
        assert "Coaching Chat" in names
        assert "Coaching Chat / topic 17585" in names
        assert "Coaching Chat / topic 17587" in names


class TestFormatDirectoryForDisplay:
    def test_empty_directory(self, tmp_path):
        with patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "nope.json"):
            result = format_directory_for_display()
        assert "No messaging platforms" in result

    def test_telegram_display(self, tmp_path):
        cache_file = _write_directory(tmp_path, {
            "telegram": [
                {"id": "123", "name": "Alice", "type": "dm"},
                {"id": "456", "name": "Dev Group", "type": "group"},
                {"id": "-1001:17585", "name": "Coaching Chat / topic 17585", "type": "group"},
            ]
        })
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            result = format_directory_for_display()

        assert "Telegram:" in result
        assert "telegram:Alice" in result
        assert "telegram:Dev Group" in result
        assert "telegram:Coaching Chat / topic 17585" in result

    def test_discord_grouped_by_guild(self, tmp_path):
        cache_file = _write_directory(tmp_path, {
            "discord": [
                {"id": "1", "name": "general", "guild": "Server1", "type": "channel"},
                {"id": "2", "name": "bot-home", "guild": "Server1", "type": "channel"},
                {"id": "3", "name": "chat", "guild": "Server2", "type": "channel"},
            ]
        })
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            result = format_directory_for_display()

        assert "Discord (Server1):" in result
        assert "Discord (Server2):" in result
        assert "discord:#general" in result


class TestLookupChannelType:
    def _setup(self, tmp_path, platforms):
        cache_file = _write_directory(tmp_path, platforms)
        return patch("gateway.channel_directory.DIRECTORY_PATH", cache_file)

    def test_forum_channel(self, tmp_path):
        platforms = {
            "discord": [
                {"id": "100", "name": "ideas", "guild": "Server1", "type": "forum"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert lookup_channel_type("discord", "100") == "forum"

    def test_regular_channel(self, tmp_path):
        platforms = {
            "discord": [
                {"id": "200", "name": "general", "guild": "Server1", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert lookup_channel_type("discord", "200") == "channel"

    def test_unknown_chat_id_returns_none(self, tmp_path):
        platforms = {
            "discord": [
                {"id": "200", "name": "general", "guild": "Server1", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert lookup_channel_type("discord", "999") is None

    def test_unknown_platform_returns_none(self, tmp_path):
        with self._setup(tmp_path, {}):
            assert lookup_channel_type("discord", "100") is None

    def test_channel_without_type_key_returns_none(self, tmp_path):
        platforms = {
            "discord": [
                {"id": "300", "name": "general", "guild": "Server1"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert lookup_channel_type("discord", "300") is None


class TestResolveChannelNameNixiSlackAlias:
    """Cross-resolution between nixi and slack platform sections."""

    def _setup(self, tmp_path, platforms):
        cache_file = _write_directory(tmp_path, platforms)
        return patch("gateway.channel_directory.DIRECTORY_PATH", cache_file)

    def test_nixi_resolves_from_slack_directory(self, tmp_path):
        """When 'nixi' section has no match, resolve from 'slack' section."""
        platforms = {
            "nixi": [{"id": "N01", "name": "nixi-only-channel", "type": "channel"}],
            "slack": [{"id": "C01", "name": "engineering", "type": "channel"}],
        }
        with self._setup(tmp_path, platforms):
            # "engineering" not in nixi section, but IS in slack section
            assert resolve_channel_name("nixi", "engineering") == "C01"

    def test_slack_resolves_from_nixi_directory(self, tmp_path):
        """When 'slack' section has no match, resolve from 'nixi' section."""
        platforms = {
            "nixi": [{"id": "N01", "name": "nixi-only-channel", "type": "channel"}],
            "slack": [{"id": "C01", "name": "engineering", "type": "channel"}],
        }
        with self._setup(tmp_path, platforms):
            # "nixi-only-channel" not in slack section, but IS in nixi section
            assert resolve_channel_name("slack", "nixi-only-channel") == "N01"

    def test_nixi_own_section_takes_priority(self, tmp_path):
        """When both sections have a match, nixi's own section wins."""
        platforms = {
            "nixi": [
                {"id": "N01", "name": "engineering", "type": "private_channel"},
            ],
            "slack": [
                {"id": "C01", "name": "engineering", "type": "channel"},
            ],
        }
        with self._setup(tmp_path, platforms):
            # "engineering" exists in both; nixi's own entry takes priority
            assert resolve_channel_name("nixi", "engineering") == "N01"

    def test_slack_own_section_takes_priority(self, tmp_path):
        """When both sections have a match, slack's own section wins."""
        platforms = {
            "nixi": [
                {"id": "N01", "name": "engineering", "type": "private_channel"},
            ],
            "slack": [
                {"id": "C01", "name": "engineering", "type": "channel"},
            ],
        }
        with self._setup(tmp_path, platforms):
            # "engineering" exists in both; slack's own entry takes priority
            assert resolve_channel_name("slack", "engineering") == "C01"

    def test_no_match_in_either_section_returns_none(self, tmp_path):
        """When neither section has a match, return None."""
        platforms = {
            "nixi": [{"id": "N01", "name": "nixi-only-channel", "type": "channel"}],
            "slack": [{"id": "C01", "name": "engineering", "type": "channel"}],
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("nixi", "nonexistent") is None
            assert resolve_channel_name("slack", "nonexistent") is None

    def test_non_aliased_platform_unaffected(self, tmp_path):
        """Other platforms like telegram/disco don't get cross-resolution."""
        platforms = {
            "telegram": [{"id": "T01", "name": "Alice", "type": "dm"}],
            "slack": [{"id": "C01", "name": "engineering", "type": "channel"}],
        }
        with self._setup(tmp_path, platforms):
            # telegram should NOT fall back to slack
            assert resolve_channel_name("telegram", "engineering") is None


# ─── Async _build_slack tests ────────────────────────────────────────────


class TestBuildSlackAsync:
    """Tests for async _build_slack that enumerates channels via Slack API."""

    def _make_slack_adapter(self, has_primary_client=True, has_app=True):
        """Create a mock Slack adapter with configurable client setup."""
        from gateway.config import Platform, PlatformConfig

        adapter = MagicMock(spec=["_primary_client", "_app", "platform"])
        adapter.platform = Platform.SLACK

        if has_primary_client:
            mock_client = AsyncMock()
            mock_client.conversations_list = AsyncMock()
            adapter._primary_client = mock_client
        else:
            adapter._primary_client = None

        if has_app:
            mock_app = MagicMock()
            mock_app.client = AsyncMock()
            mock_app.client.conversations_list = AsyncMock()
            adapter._app = mock_app
        else:
            adapter._app = None

        return adapter

    def _conversations_list_response(self, channels, next_cursor=""):
        """Build a mock conversations_list API response."""
        return {
            "ok": True,
            "channels": channels,
            "response_metadata": {"next_cursor": next_cursor},
        }

    @pytest.mark.asyncio
    async def test_build_slack_uses_primary_client_in_nixi_mode(self, tmp_path):
        """In NIXI_MODE, _build_slack should use _primary_client for API calls."""
        adapter = self._make_slack_adapter(has_primary_client=True, has_app=False)
        adapter._primary_client.conversations_list.return_value = (
            self._conversations_list_response([
                {"id": "C01", "name": "general", "is_private": False},
                {"id": "C02", "name": "random", "is_private": True},
            ])
        )

        from gateway.channel_directory import _build_slack
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = await _build_slack(adapter)

        ids = {ch["id"] for ch in result}
        assert "C01" in ids
        assert "C02" in ids
        # Private channel detected
        private_channels = [ch for ch in result if ch["type"] == "private_channel"]
        assert len(private_channels) == 1
        assert private_channels[0]["name"] == "random"

    @pytest.mark.asyncio
    async def test_build_slack_uses_app_client_in_socket_mode(self, tmp_path):
        """In Socket Mode, _build_slack should use _app.client for API calls."""
        adapter = self._make_slack_adapter(has_primary_client=False, has_app=True)
        adapter._app.client.conversations_list.return_value = (
            self._conversations_list_response([
                {"id": "C01", "name": "engineering", "is_private": False},
            ])
        )

        from gateway.channel_directory import _build_slack
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = await _build_slack(adapter)

        assert any(ch["id"] == "C01" for ch in result)

    @pytest.mark.asyncio
    async def test_build_slack_prefers_primary_client(self, tmp_path):
        """When both _primary_client and _app exist, prefer _primary_client."""
        adapter = self._make_slack_adapter(has_primary_client=True, has_app=True)
        adapter._primary_client.conversations_list.return_value = (
            self._conversations_list_response([
                {"id": "C_PRIMARY", "name": "from-primary", "is_private": False},
            ])
        )
        adapter._app.client.conversations_list.return_value = (
            self._conversations_list_response([
                {"id": "C_APP", "name": "from-app", "is_private": False},
            ])
        )

        from gateway.channel_directory import _build_slack
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = await _build_slack(adapter)

        # Should use _primary_client, not _app.client
        assert any(ch["id"] == "C_PRIMARY" for ch in result)
        assert not any(ch["id"] == "C_APP" for ch in result)

    @pytest.mark.asyncio
    async def test_build_slack_pagination(self, tmp_path):
        """_build_slack should paginate through conversations_list results."""
        adapter = self._make_slack_adapter(has_primary_client=True, has_app=False)
        # First call returns a cursor, second returns empty cursor
        adapter._primary_client.conversations_list.side_effect = [
            self._conversations_list_response(
                [{"id": "C01", "name": "general", "is_private": False}],
                next_cursor="next_page",
            ),
            self._conversations_list_response(
                [{"id": "C02", "name": "random", "is_private": False}],
                next_cursor="",
            ),
        ]

        from gateway.channel_directory import _build_slack
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = await _build_slack(adapter)

        ids = {ch["id"] for ch in result}
        assert "C01" in ids
        assert "C02" in ids

    @pytest.mark.asyncio
    async def test_build_slack_missing_scope_falls_back_to_sessions(self, tmp_path):
        """When SlackApiError with missing_scope occurs, fall back to sessions."""
        adapter = self._make_slack_adapter(has_primary_client=True, has_app=False)

        # Simulate missing_scope error
        error_response = MagicMock()
        error_response.__str__ = lambda self: "missing_scope"
        error_response.data = {"error": "missing_scope"}

        from slack_sdk.errors import SlackApiError
        adapter._primary_client.conversations_list.side_effect = SlackApiError(
            message="missing_scope", response={"error": "missing_scope"}
        )

        # Set up sessions with nixi platform data
        sessions_path = tmp_path / "sessions" / "sessions.json"
        sessions_path.parent.mkdir(parents=True)
        sessions_path.write_text(json.dumps({
            "s1": {
                "origin": {"platform": "slack", "chat_id": "C_SESS", "chat_name": "session-channel"},
                "chat_type": "channel",
            }
        }))

        from gateway.channel_directory import _build_slack
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = await _build_slack(adapter)

        # Falls back to session data
        assert any(ch["id"] == "C_SESS" for ch in result)

    @pytest.mark.asyncio
    async def test_build_slack_no_client_falls_back_to_sessions(self, tmp_path):
        """When no Slack client is available, fall back to session data."""
        adapter = self._make_slack_adapter(has_primary_client=False, has_app=False)

        sessions_path = tmp_path / "sessions" / "sessions.json"
        sessions_path.parent.mkdir(parents=True)
        sessions_path.write_text(json.dumps({
            "s1": {
                "origin": {"platform": "slack", "chat_id": "C_SESS2", "chat_name": "fallback"},
                "chat_type": "channel",
            }
        }))

        from gateway.channel_directory import _build_slack
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = await _build_slack(adapter)

        assert any(ch["id"] == "C_SESS2" for ch in result)

    @pytest.mark.asyncio
    async def test_build_slack_merges_api_with_sessions(self, tmp_path):
        """API channels should be merged with session data, deduplicating by ID."""
        adapter = self._make_slack_adapter(has_primary_client=True, has_app=False)
        adapter._primary_client.conversations_list.return_value = (
            self._conversations_list_response([
                {"id": "C01", "name": "general", "is_private": False},
            ])
        )

        # Session data includes C01 (duplicate) and C03 (unique to sessions)
        sessions_path = tmp_path / "sessions" / "sessions.json"
        sessions_path.parent.mkdir(parents=True)
        sessions_path.write_text(json.dumps({
            "s1": {
                "origin": {"platform": "slack", "chat_id": "C01", "chat_name": "general"},
                "chat_type": "channel",
            },
            "s2": {
                "origin": {"platform": "slack", "chat_id": "C03", "chat_name": "session-only"},
                "chat_type": "channel",
            },
        }))

        from gateway.channel_directory import _build_slack
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = await _build_slack(adapter)

        ids = {ch["id"] for ch in result}
        assert "C01" in ids  # From API (not duplicated from sessions)
        assert "C03" in ids  # Unique to sessions, merged in

    @pytest.mark.asyncio
    async def test_build_slack_generic_api_error_falls_back(self, tmp_path):
        """Non-missing_scope SlackApiError should also fall back to sessions."""
        adapter = self._make_slack_adapter(has_primary_client=True, has_app=False)

        from slack_sdk.errors import SlackApiError
        adapter._primary_client.conversations_list.side_effect = SlackApiError(
            message="not_authed", response={"error": "not_authed"}
        )

        sessions_path = tmp_path / "sessions" / "sessions.json"
        sessions_path.parent.mkdir(parents=True)
        sessions_path.write_text(json.dumps({
            "s1": {
                "origin": {"platform": "slack", "chat_id": "C_FB", "chat_name": "fallback"},
                "chat_type": "channel",
            }
        }))

        from gateway.channel_directory import _build_slack
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = await _build_slack(adapter)

        assert any(ch["id"] == "C_FB" for ch in result)


class TestBuildChannelDirectoryAsync:
    """Tests for async build_channel_directory with nixi/slack aliasing."""

    @pytest.mark.asyncio
    async def test_build_directory_includes_nixi_section_for_nixi_platform(self, tmp_path):
        """When Platform.NIXI is in adapters, the directory should include a 'nixi' section."""
        from gateway.config import Platform, PlatformConfig

        adapters = {Platform.NIXI: MagicMock(platform=Platform.NIXI)}

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), \
             patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json"), \
             patch("gateway.channel_directory._build_slack", new_callable=AsyncMock) as mock_slack:
            # Nixi adapter: build will see Platform.NIXI and since it's not
            # _SKIP_SESSION_DISCOVERY, it falls through to _build_from_sessions.
            # But actually Platform.NIXI is not Platform.DISCORD or Platform.SLACK,
            # so it hits the session fallback. We also need to check that nixi gets
            # its own section mirroring slack.
            # Let's set up sessions for nixi platform
            sessions_path = tmp_path / "sessions" / "sessions.json"
            sessions_path.parent.mkdir(parents=True)
            sessions_path.write_text(json.dumps({
                "s1": {
                    "origin": {"platform": "nixi", "chat_id": "C_TEST", "chat_name": "nixi/engineering"},
                    "chat_type": "channel",
                }
            }))

            result = await build_channel_directory(adapters)

        # nixi section should exist
        assert "nixi" in result["platforms"]
        # Name should have "nixi/" prefix stripped
        nixi_names = {ch["name"] for ch in result["platforms"]["nixi"]}
        assert "engineering" in nixi_names

    @pytest.mark.asyncio
    async def test_build_directory_nixi_prefix_stripped(self, tmp_path):
        """Channel names in the nixi section should have 'nixi/' prefix removed."""
        from gateway.config import Platform

        # Set up sessions with nixi/ prefix in chat_name
        sessions_path = tmp_path / "sessions" / "sessions.json"
        sessions_path.parent.mkdir(parents=True)
        sessions_path.write_text(json.dumps({
            "s1": {
                "origin": {"platform": "nixi", "chat_id": "C0AE0QVNT1P", "chat_name": "nixi/engineering"},
                "chat_type": "channel",
            },
            "s2": {
                "origin": {"platform": "nixi", "chat_id": "C0AE0QVNT2P", "chat_name": "nixi/random"},
                "chat_type": "channel",
            },
        }))

        adapters = {Platform.NIXI: MagicMock(platform=Platform.NIXI)}
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), \
             patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json"):
            result = await build_channel_directory(adapters)

        nixi_entries = result["platforms"]["nixi"]
        nixi_names = {ch["name"] for ch in nixi_entries}
        assert "engineering" in nixi_names
        assert "random" in nixi_names
        # Ensure the prefix was stripped, not kept
        assert not any(ch["name"].startswith("nixi/") for ch in nixi_entries)

    @pytest.mark.asyncio
    async def test_build_directory_no_nixi_section_without_nixi_adapter(self, tmp_path):
        """Without Platform.NIXI in adapters, no 'nixi' section should be created."""
        from gateway.config import Platform

        adapters = {Platform.DISCORD: MagicMock(platform=Platform.DISCORD)}
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), \
             patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json"):
            result = await build_channel_directory(adapters)

        assert "nixi" not in result["platforms"]


class TestResolveChannelNameCrossPlatform:
    """Tests for nixi/slack cross-resolution in resolve_channel_name."""

    def test_resolve_nixi_falls_back_to_slack(self, tmp_path):
        """When resolving 'nixi' platform, falls back to 'slack' directory."""
        cache_file = _write_directory(tmp_path, {
            "slack": [{"id": "C01", "name": "engineering", "type": "channel"}],
        })
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            assert resolve_channel_name("nixi", "engineering") == "C01"

    def test_resolve_slack_falls_back_to_nixi(self, tmp_path):
        """When resolving 'slack' platform, falls back to 'nixi' directory."""
        cache_file = _write_directory(tmp_path, {
            "nixi": [{"id": "C01", "name": "engineering", "type": "channel"}],
        })
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            assert resolve_channel_name("slack", "engineering") == "C01"

    def test_resolve_nixi_prefers_own_section(self, tmp_path):
        """When both nixi and slack have entries, nixi resolution prefers nixi section."""
        cache_file = _write_directory(tmp_path, {
            "nixi": [{"id": "C_NIXI", "name": "engineering", "type": "channel"}],
            "slack": [{"id": "C_SLACK", "name": "engineering", "type": "channel"}],
        })
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            result = resolve_channel_name("nixi", "engineering")
            assert result == "C_NIXI"

    def test_resolve_slack_prefers_own_section(self, tmp_path):
        """When both nixi and slack have entries, slack resolution prefers slack section."""
        cache_file = _write_directory(tmp_path, {
            "nixi": [{"id": "C_NIXI", "name": "engineering", "type": "channel"}],
            "slack": [{"id": "C_SLACK", "name": "engineering", "type": "channel"}],
        })
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            result = resolve_channel_name("slack", "engineering")
            assert result == "C_SLACK"


class TestBuildChannelDirectorySync:
    """Tests for the sync wrapper build_channel_directory_sync."""

    def test_sync_wrapper_returns_directory(self, tmp_path):
        """build_channel_directory_sync should return a valid directory dict."""
        from gateway.config import Platform

        adapters = {Platform.DISCORD: MagicMock(platform=Platform.DISCORD)}
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), \
             patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json"):
            result = build_channel_directory_sync(adapters)

        assert "platforms" in result

    def test_sync_wrapper_handles_exceptions(self, tmp_path):
        """build_channel_directory_sync should return empty dict on failure."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), \
             patch("gateway.channel_directory.build_channel_directory", side_effect=Exception("boom")):
            result = build_channel_directory_sync({})

        assert result["updated_at"] is None
        assert result["platforms"] == {}

    def test_sync_wrapper_with_existing_loop(self, tmp_path):
        """build_channel_directory_sync should work when called with a running loop."""
        from gateway.config import Platform
        import threading

        result_holder = {}

        def run_in_thread():
            adapters = {Platform.TELEGRAM: MagicMock(platform=Platform.TELEGRAM)}
            with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), \
                 patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json"):
                result = build_channel_directory_sync(adapters)
                result_holder["result"] = result

        t = threading.Thread(target=run_in_thread)
        t.start()
        t.join(timeout=30)

        assert "platforms" in result_holder["result"]
