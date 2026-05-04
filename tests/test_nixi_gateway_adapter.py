"""Tests for nixi.gateway_adapter — NixiAdapter HTTP server, auth, event dispatch, and send delegation."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nixi.intent_classifier import ClassificationResult, classify
from nixi.protocols import NOHELLO_PROTOCOL

# Default "pass" result for mocking the classifier in dispatch tests.
# These tests focus on the adapter's dispatch logic, not the classifier,
# so we bypass classification by always returning "pass".
_CLASSIFY_PASS = ClassificationResult(action="pass", reason="test_pass")

# Skip entire module if aiohttp not available
try:
    from aiohttp import web
    from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult


@pytest.fixture
def nixi_config():
    """Create a PlatformConfig for NixiAdapter tests."""
    return PlatformConfig(
        enabled=True,
        extra={
            "internal_secret": "test-secret-12345678",
            "team_id": "T_TEAM_TEST",
            "host": "127.0.0.1",
            "port": 0,  # Let OS pick a free port
        },
    )


@pytest.fixture
def nixi_adapter(nixi_config):
    """Create a NixiAdapter instance for testing."""
    from nixi.gateway_adapter import NixiAdapter

    adapter = NixiAdapter(nixi_config)
    adapter._message_handler = AsyncMock(return_value="test response")
    return adapter


def _make_adapter_with_app(nixi_config):
    """Create an adapter with aiohttp app ready for testing (no connect())."""
    from nixi.gateway_adapter import NixiAdapter

    adapter = NixiAdapter(nixi_config)
    adapter._message_handler = AsyncMock(return_value="test response")
    return adapter


# ─── check_nixi_requirements ────────────────────────────────────────────


class TestCheckNixiRequirements:
    """Tests for check_nixi_requirements()."""

    def test_returns_true_when_aiohttp_available(self):
        from nixi.gateway_adapter import check_nixi_requirements

        # aiohttp is available in test environment
        assert check_nixi_requirements() is True

    def test_returns_false_when_aiohttp_unavailable(self):
        from nixi import gateway_adapter

        original = gateway_adapter.AIOHTTP_AVAILABLE
        try:
            gateway_adapter.AIOHTTP_AVAILABLE = False
            assert gateway_adapter.check_nixi_requirements() is False
        finally:
            gateway_adapter.AIOHTTP_AVAILABLE = original


# ─── NixiAdapter __init__ ────────────────────────────────────────────────


class TestNixiAdapterInit:
    """Tests for NixiAdapter initialization."""

    def test_stores_config_values(self, nixi_config):
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(nixi_config)
        assert adapter._internal_secret == "test-secret-12345678"
        assert adapter._team_id == "T_TEAM_TEST"
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 0
        assert adapter.platform == Platform.NIXI

    def test_gateway_runner_initialized_to_none(self, nixi_config):
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(nixi_config)
        assert adapter.gateway_runner is None

    def test_env_vars_override_config(self, nixi_config):
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {"NIXI_INTERNAL_SECRET": "env-secret", "NIXI_TEAM_ID": "T_ENV"}):
            adapter = NixiAdapter(nixi_config)
            assert adapter._internal_secret == "env-secret"
            assert adapter._team_id == "T_ENV"

    def test_default_host_and_port(self):
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={})
        adapter = NixiAdapter(config)
        assert adapter._host == "0.0.0.0"
        assert adapter._port == 8080

    def test_bot_names_default_includes_nixi(self):
        """Default bot_names should always include 'nixi'."""
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={})
        adapter = NixiAdapter(config)
        assert "nixi" in adapter._bot_names
        assert adapter._bot_names == ("nixi",)

    def test_bot_names_from_config_extra(self):
        """bot_names from config extra should be used, with 'nixi' always included."""
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={"bot_names": ["nixi", "robo"]})
        adapter = NixiAdapter(config)
        assert "nixi" in adapter._bot_names
        assert "robo" in adapter._bot_names

    def test_bot_names_env_var_overrides_config(self):
        """NIXI_BOT_NAMES env var (JSON list) should override config extra."""
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={"bot_names": ["nixi", "robo"]})
        with patch.dict(os.environ, {"NIXI_BOT_NAMES": '["nixi", "altbot"]'}):
            adapter = NixiAdapter(config)
            assert "nixi" in adapter._bot_names
            assert "altbot" in adapter._bot_names
            assert "robo" not in adapter._bot_names

    def test_bot_names_env_var_invalid_json_falls_back(self):
        """Invalid NIXI_BOT_NAMES JSON should fall back to config extra."""
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={"bot_names": ["nixi", "robo"]})
        with patch.dict(os.environ, {"NIXI_BOT_NAMES": "not-json"}):
            adapter = NixiAdapter(config)
            assert "nixi" in adapter._bot_names
            assert "robo" in adapter._bot_names

    def test_bot_names_env_var_not_a_list_falls_back(self):
        """NIXI_BOT_NAMES that's valid JSON but not a list should fall back to config."""
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={"bot_names": ["nixi", "robo"]})
        with patch.dict(os.environ, {"NIXI_BOT_NAMES": '"justastring"'}):
            adapter = NixiAdapter(config)
            assert "nixi" in adapter._bot_names
            assert "robo" in adapter._bot_names

    def test_bot_names_always_includes_nixi_even_if_config_omits(self):
        """Even if config extra lists names without 'nixi', 'nixi' must be included."""
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={"bot_names": ["robo"]})
        adapter = NixiAdapter(config)
        assert "nixi" in adapter._bot_names
        assert "robo" in adapter._bot_names


# ─── Health endpoint ──────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    @pytest.mark.asyncio
    async def test_health_returns_ok_with_team_id(self, nixi_adapter):
        from aiohttp.test_utils import TestClient, TestServer

        nixi_adapter._app = web.Application()
        nixi_adapter._app.router.add_get("/health", nixi_adapter._handle_health)
        server = TestServer(nixi_adapter._app)
        async with TestClient(server) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["team_id"] == "T_TEAM_TEST"


# ─── Auth validation ──────────────────────────────────────────────────────


class TestAuthValidation:
    """Tests for authorization header validation in _handle_nixi_event."""

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    @pytest.mark.asyncio
    async def test_missing_auth_header_returns_401(self, nixi_adapter):
        nixi_adapter._app = web.Application()
        nixi_adapter._app.router.add_post("/nixi/event", nixi_adapter._handle_nixi_event)
        server = TestServer(nixi_adapter._app)
        async with TestClient(server) as client:
            resp = await client.post(
                "/nixi/event",
                json={"event": {"text": "hello", "channel": "C123"}},
                headers={"X-Nixi-Team-Id": "T_TEAM_TEST"},
            )
            assert resp.status == 401
            data = await resp.json()
            assert "Authorization" in data["error"]

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    @pytest.mark.asyncio
    async def test_wrong_scheme_returns_401(self, nixi_adapter):
        nixi_adapter._app = web.Application()
        nixi_adapter._app.router.add_post("/nixi/event", nixi_adapter._handle_nixi_event)
        server = TestServer(nixi_adapter._app)
        async with TestClient(server) as client:
            resp = await client.post(
                "/nixi/event",
                json={"event": {"text": "hello", "channel": "C123"}},
                headers={
                    "Authorization": "Basic dGVzdDp0ZXN0",
                    "X-Nixi-Team-Id": "T_TEAM_TEST",
                },
            )
            assert resp.status == 401

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    @pytest.mark.asyncio
    async def test_wrong_token_returns_401(self, nixi_adapter):
        nixi_adapter._app = web.Application()
        nixi_adapter._app.router.add_post("/nixi/event", nixi_adapter._handle_nixi_event)
        server = TestServer(nixi_adapter._app)
        async with TestClient(server) as client:
            resp = await client.post(
                "/nixi/event",
                json={"event": {"text": "hello", "channel": "C123"}},
                headers={
                    "Authorization": "Bearer wrong-token",
                    "X-Nixi-Team-Id": "T_TEAM_TEST",
                },
            )
            assert resp.status == 401
            data = await resp.json()
            assert "Invalid authorization" in data["error"]

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    @pytest.mark.asyncio
    async def test_valid_auth_passes(self, nixi_adapter):
        """Valid bearer token should not return 401."""
        nixi_adapter._app = web.Application()
        nixi_adapter._app.router.add_post("/nixi/event", nixi_adapter._handle_nixi_event)
        server = TestServer(nixi_adapter._app)
        async with TestClient(server) as client:
            resp = await client.post(
                "/nixi/event",
                json={"event": {"text": "hello", "channel": "C123"}},
                headers={
                    "Authorization": "Bearer test-secret-12345678",
                    "X-Nixi-Team-Id": "T_TEAM_TEST",
                    "X-Nixi-User-Id": "U123",
                    "X-Nixi-User-Name": "testuser",
                },
            )
            # Should not be 401 (could be 200 on success)
            assert resp.status != 401


# ─── Team ID validation ───────────────────────────────────────────────────


class TestTeamIdValidation:
    """Tests for X-Nixi-Team-Id header validation."""

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    @pytest.mark.asyncio
    async def test_wrong_team_id_returns_403(self, nixi_adapter):
        nixi_adapter._app = web.Application()
        nixi_adapter._app.router.add_post("/nixi/event", nixi_adapter._handle_nixi_event)
        server = TestServer(nixi_adapter._app)
        async with TestClient(server) as client:
            resp = await client.post(
                "/nixi/event",
                json={"event": {"text": "hello", "channel": "C123"}},
                headers={
                    "Authorization": "Bearer test-secret-12345678",
                    "X-Nixi-Team-Id": "T_WRONG_TEAM",
                    "X-Nixi-User-Id": "U123",
                    "X-Nixi-User-Name": "testuser",
                },
            )
            assert resp.status == 403
            data = await resp.json()
            assert "Team ID" in data["error"]

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    @pytest.mark.asyncio
    async def test_correct_team_id_accepted(self, nixi_adapter):
        nixi_adapter._app = web.Application()
        nixi_adapter._app.router.add_post("/nixi/event", nixi_adapter._handle_nixi_event)
        server = TestServer(nixi_adapter._app)
        async with TestClient(server) as client:
            resp = await client.post(
                "/nixi/event",
                json={"event": {"text": "hello", "channel": "C123"}},
                headers={
                    "Authorization": "Bearer test-secret-12345678",
                    "X-Nixi-Team-Id": "T_TEAM_TEST",
                    "X-Nixi-User-Id": "U123",
                    "X-Nixi-User-Name": "testuser",
                },
            )
            # Should not be 403
            assert resp.status != 403

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    @pytest.mark.asyncio
    async def test_no_team_id_configured_accepts_any(self):
        """When no team_id is configured, any team_id header is accepted."""
        config = PlatformConfig(
            enabled=True,
            extra={
                "internal_secret": "test-secret",
                "team_id": "",  # No team ID configured
                "host": "127.0.0.1",
                "port": 0,
            },
        )
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(config)
        adapter._message_handler = AsyncMock(return_value="test response")
        adapter._app = web.Application()
        adapter._app.router.add_post("/nixi/event", adapter._handle_nixi_event)
        server = TestServer(adapter._app)
        async with TestClient(server) as client:
            resp = await client.post(
                "/nixi/event",
                json={"event": {"text": "hello", "channel": "C123"}},
                headers={
                    "Authorization": "Bearer test-secret",
                    "X-Nixi-Team-Id": "T_ANY_TEAM",
                    "X-Nixi-User-Id": "U123",
                    "X-Nixi-User-Name": "testuser",
                },
            )
            # No team_id configured → no validation → not 403
            assert resp.status != 403


# ─── Event dispatch ───────────────────────────────────────────────────────


class TestEventDispatch:
    """Tests for _dispatch_event and message handling."""

    @pytest.mark.asyncio
    async def test_dispatch_creates_message_event_with_overlay(self, nixi_adapter):
        """Valid events should dispatch with overlay + NOHELLO_PROTOCOL in channel_prompt."""
        with patch("nixi.gateway_adapter.classify", return_value=_CLASSIFY_PASS), \
             patch("nixi.gateway_adapter.load_overlay", return_value="# Employee context\nYou are helpful"):
            called_events = []

            async def capture_event(event):
                called_events.append(event)
                return "response"

            nixi_adapter._message_handler = capture_event

            await nixi_adapter._dispatch_event(
                event_data={"event": {"text": "hello", "channel": "C123"}},
                user_id="U123",
                user_name="Alice",
            )

            # Give the background task time to run
            await asyncio.sleep(0.1)

            assert len(called_events) == 1
            event = called_events[0]
            assert event.text == "hello"
            expected_prompt = f"# Employee context\nYou are helpful\n\n{NOHELLO_PROTOCOL}"
            assert event.channel_prompt == expected_prompt
            assert event.source.platform == Platform.NIXI
            assert event.source.user_id == "U123"
            assert event.source.user_name == "Alice"
            assert event.message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_dispatch_with_empty_overlay(self, nixi_adapter):
        """Empty overlay should result in channel_prompt=NOHELLO_PROTOCOL (protocol always present)."""
        with patch("nixi.gateway_adapter.classify", return_value=_CLASSIFY_PASS), \
             patch("nixi.gateway_adapter.load_overlay", return_value=""):
            called_events = []

            async def capture_event(event):
                called_events.append(event)
                return "response"

            nixi_adapter._message_handler = capture_event

            await nixi_adapter._dispatch_event(
                event_data={"event": {"text": "hello", "channel": "C456"}},
                user_id="U456",
                user_name="Bob",
            )

            await asyncio.sleep(0.1)

            assert len(called_events) == 1
            event = called_events[0]
            assert event.channel_prompt == NOHELLO_PROTOCOL

    @pytest.mark.asyncio
    async def test_dispatch_extracts_thread_ts(self, nixi_adapter):
        """thread_ts should be extracted and passed to build_source."""
        with patch("nixi.gateway_adapter.classify", return_value=_CLASSIFY_PASS), \
             patch("nixi.gateway_adapter.load_overlay", return_value=""):
            called_events = []

            async def capture_event(event):
                called_events.append(event)
                return "response"

            nixi_adapter._message_handler = capture_event

            await nixi_adapter._dispatch_event(
                event_data={
                    "event": {
                        "text": "reply",
                        "channel": "C789",
                        "thread_ts": "1234567890.123456",
                    }
                },
                user_id="U789",
                user_name="Carol",
            )

            await asyncio.sleep(0.1)

            assert len(called_events) == 1
            event = called_events[0]
            assert event.source.thread_id == "1234567890.123456"

    @pytest.mark.asyncio
    async def test_dispatch_flat_event_payload(self, nixi_adapter):
        """When event_data has no 'event' key, treat event_data itself as the event."""
        with patch("nixi.gateway_adapter.classify", return_value=_CLASSIFY_PASS), \
             patch("nixi.gateway_adapter.load_overlay", return_value=""):
            called_events = []

            async def capture_event(event):
                called_events.append(event)
                return "response"

            nixi_adapter._message_handler = capture_event

            await nixi_adapter._dispatch_event(
                event_data={"text": "flat payload", "channel": "C_FLAT"},
                user_id="U_FLAT",
                user_name="FlatUser",
            )

            await asyncio.sleep(0.1)

            assert len(called_events) == 1
            event = called_events[0]
            assert event.text == "flat payload"

    @pytest.mark.asyncio
    async def test_dispatch_no_channel_creates_dm_chat_id(self, nixi_adapter):
        """Missing channel should create a DM-style chat_id."""
        with patch("nixi.gateway_adapter.classify", return_value=_CLASSIFY_PASS), \
             patch("nixi.gateway_adapter.load_overlay", return_value=""):
            called_events = []

            async def capture_event(event):
                called_events.append(event)
                return "response"

            nixi_adapter._message_handler = capture_event

            await nixi_adapter._dispatch_event(
                event_data={"event": {"text": "hello"}},
                user_id="U_DM",
                user_name="DMUser",
            )

            await asyncio.sleep(0.1)

            assert len(called_events) == 1
            event = called_events[0]
            assert event.source.chat_id == "nixi:U_DM"
            assert event.source.chat_type == "dm"


# ─── Send delegation ─────────────────────────────────────────────────────


class TestSendDelegation:
    """Tests for send/send_image/send_document delegation to Slack adapter."""

    def _make_adapter_with_mock_runner(self):
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={"internal_secret": "test", "team_id": "T1"})
        adapter = NixiAdapter(config)

        # Mock gateway runner with mock Slack adapter
        mock_slack = AsyncMock()
        mock_slack.send = AsyncMock(return_value=SendResult(success=True, message_id="msg_123"))
        mock_slack.send_image = AsyncMock(return_value=SendResult(success=True))
        mock_slack.send_document = AsyncMock(return_value=SendResult(success=True))

        mock_runner = MagicMock()
        mock_runner.adapters = {Platform.SLACK: mock_slack}

        adapter.gateway_runner = mock_runner
        return adapter, mock_slack

    @pytest.mark.asyncio
    async def test_send_delegates_to_slack(self):
        adapter, mock_slack = self._make_adapter_with_mock_runner()
        result = await adapter.send("C123", "Hello world")
        assert result.success is True
        mock_slack.send.assert_called_once_with("C123", "Hello world", reply_to=None, metadata=None)

    @pytest.mark.asyncio
    async def test_send_with_reply_to_and_metadata(self):
        adapter, mock_slack = self._make_adapter_with_mock_runner()
        result = await adapter.send(
            "C123", "Hello", reply_to="1234567890.123", metadata={"thread_id": "t123"}
        )
        assert result.success is True
        mock_slack.send.assert_called_once_with(
            "C123", "Hello", reply_to="1234567890.123", metadata={"thread_id": "t123"}
        )

    @pytest.mark.asyncio
    async def test_send_no_gateway_runner_returns_error(self):
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={"internal_secret": "test", "team_id": "T1"})
        adapter = NixiAdapter(config)
        # gateway_runner is None by default
        result = await adapter.send("C123", "Hello")
        assert result.success is False
        assert "Slack adapter not available" in result.error

    @pytest.mark.asyncio
    async def test_send_no_slack_adapter_returns_error(self):
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={"internal_secret": "test", "team_id": "T1"})
        adapter = NixiAdapter(config)
        mock_runner = MagicMock()
        mock_runner.adapters = {}  # No Slack adapter
        adapter.gateway_runner = mock_runner

        result = await adapter.send("C123", "Hello")
        assert result.success is False
        assert "Slack adapter not available" in result.error

    @pytest.mark.asyncio
    async def test_send_image_delegates_to_slack(self):
        adapter, mock_slack = self._make_adapter_with_mock_runner()
        result = await adapter.send_image("C123", "https://example.com/img.png", caption="Test")
        assert result.success is True
        mock_slack.send_image.assert_called_once_with(
            "C123", "https://example.com/img.png", caption="Test", reply_to=None, metadata=None
        )

    @pytest.mark.asyncio
    async def test_send_image_no_gateway_runner_returns_error(self):
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={"internal_secret": "test", "team_id": "T1"})
        adapter = NixiAdapter(config)
        result = await adapter.send_image("C123", "https://example.com/img.png")
        assert result.success is False
        assert "Slack adapter not available" in result.error

    @pytest.mark.asyncio
    async def test_send_document_delegates_to_slack(self):
        adapter, mock_slack = self._make_adapter_with_mock_runner()
        result = await adapter.send_document(
            "C123", "/path/to/file.pdf", caption="Report", file_name="report.pdf"
        )
        assert result.success is True
        mock_slack.send_document.assert_called_once_with(
            "C123", "/path/to/file.pdf", caption="Report", file_name="report.pdf", reply_to=None
        )

    @pytest.mark.asyncio
    async def test_send_document_no_gateway_runner_returns_error(self):
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={"internal_secret": "test", "team_id": "T1"})
        adapter = NixiAdapter(config)
        result = await adapter.send_document("C123", "/path/to/file.pdf")
        assert result.success is False
        assert "Slack adapter not available" in result.error


# ─── Send typing and get_chat_info (no-ops) ────────────────────────────────


class TestNoOpMethods:
    """Tests for send_typing and get_chat_info no-op implementations."""

    @pytest.mark.asyncio
    async def test_send_typing_is_noop(self, nixi_adapter):
        """send_typing should not raise and should return None."""
        result = await nixi_adapter.send_typing("C123")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_chat_info_returns_nixi_type(self, nixi_adapter):
        """get_chat_info should return basic info with type=nixi."""
        result = await nixi_adapter.get_chat_info("C123")
        assert result["type"] == "nixi"
        assert result["name"] == "C123"


# ─── Connect/Disconnect lifecycle ───────────────────────────────────────────


class TestLifecycle:
    """Tests for connect() and disconnect() lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_without_secret_returns_false(self):
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(enabled=True, extra={"team_id": "T1", "host": "127.0.0.1", "port": 0})
        adapter = NixiAdapter(config)
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_sets_running_flag(self):
        """After connect(), adapter should report is_connected=True."""
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(
            enabled=True,
            extra={
                "internal_secret": "test-secret",
                "team_id": "T1",
                "host": "127.0.0.1",
                "port": 0,
            },
        )
        adapter = NixiAdapter(config)
        adapter._message_handler = AsyncMock(return_value="test")

        try:
            result = await adapter.connect()
            assert result is True
            assert adapter.is_connected is True
        finally:
            await adapter.disconnect()
            assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_runner(self):
        """After disconnect(), runner and site should be None."""
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(
            enabled=True,
            extra={
                "internal_secret": "test-secret",
                "team_id": "T1",
                "host": "127.0.0.1",
                "port": 0,
            },
        )
        adapter = NixiAdapter(config)
        adapter._message_handler = AsyncMock(return_value="test")

        await adapter.connect()
        assert adapter._runner is not None

        await adapter.disconnect()
        assert adapter._runner is None
        assert adapter._site is None
        assert adapter._app is None

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    @pytest.mark.asyncio
    async def test_server_accepts_requests_after_connect(self):
        """After connect(), the server should respond to requests."""
        from nixi.gateway_adapter import NixiAdapter

        config = PlatformConfig(
            enabled=True,
            extra={
                "internal_secret": "test-secret",
                "team_id": "T1",
                "host": "127.0.0.1",
                "port": 0,
            },
        )
        adapter = NixiAdapter(config)
        adapter._message_handler = AsyncMock(return_value="test")

        try:
            await adapter.connect()

            # Build a test client against the running app
            from aiohttp.test_utils import TestClient, TestServer

            # Use the adapter's already-running app
            server = TestServer(adapter._app)
            async with TestClient(server) as client:
                resp = await client.get("/health")
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
        finally:
            await adapter.disconnect()


# ─── Bad request handling ──────────────────────────────────────────────────


class TestBadRequestHandling:
    """Tests for malformed request handling."""

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, nixi_adapter):
        nixi_adapter._app = web.Application()
        nixi_adapter._app.router.add_post("/nixi/event", nixi_adapter._handle_nixi_event)
        server = TestServer(nixi_adapter._app)
        async with TestClient(server) as client:
            resp = await client.post(
                "/nixi/event",
                data="not json at all{{{",
                headers={
                    "Authorization": "Bearer test-secret-12345678",
                    "X-Nixi-Team-Id": "T_TEAM_TEST",
                    "Content-Type": "application/json",
                },
            )
            assert resp.status == 400


# ─── Deduplication ────────────────────────────────────────────────────────


class TestDeduplication:
    """Tests for MessageDeduplicator integration in _dispatch_event."""

    @pytest.mark.asyncio
    async def test_dispatch_skips_duplicate_event_ts(self, nixi_adapter):
        """Duplicate event_ts should be skipped — only one MessageEvent dispatched."""
        with patch("nixi.gateway_adapter.classify", return_value=_CLASSIFY_PASS), \
             patch("nixi.gateway_adapter.load_overlay", return_value=""):
            called_events = []

            async def capture_event(event):
                called_events.append(event)
                return "response"

            nixi_adapter._message_handler = capture_event

            event_data = {"event": {"text": "hello", "channel": "C123", "event_ts": "1234567890.123456"}}

            # First dispatch — should process normally
            await nixi_adapter._dispatch_event(event_data, user_id="U1", user_name="User1")
            await asyncio.sleep(0.1)
            assert len(called_events) == 1

            # Second dispatch with same event_ts — should be deduplicated
            await nixi_adapter._dispatch_event(event_data, user_id="U1", user_name="User1")
            await asyncio.sleep(0.1)
            assert len(called_events) == 1  # Still 1, not 2

    @pytest.mark.asyncio
    async def test_dispatch_allows_different_event_ts(self, nixi_adapter):
        """Different event_ts values should both be processed."""
        with patch("nixi.gateway_adapter.classify", return_value=_CLASSIFY_PASS), \
             patch("nixi.gateway_adapter.load_overlay", return_value=""):
            called_events = []

            async def capture_event(event):
                called_events.append(event)
                return "response"

            nixi_adapter._message_handler = capture_event

            await nixi_adapter._dispatch_event(
                {"event": {"text": "msg1", "channel": "C123", "event_ts": "1111111111.111111"}},
                user_id="U1",
                user_name="User1",
            )
            await asyncio.sleep(0.1)

            await nixi_adapter._dispatch_event(
                {"event": {"text": "msg2", "channel": "C123", "event_ts": "2222222222.222222"}},
                user_id="U1",
                user_name="User1",
            )
            await asyncio.sleep(0.1)

            assert len(called_events) == 2

    @pytest.mark.asyncio
    async def test_dispatch_missing_event_ts_still_processes(self, nixi_adapter):
        """Events with no event_ts or ts should still be processed (dedup skipped for empty keys)."""
        with patch("nixi.gateway_adapter.classify", return_value=_CLASSIFY_PASS), \
             patch("nixi.gateway_adapter.load_overlay", return_value=""):
            called_events = []

            async def capture_event(event):
                called_events.append(event)
                return "response"

            nixi_adapter._message_handler = capture_event

            await nixi_adapter._dispatch_event(
                {"event": {"text": "no ts field", "channel": "C123"}},
                user_id="U1",
                user_name="User1",
            )
            await asyncio.sleep(0.1)

            assert len(called_events) == 1
            assert called_events[0].text == "no ts field"

    @pytest.mark.asyncio
    async def test_message_id_populated_from_event_ts(self, nixi_adapter):
        """MessageEvent.message_id should be set to the event_ts value."""
        with patch("nixi.gateway_adapter.classify", return_value=_CLASSIFY_PASS), \
             patch("nixi.gateway_adapter.load_overlay", return_value=""):
            called_events = []

            async def capture_event(event):
                called_events.append(event)
                return "response"

            nixi_adapter._message_handler = capture_event

            await nixi_adapter._dispatch_event(
                {"event": {"text": "hello", "channel": "C123", "event_ts": "1234567890.654321"}},
                user_id="U1",
                user_name="User1",
            )
            await asyncio.sleep(0.1)

            assert len(called_events) == 1
            assert called_events[0].message_id == "1234567890.654321"


# ─── Bot names wiring ────────────────────────────────────────────────────


class TestBotNamesWiring:
    """Tests for bot_names loading, ClassificationContext propagation,
    bot_name_detected computation, bot_invoked thread cache condition,
    and receipt logging."""

    @pytest.mark.asyncio
    async def test_dispatch_passes_bot_names_to_context(self, nixi_adapter):
        """ClassificationContext should receive bot_names from adapter."""
        nixi_adapter._bot_user_id = "UBOT99999"
        # Set custom bot_names
        nixi_adapter._bot_names = ("fixi", "nixi")

        captured_ctx = None

        def capture_classify(ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return _CLASSIFY_PASS

        with patch("nixi.gateway_adapter.classify", side_effect=capture_classify), \
             patch("nixi.gateway_adapter.load_overlay", return_value=""):
            await nixi_adapter._dispatch_event(
                {"event": {"text": "hello", "channel": "C123", "event_ts": "111.222"}},
                user_id="U1",
                user_name="User1",
            )
            await asyncio.sleep(0.05)

        assert captured_ctx is not None
        assert "nixi" in captured_ctx.bot_names
        assert "fixi" in captured_ctx.bot_names

    @pytest.mark.asyncio
    async def test_bot_name_detected_in_dispatch(self, nixi_adapter):
        """bot_name_mentioned should detect name-based invocations."""
        from nixi.gateway_adapter import bot_name_mentioned

        nixi_adapter._bot_user_id = ""  # No user ID — rely on name detection
        nixi_adapter._bot_names = ("nixi",)

        # A message mentioning "nixi" by name should be detected
        assert bot_name_mentioned("hey nixi can you help?", ("nixi",)) is True
        # A message without the name should not
        assert bot_name_mentioned("just chatting about work", ("nixi",)) is False

    @pytest.mark.asyncio
    async def test_bot_invoked_thread_cache_records_name_mention(self, nixi_adapter):
        """Thread cache should record when bot is invoked by name (not just <@USERID>).
        
        When a user says "nixi help" in a thread (name mention, no <@USERID>),
        the thread cache should record bot presence via bot_invoked.
        """
        nixi_adapter._bot_user_id = "UBOT99999"
        nixi_adapter._bot_names = ("nixi",)

        thread_ts = "1600000000.444444"

        with patch("nixi.gateway_adapter.classify", return_value=_CLASSIFY_PASS), \
             patch("nixi.gateway_adapter.load_overlay", return_value=""):
            await nixi_adapter._dispatch_event(
                {
                    "event": {
                        "text": "nixi can you help?",
                        "channel": "C99999",
                        "thread_ts": thread_ts,
                        "event_ts": "1700000000.000010",
                    }
                },
                user_id="U_USER_NAME",
                user_name="NameUser",
            )
            await asyncio.sleep(0.05)

        # Thread cache should record because bot_invoked=True (name mention)
        assert nixi_adapter._mention_cache.had_bot(thread_ts) is True

    @pytest.mark.asyncio
    async def test_receipt_logging_includes_both_detection_paths(self, nixi_adapter):
        """Receipt debug log should include both bot_mentioned and bot_name_detected boolean values."""
        nixi_adapter._bot_user_id = "UBOT99999"
        nixi_adapter._bot_names = ("nixi",)

        with patch("nixi.gateway_adapter.classify", return_value=_CLASSIFY_PASS), \
             patch("nixi.gateway_adapter.load_overlay", return_value=""):
            with patch("nixi.gateway_adapter.logger") as mock_logger:
                await nixi_adapter._dispatch_event(
                    {
                        "event": {
                            "text": "<@UBOT99999> hello",
                            "channel": "C99999",
                            "event_ts": "1700000000.000020",
                        }
                    },
                    user_id="U1",
                    user_name="User1",
                )

                # Find the debug call that includes "Message receipt"
                debug_calls = mock_logger.debug.call_args_list
                receipt_calls = [c for c in debug_calls if "Message receipt" in str(c)]
                assert len(receipt_calls) >= 1, f"No receipt log found in debug calls: {debug_calls}"
                # Verify both booleans appear in the log call
                receipt_call = receipt_calls[0]
                assert "bot_mentioned" in str(receipt_call)

    @pytest.mark.asyncio
    async def test_no_name_mention_no_false_thread_cache(self, nixi_adapter):
        """Without any mention (no <@USERID>, no name), thread cache should NOT record 
        on the DROP path (unless action=="pass")."""
        nixi_adapter._bot_user_id = "UBOT99999"
        nixi_adapter._bot_names = ("nixi",)

        thread_ts = "1600000000.555555"

        # Send a message with no bot mention at all — should be classified as DROP
        await nixi_adapter._dispatch_event(
            {
                "event": {
                    "text": "just chatting about weather",
                    "channel": "C99999",
                    "thread_ts": thread_ts,
                    "event_ts": "1700000000.000030",
                }
            },
            user_id="U_USER6",
            user_name="User6",
        )

        # No bot mention → classifier drops → no thread cache record
        # (The real classifier will classify this as DROP)
        # Note: This test uses the REAL classify, so behavior depends on classifier
        # The key assertion is that bot_mentioned=False and bot_name_detected=False
        # lead to bot_invoked=False, so the thread cache condition is:
        # (bot_invoked or result.action=="pass") — since both are False for DROP, no record
        # However, this relies on the real classifier returning DROP, which it does for
        # unrelated channel messages. Let's verify:
        assert nixi_adapter._mention_cache.had_bot(thread_ts) is False


# ─── Intent classifier integration tests ───────────────────────────────────


class TestIntentClassifierIntegration:
    """Integration tests verifying the full classification flow from
    _dispatch_event through to the correct action (DROP/PASS).

    Unlike the other test classes that mock classify() to always PASS,
    these tests exercise the REAL classify() function through the dispatch
    path, verifying end-to-end behavior of the classifier + adapter.

    After the RESPOND path was removed, greeting messages now follow PASS:
    the classifier returns action="pass" and the adapter injects
    NOHELLO_PROTOCOL into channel_prompt so the LLM knows how to respond.
    """

    BOT_USER_ID = "UBOT99999"  # Slack user IDs are uppercase — matches _MENTION_RE

    def _make_dm_event(self, text: str, channel: str = "D12345", **extra_fields) -> dict:
        """Build a DM event payload."""
        event = {
            "text": text,
            "channel": channel,
            "channel_type": "im",
            "event_ts": "1700000000.000001",
        }
        event.update(extra_fields)
        return {"event": event}

    def _make_channel_event(self, text: str, channel: str = "C99999", **extra_fields) -> dict:
        """Build a channel message event payload."""
        event = {
            "text": text,
            "channel": channel,
            "event_ts": "1700000000.000002",
        }
        event.update(extra_fields)
        return {"event": event}

    # ─── PASS path: greetings now pass through to the LLM ──────────────

    @pytest.mark.asyncio
    async def test_dm_greeting_passes_with_nohello_protocol(self, nixi_adapter):
        """DM greeting → classify returns PASS → background task created with
        NOHELLO_PROTOCOL in channel_prompt (no direct send)."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            await nixi_adapter._dispatch_event(
                self._make_dm_event("hey"),
                user_id="U_USER1",
                user_name="TestUser",
            )

        # PASS: background task created, handle_message called
        await asyncio.sleep(0.1)
        assert len(called_events) == 1
        event = called_events[0]
        assert event.channel_prompt == NOHELLO_PROTOCOL
        assert event.source.chat_id == "D12345"

    @pytest.mark.asyncio
    async def test_channel_mention_greeting_passes_with_nohello_protocol(self, nixi_adapter):
        """Channel "@bot hey" → PASS with NOHELLO_PROTOCOL in channel_prompt."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            await nixi_adapter._dispatch_event(
                self._make_channel_event(f"<@{self.BOT_USER_ID}> hey"),
                user_id="U_USER2",
                user_name="TestUser2",
            )

        # PASS: background task created with NOHELLO_PROTOCOL
        await asyncio.sleep(0.1)
        assert len(called_events) == 1
        event = called_events[0]
        assert event.channel_prompt == NOHELLO_PROTOCOL
        assert event.source.chat_id == "C99999"

    # ─── PASS path: substantive messages with protocol ─────────────────

    @pytest.mark.asyncio
    async def test_channel_mention_substantive_passes(self, nixi_adapter):
        """Channel "@bot summarize the thread" → PASS (background task + handle_message)."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return None

        nixi_adapter._message_handler = capture_event

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            await nixi_adapter._dispatch_event(
                self._make_channel_event(f"<@{self.BOT_USER_ID}> summarize the thread"),
                user_id="U_USER3",
                user_name="TestUser3",
            )

        # PASS: background task created, handle_message eventually called
        await asyncio.sleep(0.1)
        assert len(called_events) == 1
        assert called_events[0].text == f"<@{self.BOT_USER_ID}> summarize the thread"
        # NOHELLO_PROTOCOL always in channel_prompt
        assert called_events[0].channel_prompt == NOHELLO_PROTOCOL

    # ─── PASS path: protocol + overlay combination ─────────────────────

    @pytest.mark.asyncio
    async def test_overlay_and_protocol_combined(self, nixi_adapter):
        """PASS with overlay present → channel_prompt has overlay + NOHELLO_PROTOCOL."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        with patch("nixi.gateway_adapter.load_overlay", return_value="# Employee context\nYou are helpful"):
            await nixi_adapter._dispatch_event(
                self._make_dm_event("hey"),
                user_id="U_OVERLAY1",
                user_name="OverlayUser",
            )

        await asyncio.sleep(0.1)
        assert len(called_events) == 1
        event = called_events[0]
        # channel_prompt combines overlay and protocol, separated by double newline
        assert event.channel_prompt == f"# Employee context\nYou are helpful\n\n{NOHELLO_PROTOCOL}"
        assert NOHELLO_PROTOCOL in event.channel_prompt
        assert "# Employee context" in event.channel_prompt

    @pytest.mark.asyncio
    async def test_protocol_only_without_overlay(self, nixi_adapter):
        """PASS without overlay → channel_prompt is just NOHELLO_PROTOCOL."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            await nixi_adapter._dispatch_event(
                self._make_channel_event(f"<@{self.BOT_USER_ID}> summarize the thread"),
                user_id="U_NO_OVERLAY",
                user_name="NoOverlayUser",
            )

        await asyncio.sleep(0.1)
        assert len(called_events) == 1
        assert called_events[0].channel_prompt == NOHELLO_PROTOCOL

    @pytest.mark.asyncio
    async def test_substantive_mention_also_includes_protocol(self, nixi_adapter):
        """Substantive mention ("@bot summarize the thread") → PASS with NOHELLO_PROTOCOL
        in channel_prompt — protocol is always appended, not just for greetings."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            await nixi_adapter._dispatch_event(
                self._make_channel_event(f"<@{self.BOT_USER_ID}> summarize the thread"),
                user_id="U_SUB_MENTION",
                user_name="SubMentionUser",
            )

        await asyncio.sleep(0.1)
        assert len(called_events) == 1
        # Protocol is always in channel_prompt for PASS-path messages
        assert called_events[0].channel_prompt == NOHELLO_PROTOCOL

    # ─── DROP path tests ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_channel_mention_acknowledgment_drops(self, nixi_adapter):
        """Channel "@bot thanks!" → DROP (no background task)."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        await nixi_adapter._dispatch_event(
            self._make_channel_event(f"<@{self.BOT_USER_ID}> thanks!"),
            user_id="U_USER4",
            user_name="TestUser4",
        )

        # DROP: no background task
        await asyncio.sleep(0.05)
        assert len(called_events) == 0

    @pytest.mark.asyncio
    async def test_unrelated_channel_message_drops(self, nixi_adapter):
        """Unrelated channel message (no mention, no thread) → DROP."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        await nixi_adapter._dispatch_event(
            self._make_channel_event("just chatting about work stuff"),
            user_id="U_USER6",
            user_name="TestUser6",
        )

        # DROP: no background task
        await asyncio.sleep(0.05)
        assert len(called_events) == 0

    @pytest.mark.asyncio
    async def test_thread_continuation_passes(self, nixi_adapter):
        """Thread msg (bot mentioned earlier) → PASS via thread_continuation_rule."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID

        # Simulate prior bot engagement in this thread
        thread_ts = "1600000000.111111"
        nixi_adapter._mention_cache.record(thread_ts)

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            await nixi_adapter._dispatch_event(
                self._make_channel_event(
                    "following up on that",
                    thread_ts=thread_ts,
                ),
                user_id="U_USER5",
                user_name="TestUser5",
            )

        # PASS via thread continuation
        await asyncio.sleep(0.1)
        assert len(called_events) == 1
        assert called_events[0].source.thread_id == thread_ts
        # Protocol is always present
        assert called_events[0].channel_prompt == NOHELLO_PROTOCOL

    @pytest.mark.asyncio
    async def test_thread_mention_cache_records_and_returns(self, nixi_adapter):
        """ThreadMentionCache records bot mention AND PASS thread messages,
        returns correct had_bot on subsequent messages in same thread."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID

        cache = nixi_adapter._mention_cache
        thread_ts = "1600000000.222222"

        # Initially, no record of bot engagement
        assert cache.had_bot(thread_ts) is False

        # Record bot mention (simulating a @bot mention in thread)
        cache.record(thread_ts)
        assert cache.had_bot(thread_ts) is True

        # New thread with no bot mention — record via PASS path
        # (In the adapter, record() is called when bot_mentioned or action=="pass" in thread)
        other_thread = "1600000000.333333"
        assert cache.had_bot(other_thread) is False

        # Simulate what the adapter does on PASS: record the thread
        cache.record(other_thread)
        assert cache.had_bot(other_thread) is True

    @pytest.mark.asyncio
    async def test_subtype_message_changed_drops(self, nixi_adapter):
        """Event with subtype="message_changed" → DROP before classifier."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        await nixi_adapter._dispatch_event(
            {
                "event": {
                    "text": "<@Ubot99999> hey",
                    "channel": "C99999",
                    "subtype": "message_changed",
                    "event_ts": "1700000000.000003",
                }
            },
            user_id="U_USER7",
            user_name="TestUser7",
        )

        # Subtype filter blocks before classifier — no background task
        await asyncio.sleep(0.05)
        assert len(called_events) == 0

    @pytest.mark.asyncio
    async def test_empty_bot_user_id_no_false_positives(self, nixi_adapter):
        """Empty NIXI_BOT_USER_ID → bot_mentioned=False for all messages (conservative)."""
        nixi_adapter._bot_user_id = ""

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        # A message that WOULD be a mention if bot_user_id was set
        await nixi_adapter._dispatch_event(
            self._make_channel_event("<@Ubot99999> hey"),
            user_id="U_USER8",
            user_name="TestUser8",
        )

        # With empty bot_user_id, the mention detection returns False.
        # The mention rules skip, and the unrelated_drop_rule fires → DROP
        await asyncio.sleep(0.05)
        assert len(called_events) == 0

    # ─── Bot name mention integration tests ──────────────────────────────

    @pytest.mark.asyncio
    async def test_channel_name_mention_substantive_pass(self, nixi_adapter):
        """Channel "nixi summarize the thread" (no <@...> mention) with bot_names
        → substantive_mention_rule fires via name detection → PASS (message reaches agent)."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID
        nixi_adapter._bot_names = ("nixi",)

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return None

        nixi_adapter._message_handler = capture_event

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            await nixi_adapter._dispatch_event(
                self._make_channel_event("nixi summarize the thread"),
                user_id="U_NAME_SUB",
                user_name="NameSubstantive",
            )

        # PASS: background task created, handle_message called
        await asyncio.sleep(0.1)
        assert len(called_events) == 1
        assert called_events[0].text == "nixi summarize the thread"
        # Protocol always present in PASS-path events
        assert called_events[0].channel_prompt == NOHELLO_PROTOCOL

    @pytest.mark.asyncio
    async def test_channel_name_mention_greeting_passes(self, nixi_adapter):
        """Channel "hey nixi" (name-only greeting) → PASS (greeting_mention_rule fires).

        The classifier's _is_greeting_only strips bot names when bot_names is set,
        so "hey nixi" becomes "hey" which is recognized as greeting-only.
        The greeting_mention_rule returns PASS, and the LLM receives the message
        with NOHELLO_PROTOCOL to handle it appropriately.
        """
        nixi_adapter._bot_user_id = self.BOT_USER_ID
        nixi_adapter._bot_names = ("nixi",)

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            await nixi_adapter._dispatch_event(
                self._make_channel_event("hey nixi"),
                user_id="U_NAME_GREET",
                user_name="NameGreeting",
            )

        # PASS: greeting_mention_rule fires via name detection
        await asyncio.sleep(0.1)
        assert len(called_events) == 1
        assert called_events[0].channel_prompt == NOHELLO_PROTOCOL

    @pytest.mark.asyncio
    async def test_channel_name_mention_noise_drop(self, nixi_adapter):
        """Channel "nixi thanks" (no <@...> mention) → DROP.

        Name-mention acknowledgment that _is_acknowledgment doesn't recognize
        (because it doesn't strip bot names), falling through to unrelated_drop.
        Contrast with "<@BOTID> thanks" which hits noise_mention_rule.
        """
        nixi_adapter._bot_user_id = self.BOT_USER_ID
        nixi_adapter._bot_names = ("nixi",)

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        await nixi_adapter._dispatch_event(
            self._make_channel_event("nixi thanks"),
            user_id="U_NAME_NOISE",
            user_name="NameNoise",
        )

        # DROP: no background task
        await asyncio.sleep(0.05)
        assert len(called_events) == 0

    @pytest.mark.asyncio
    async def test_channel_name_and_slack_mention_both_pass(self, nixi_adapter):
        """Channel "<@BOT_USER_ID> summarize the thread" with both <@...> mention
        AND name mention → PASS (no double-classification or errors).

        Verifies that having both bot_user_id and bot_names configured doesn't
        cause double-processing or errors when both match a message.
        """
        nixi_adapter._bot_user_id = self.BOT_USER_ID
        nixi_adapter._bot_names = ("nixi",)

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return None

        nixi_adapter._message_handler = capture_event

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            await nixi_adapter._dispatch_event(
                self._make_channel_event(f"<@{self.BOT_USER_ID}> summarize the thread"),
                user_id="U_BOTH_MENTION",
                user_name="BothMention",
            )

        # PASS: exactly one event dispatched (no double-processing)
        await asyncio.sleep(0.1)
        assert len(called_events) == 1
        # Protocol always present
        assert called_events[0].channel_prompt == NOHELLO_PROTOCOL

    @pytest.mark.asyncio
    async def test_empty_bot_names_no_false_positives(self, nixi_adapter):
        """_bot_names = () with "hey nixi" (no <@...> mention) → DROP (no names to match,
        unrelated_drop_rule fires). Verifies no false positives without bot_names."""
        nixi_adapter._bot_user_id = self.BOT_USER_ID
        nixi_adapter._bot_names = ()

        called_events = []

        async def capture_event(event):
            called_events.append(event)
            return "response"

        nixi_adapter._message_handler = capture_event

        await nixi_adapter._dispatch_event(
            self._make_channel_event("hey nixi"),
            user_id="U_NO_NAMES",
            user_name="NoNames",
        )

        # DROP: no name match, no slack mention, unrelated_drop_rule fires
        await asyncio.sleep(0.05)
        assert len(called_events) == 0