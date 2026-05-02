"""Tests for nixi.gateway_adapter — NixiAdapter HTTP server, auth, event dispatch, and send delegation."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nixi.intent_classifier import ClassificationResult

# Default "pass" result for mocking the classifier in dispatch tests.
# These tests focus on the adapter's dispatch logic, not the classifier,
# so we bypass classification by always returning "pass".
_CLASSIFY_PASS = ClassificationResult(action="pass", response_text=None, reason="test_pass")

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
        """Valid events should dispatch with employee overlay in channel_prompt."""
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
            assert event.channel_prompt == "# Employee context\nYou are helpful"
            assert event.source.platform == Platform.NIXI
            assert event.source.user_id == "U123"
            assert event.source.user_name == "Alice"
            assert event.message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_dispatch_with_empty_overlay(self, nixi_adapter):
        """Empty overlay should result in channel_prompt=None."""
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
            assert event.channel_prompt is None

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