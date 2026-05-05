"""
Tests for Slack platform adapter.

Covers: app_mention handler, send_document, send_video,
        incoming document handling, message routing.

Note: slack-bolt may not be installed in the test environment.
We mock the slack modules at import time to avoid collection errors.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
)


# ---------------------------------------------------------------------------
# Mock the slack-bolt package if it's not installed
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    """Install mock slack modules so SlackAdapter can be imported."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return  # Real library installed

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

# Patch SLACK_AVAILABLE before importing the adapter
import gateway.platforms.slack as _slack_mod
_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.slack import SlackAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    # Mock the Slack app client
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    # Capture events instead of processing them
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point document cache to tmp_path so tests don't touch ~/.hermes."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )


# ---------------------------------------------------------------------------
# TestAppMentionHandler
# ---------------------------------------------------------------------------

class TestAppMentionHandler:
    """Verify that the app_mention event handler is registered."""

    def test_app_mention_registered_on_connect(self):
        """connect() should register message + assistant lifecycle handlers."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        # Track which events get registered
        registered_events = []
        registered_commands = []

        mock_app = MagicMock()

        def mock_event(event_type):
            def decorator(fn):
                registered_events.append(event_type)
                return fn
            return decorator

        def mock_command(cmd):
            def decorator(fn):
                registered_commands.append(cmd)
                return fn
            return decorator

        mock_app.event = mock_event
        mock_app.command = mock_command
        mock_app.client = AsyncMock()
        mock_app.client.auth_test = AsyncMock(return_value={
            "user_id": "U_BOT",
            "user": "testbot",
        })

        # Mock AsyncWebClient so multi-workspace auth_test is awaitable
        mock_web_client = AsyncMock()
        mock_web_client.auth_test = AsyncMock(return_value={
            "user_id": "U_BOT",
            "user": "testbot",
            "team_id": "T_FAKE",
            "team": "FakeTeam",
        })

        with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
             patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
             patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=MagicMock()), \
             patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
             patch("asyncio.create_task"):
            asyncio.run(adapter.connect())

        assert "message" in registered_events
        assert "app_mention" in registered_events
        assert "assistant_thread_started" in registered_events
        assert "assistant_thread_context_changed" in registered_events
        assert "/hermes" in registered_commands


class TestSlackConnectCleanup:
    """Regression coverage for failed connect() cleanup."""

    @pytest.mark.asyncio
    async def test_releases_platform_lock_when_auth_fails(self):
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        mock_app = MagicMock()
        mock_web_client = AsyncMock()
        mock_web_client.auth_test = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
             patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
             patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=MagicMock()), \
             patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
             patch("gateway.status.release_scoped_lock") as mock_release:
            result = await adapter.connect()

        assert result is False
        mock_release.assert_called_once_with("slack-app-token", "xapp-fake")
        assert adapter._platform_lock_identity is None


# ---------------------------------------------------------------------------
# TestSendDocument
# ---------------------------------------------------------------------------

class TestSendDocument:
    @pytest.mark.asyncio
    async def test_send_document_success(self, adapter, tmp_path):
        test_file = tmp_path / "report.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake content")

        adapter._app.client.files_upload_v2 = AsyncMock(return_value={"ok": True})

        result = await adapter.send_document(
            chat_id="C123",
            file_path=str(test_file),
            caption="Here's the report",
        )

        assert result.success
        adapter._app.client.files_upload_v2.assert_called_once()
        call_kwargs = adapter._app.client.files_upload_v2.call_args[1]
        assert call_kwargs["channel"] == "C123"
        assert call_kwargs["file"] == str(test_file)
        assert call_kwargs["filename"] == "report.pdf"
        assert call_kwargs["initial_comment"] == "Here's the report"

    @pytest.mark.asyncio
    async def test_send_document_custom_name(self, adapter, tmp_path):
        test_file = tmp_path / "data.csv"
        test_file.write_bytes(b"a,b,c\n1,2,3")

        adapter._app.client.files_upload_v2 = AsyncMock(return_value={"ok": True})

        result = await adapter.send_document(
            chat_id="C123",
            file_path=str(test_file),
            file_name="quarterly-report.csv",
        )

        assert result.success
        call_kwargs = adapter._app.client.files_upload_v2.call_args[1]
        assert call_kwargs["filename"] == "quarterly-report.csv"

    @pytest.mark.asyncio
    async def test_send_document_missing_file(self, adapter):
        result = await adapter.send_document(
            chat_id="C123",
            file_path="/nonexistent/file.pdf",
        )

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_document_not_connected(self, adapter):
        adapter._app = None
        result = await adapter.send_document(
            chat_id="C123",
            file_path="/some/file.pdf",
        )

        assert not result.success
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_document_api_error_falls_back(self, adapter, tmp_path):
        test_file = tmp_path / "doc.pdf"
        test_file.write_bytes(b"content")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=RuntimeError("Slack API error")
        )

        # Should fall back to base class (text message)
        result = await adapter.send_document(
            chat_id="C123",
            file_path=str(test_file),
        )

        # Base class send() is also mocked, so check it was attempted
        adapter._app.client.chat_postMessage.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_document_with_thread(self, adapter, tmp_path):
        test_file = tmp_path / "notes.txt"
        test_file.write_bytes(b"some notes")

        adapter._app.client.files_upload_v2 = AsyncMock(return_value={"ok": True})

        result = await adapter.send_document(
            chat_id="C123",
            file_path=str(test_file),
            reply_to="1234567890.123456",
        )

        assert result.success
        call_kwargs = adapter._app.client.files_upload_v2.call_args[1]
        assert call_kwargs["thread_ts"] == "1234567890.123456"


# ---------------------------------------------------------------------------
# TestSendVideo
# ---------------------------------------------------------------------------

class TestSendVideo:
    @pytest.mark.asyncio
    async def test_send_video_success(self, adapter, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video data")

        adapter._app.client.files_upload_v2 = AsyncMock(return_value={"ok": True})

        result = await adapter.send_video(
            chat_id="C123",
            video_path=str(video),
            caption="Check this out",
        )

        assert result.success
        call_kwargs = adapter._app.client.files_upload_v2.call_args[1]
        assert call_kwargs["filename"] == "clip.mp4"
        assert call_kwargs["initial_comment"] == "Check this out"

    @pytest.mark.asyncio
    async def test_send_video_missing_file(self, adapter):
        result = await adapter.send_video(
            chat_id="C123",
            video_path="/nonexistent/video.mp4",
        )

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_video_not_connected(self, adapter):
        adapter._app = None
        result = await adapter.send_video(
            chat_id="C123",
            video_path="/some/video.mp4",
        )

        assert not result.success
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_video_api_error_falls_back(self, adapter, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=RuntimeError("Slack API error")
        )

        # Should fall back to base class (text message)
        result = await adapter.send_video(
            chat_id="C123",
            video_path=str(video),
        )

        adapter._app.client.chat_postMessage.assert_called_once()


# ---------------------------------------------------------------------------
# TestIncomingDocumentHandling
# ---------------------------------------------------------------------------

class TestIncomingDocumentHandling:
    def _make_event(self, files=None, text="hello", channel_type="im"):
        """Build a mock Slack message event with file attachments."""
        return {
            "text": text,
            "user": "U_USER",
            "channel": "C123",
            "channel_type": channel_type,
            "ts": "1234567890.000001",
            "files": files or [],
        }

    @pytest.mark.asyncio
    async def test_pdf_document_cached(self, adapter):
        """A PDF attachment should be downloaded, cached, and set as DOCUMENT type."""
        pdf_bytes = b"%PDF-1.4 fake content"

        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.return_value = pdf_bytes
            event = self._make_event(files=[{
                "mimetype": "application/pdf",
                "name": "report.pdf",
                "url_private_download": "https://files.slack.com/report.pdf",
                "size": len(pdf_bytes),
            }])
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.message_type == MessageType.DOCUMENT
        assert len(msg_event.media_urls) == 1
        assert os.path.exists(msg_event.media_urls[0])
        assert msg_event.media_types == ["application/pdf"]

    @pytest.mark.asyncio
    async def test_txt_document_injects_content(self, adapter):
        """A .txt file under 100KB should have its content injected into event text."""
        content = b"Hello from a text file"

        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.return_value = content
            event = self._make_event(
                text="summarize this",
                files=[{
                    "mimetype": "text/plain",
                    "name": "notes.txt",
                    "url_private_download": "https://files.slack.com/notes.txt",
                    "size": len(content),
                }],
            )
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert "Hello from a text file" in msg_event.text
        assert "[Content of notes.txt]" in msg_event.text
        assert "summarize this" in msg_event.text

    @pytest.mark.asyncio
    async def test_md_document_injects_content(self, adapter):
        """A .md file under 100KB should have its content injected."""
        content = b"# Title\nSome markdown content"

        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.return_value = content
            event = self._make_event(files=[{
                "mimetype": "text/markdown",
                "name": "readme.md",
                "url_private_download": "https://files.slack.com/readme.md",
                "size": len(content),
            }], text="")
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert "# Title" in msg_event.text

    @pytest.mark.asyncio
    async def test_large_txt_not_injected(self, adapter):
        """A .txt file over 100KB should be cached but NOT injected."""
        content = b"x" * (200 * 1024)

        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.return_value = content
            event = self._make_event(files=[{
                "mimetype": "text/plain",
                "name": "big.txt",
                "url_private_download": "https://files.slack.com/big.txt",
                "size": len(content),
            }], text="")
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert len(msg_event.media_urls) == 1
        assert "[Content of" not in (msg_event.text or "")

    @pytest.mark.asyncio
    async def test_zip_file_cached(self, adapter):
        """A .zip file should be cached as a supported document."""
        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.return_value = b"PK\x03\x04zip"
            event = self._make_event(files=[{
                "mimetype": "application/zip",
                "name": "archive.zip",
                "url_private_download": "https://files.slack.com/archive.zip",
                "size": 1024,
            }])
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.message_type == MessageType.DOCUMENT
        assert len(msg_event.media_urls) == 1
        assert msg_event.media_types == ["application/zip"]

    @pytest.mark.asyncio
    async def test_oversized_document_skipped(self, adapter):
        """A document over 20MB should be skipped."""
        event = self._make_event(files=[{
            "mimetype": "application/pdf",
            "name": "huge.pdf",
            "url_private_download": "https://files.slack.com/huge.pdf",
            "size": 25 * 1024 * 1024,
        }])
        await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert len(msg_event.media_urls) == 0

    @pytest.mark.asyncio
    async def test_document_download_error_handled(self, adapter):
        """If document download fails, handler should not crash."""
        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.side_effect = RuntimeError("download failed")
            event = self._make_event(files=[{
                "mimetype": "application/pdf",
                "name": "report.pdf",
                "url_private_download": "https://files.slack.com/report.pdf",
                "size": 1024,
            }])
            await adapter._handle_slack_message(event)

        # Handler should still be called (the exception is caught)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_still_handled(self, adapter):
        """Image attachments should still go through the image path, not document."""
        with patch.object(adapter, "_download_slack_file", new_callable=AsyncMock) as dl:
            dl.return_value = "/tmp/cached_image.jpg"
            event = self._make_event(files=[{
                "mimetype": "image/jpeg",
                "name": "photo.jpg",
                "url_private_download": "https://files.slack.com/photo.jpg",
                "size": 1024,
            }])
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.message_type == MessageType.PHOTO


# ---------------------------------------------------------------------------
# TestMessageRouting
# ---------------------------------------------------------------------------

class TestMessageRouting:
    @pytest.mark.asyncio
    async def test_dm_processed_without_mention(self, adapter):
        """DM messages should be processed without requiring a bot mention."""
        event = {
            "text": "hello",
            "user": "U_USER",
            "channel": "D123",
            "channel_type": "im",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_channel_message_requires_mention(self, adapter):
        """Channel messages without a bot mention should be ignored."""
        event = {
            "text": "just talking",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "channel",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_channel_mention_strips_bot_id(self, adapter):
        """When mentioned in a channel, the bot mention should be stripped."""
        event = {
            "text": "<@U_BOT> what's the weather?",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "channel",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)
        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.text == "what's the weather?"
        assert "<@U_BOT>" not in msg_event.text

    @pytest.mark.asyncio
    async def test_bot_messages_ignored(self, adapter):
        """Messages from bots should be ignored."""
        event = {
            "text": "bot response",
            "bot_id": "B_OTHER",
            "channel": "C123",
            "channel_type": "im",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_message_edits_ignored(self, adapter):
        """Message edits should be ignored."""
        event = {
            "text": "edited message",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "im",
            "ts": "1234567890.000001",
            "subtype": "message_changed",
        }
        await adapter._handle_slack_message(event)
        adapter.handle_message.assert_not_called()


# ---------------------------------------------------------------------------
# TestSendTyping — assistant.threads.setStatus
# ---------------------------------------------------------------------------


class TestSendTyping:
    """Test typing indicator via assistant.threads.setStatus."""

    @pytest.mark.asyncio
    async def test_sets_status_in_thread(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C123", metadata={"thread_id": "parent_ts"})
        adapter._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="is thinking...",
        )

    @pytest.mark.asyncio
    async def test_noop_without_thread(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C123")
        adapter._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_missing_scope_gracefully(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("missing_scope")
        )
        # Should not raise
        await adapter.send_typing("C123", metadata={"thread_id": "ts1"})

    @pytest.mark.asyncio
    async def test_uses_thread_ts_fallback(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C123", metadata={"thread_ts": "fallback_ts"})
        adapter._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="fallback_ts",
            status="is thinking...",
        )

    @pytest.mark.asyncio
    async def test_uses_message_id_fallback_when_no_thread_id(self, adapter):
        """When thread_id is absent but message_id is present, use message_id as thread_ts."""
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C123", metadata={"message_id": "1234567890.123456"})
        adapter._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="1234567890.123456",
            status="is thinking...",
        )

    @pytest.mark.asyncio
    async def test_thread_id_takes_precedence_over_message_id(self, adapter):
        """thread_id should be preferred over message_id when both are present."""
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C123", metadata={"thread_id": "thread_ts", "message_id": "msg_ts"})
        adapter._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="thread_ts",
            status="is thinking...",
        )

    @pytest.mark.asyncio
    async def test_noop_without_thread_id_or_message_id(self, adapter):
        """No API call when neither thread_id nor message_id is available."""
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C123", metadata={"user_id": "U999"})
        adapter._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_with_empty_metadata(self, adapter):
        """No API call with empty metadata dict."""
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C123", metadata={})
        adapter._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_tracks_active_status_thread_with_message_id(self, adapter):
        """_active_status_threads should track the resolved thread_ts even from message_id."""
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C123", metadata={"message_id": "1234567890.123456"})
        assert adapter._active_status_threads["C123"] == "1234567890.123456"

    @pytest.mark.asyncio
    async def test_rate_limited_warning_logs_once_per_chat_id(self, adapter, caplog):
        """WARNING log should appear once per chat_id, then DEBUG on repeats."""
        import logging
        adapter._app.client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("some_error")
        )
        with caplog.at_level(logging.WARNING, logger="gateway.platforms.slack"):
            await adapter.send_typing("C123", metadata={"thread_id": "ts1"})
            await adapter.send_typing("C123", metadata={"thread_id": "ts2"})
        # First call should log WARNING, second should only log DEBUG
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING and "some_error" in r.message]
        assert len(warning_records) == 1

    @pytest.mark.asyncio
    async def test_missing_scope_suppressed_permanently(self, adapter, caplog):
        """missing_scope/invalid_auth errors should be logged once then suppressed."""
        import logging
        adapter._app.client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("missing_scope")
        )
        with caplog.at_level(logging.DEBUG, logger="gateway.platforms.slack"):
            await adapter.send_typing("C456", metadata={"thread_id": "ts1"})
            await adapter.send_typing("C456", metadata={"thread_id": "ts2"})
        # Only one log entry about missing_scope (INFO level)
        scope_records = [r for r in caplog.records if "missing_scope" in r.message]
        assert len(scope_records) == 1

    @pytest.mark.asyncio
    async def test_clears_setStatus_warned_on_success(self, adapter):
        """Successful setStatus should remove chat_id from _setStatus_warned."""
        adapter._setStatus_warned.add("C789")
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C789", metadata={"thread_id": "ts1"})
        assert "C789" not in adapter._setStatus_warned


class TestStopTyping:
    """Test stop_typing clearing the assistant thread status indicator."""

    @pytest.mark.asyncio
    async def test_clears_status_on_stop(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        # First set the indicator
        adapter._active_status_threads["C123"] = "thread_ts_1"
        await adapter.stop_typing("C123")
        adapter._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="thread_ts_1",
            status="",
        )

    @pytest.mark.asyncio
    async def test_noop_when_no_active_status(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.stop_typing("C999")
        adapter._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limited_clear_logging(self, adapter, caplog):
        """stop_typing should rate-limit warning logs like send_typing."""
        import logging
        adapter._app.client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("some_error")
        )
        adapter._active_status_threads["C123"] = "ts1"
        with caplog.at_level(logging.WARNING, logger="gateway.platforms.slack"):
            await adapter.stop_typing("C123")
            # Second call won't have _active_status_threads set, but let's test the warn logic
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING and "some_error" in r.message]
        assert len(warning_records) == 1

    @pytest.mark.asyncio
    async def test_clears_warned_on_successful_clear(self, adapter):
        """Successful clear should remove chat_id from _setStatus_warned."""
        adapter._setStatus_warned.add("C123")
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        adapter._active_status_threads["C123"] = "ts1"
        await adapter.stop_typing("C123")
        assert "C123" not in adapter._setStatus_warned


class TestFormatMessage:
    """Test markdown to Slack mrkdwn conversion."""

    def test_bold_conversion(self, adapter):
        assert adapter.format_message("**hello**") == "*hello*"

    def test_italic_asterisk_conversion(self, adapter):
        assert adapter.format_message("*hello*") == "_hello_"

    def test_italic_underscore_preserved(self, adapter):
        assert adapter.format_message("_hello_") == "_hello_"

    def test_header_to_bold(self, adapter):
        assert adapter.format_message("## Section Title") == "*Section Title*"

    def test_header_with_bold_content(self, adapter):
        # **bold** inside a header should not double-wrap
        assert adapter.format_message("## **Title**") == "*Title*"

    def test_link_conversion(self, adapter):
        result = adapter.format_message("[click here](https://example.com)")
        assert result == "<https://example.com|click here>"

    def test_link_conversion_strips_markdown_angle_brackets(self, adapter):
        result = adapter.format_message("[click here](<https://example.com>)")
        assert result == "<https://example.com|click here>"

    def test_escapes_control_characters(self, adapter):
        result = adapter.format_message("AT&T < 5 > 3")
        assert result == "AT&amp;T &lt; 5 &gt; 3"

    def test_preserves_existing_slack_entities(self, adapter):
        text = "Hey <@U123>, see <https://example.com|example> and <!here>"
        assert adapter.format_message(text) == text

    def test_strikethrough(self, adapter):
        assert adapter.format_message("~~deleted~~") == "~deleted~"

    def test_code_block_preserved(self, adapter):
        code = "```python\nx = **not bold**\n```"
        assert adapter.format_message(code) == code

    def test_inline_code_preserved(self, adapter):
        text = "Use `**raw**` syntax"
        assert adapter.format_message(text) == "Use `**raw**` syntax"

    def test_mixed_content(self, adapter):
        text = "**Bold** and *italic* with `code`"
        result = adapter.format_message(text)
        assert "*Bold*" in result
        assert "_italic_" in result
        assert "`code`" in result

    def test_empty_string(self, adapter):
        assert adapter.format_message("") == ""

    def test_none_passthrough(self, adapter):
        assert adapter.format_message(None) is None

    def test_blockquote_preserved(self, adapter):
        """Single-line blockquote > marker is preserved."""
        assert adapter.format_message("> quoted text") == "> quoted text"

    def test_multiline_blockquote(self, adapter):
        """Multi-line blockquote preserves > on each line."""
        text = "> line one\n> line two"
        assert adapter.format_message(text) == "> line one\n> line two"

    def test_blockquote_with_formatting(self, adapter):
        """Blockquote containing bold text."""
        assert adapter.format_message("> **bold quote**") == "> *bold quote*"

    def test_nested_blockquote(self, adapter):
        """Multiple > characters for nested quotes."""
        assert adapter.format_message(">> deeply quoted") == ">> deeply quoted"

    def test_blockquote_mixed_with_plain(self, adapter):
        """Blockquote lines interleaved with plain text."""
        text = "normal\n> quoted\nnormal again"
        result = adapter.format_message(text)
        assert "> quoted" in result
        assert "normal" in result

    def test_non_prefix_gt_still_escaped(self, adapter):
        """Greater-than in mid-line is still escaped."""
        assert adapter.format_message("5 > 3") == "5 &gt; 3"

    def test_blockquote_with_code(self, adapter):
        """Blockquote containing inline code."""
        result = adapter.format_message("> use `fmt.Println`")
        assert result.startswith(">")
        assert "`fmt.Println`" in result

    def test_bold_italic_combined(self, adapter):
        """Triple-star ***text*** converts to Slack bold+italic *_text_*."""
        assert adapter.format_message("***hello***") == "*_hello_*"

    def test_bold_italic_with_surrounding_text(self, adapter):
        """Bold+italic in a sentence."""
        result = adapter.format_message("This is ***important*** stuff")
        assert "*_important_*" in result

    def test_bold_italic_does_not_break_plain_bold(self, adapter):
        """**bold** still works after adding ***bold italic*** support."""
        assert adapter.format_message("**bold**") == "*bold*"

    def test_bold_italic_does_not_break_plain_italic(self, adapter):
        """*italic* still works after adding ***bold italic*** support."""
        assert adapter.format_message("*italic*") == "_italic_"

    def test_bold_italic_mixed_with_bold(self, adapter):
        """Both ***bold italic*** and **bold** in the same message."""
        result = adapter.format_message("***important*** and **bold**")
        assert "*_important_*" in result
        assert "*bold*" in result

    def test_pre_escaped_ampersand_not_double_escaped(self, adapter):
        """Already-escaped &amp; must not become &amp;amp;."""
        assert adapter.format_message("&amp;") == "&amp;"

    def test_pre_escaped_lt_not_double_escaped(self, adapter):
        """Already-escaped &lt; must not become &amp;lt;."""
        assert adapter.format_message("&lt;") == "&lt;"

    def test_pre_escaped_gt_not_double_escaped(self, adapter):
        """Already-escaped &gt; in plain text must not become &amp;gt;."""
        assert adapter.format_message("5 &gt; 3") == "5 &gt; 3"

    def test_mixed_raw_and_escaped_entities(self, adapter):
        """Raw & and pre-escaped &amp; coexist correctly."""
        result = adapter.format_message("AT&T and &amp; entity")
        assert result == "AT&amp;T and &amp; entity"

    def test_link_with_parentheses_in_url(self, adapter):
        """Wikipedia-style URL with balanced parens is not truncated."""
        result = adapter.format_message("[Foo](https://en.wikipedia.org/wiki/Foo_(bar))")
        assert result == "<https://en.wikipedia.org/wiki/Foo_(bar)|Foo>"

    def test_link_with_multiple_paren_pairs(self, adapter):
        """URL with multiple balanced paren pairs."""
        result = adapter.format_message("[text](https://example.com/a_(b)_c_(d))")
        assert result == "<https://example.com/a_(b)_c_(d)|text>"

    def test_link_without_parens_still_works(self, adapter):
        """Normal URL without parens is unaffected by regex change."""
        result = adapter.format_message("[click](https://example.com/path?q=1)")
        assert result == "<https://example.com/path?q=1|click>"

    def test_link_with_angle_brackets_and_parens(self, adapter):
        """Angle-bracket URL with parens (CommonMark syntax)."""
        result = adapter.format_message("[Foo](<https://en.wikipedia.org/wiki/Foo_(bar)>)")
        assert result == "<https://en.wikipedia.org/wiki/Foo_(bar)|Foo>"

    def test_escaping_is_idempotent(self, adapter):
        """Formatting already-formatted text produces the same result."""
        original = "AT&T < 5 > 3"
        once = adapter.format_message(original)
        twice = adapter.format_message(once)
        assert once == twice

    # --- Entity preservation (spec-compliance) ---

    def test_channel_mention_preserved(self, adapter):
        """<!channel> special mention passes through unchanged."""
        assert adapter.format_message("Attention <!channel>") == "Attention <!channel>"

    def test_everyone_mention_preserved(self, adapter):
        """<!everyone> special mention passes through unchanged."""
        assert adapter.format_message("Hey <!everyone>") == "Hey <!everyone>"

    def test_subteam_mention_preserved(self, adapter):
        """<!subteam^ID> user group mention passes through unchanged."""
        assert adapter.format_message("Paging <!subteam^S12345>") == "Paging <!subteam^S12345>"

    def test_date_formatting_preserved(self, adapter):
        """<!date^...> formatting token passes through unchanged."""
        text = "Posted <!date^1392734382^{date_pretty}|Feb 18, 2014>"
        assert adapter.format_message(text) == text

    def test_channel_link_preserved(self, adapter):
        """<#CHANNEL_ID> channel link passes through unchanged."""
        assert adapter.format_message("Join <#C12345>") == "Join <#C12345>"

    # --- Additional edge cases ---

    def test_message_only_code_block(self, adapter):
        """Entire message is a fenced code block — no conversion."""
        code = "```python\nx = 1\n```"
        assert adapter.format_message(code) == code

    def test_multiline_mixed_formatting(self, adapter):
        """Multi-line message with headers, bold, links, code, and blockquotes."""
        text = "## Title\n**bold** and [link](https://x.com)\n> quote\n`code`"
        result = adapter.format_message(text)
        assert result.startswith("*Title*")
        assert "*bold*" in result
        assert "<https://x.com|link>" in result
        assert "> quote" in result
        assert "`code`" in result

    def test_markdown_unordered_list_with_asterisk(self, adapter):
        """Asterisk list items must not trigger italic conversion."""
        text = "* item one\n* item two"
        result = adapter.format_message(text)
        assert "item one" in result
        assert "item two" in result

    def test_nested_bold_in_link(self, adapter):
        """Bold inside link label — label is stashed before bold pass."""
        result = adapter.format_message("[**bold**](https://example.com)")
        assert "https://example.com" in result
        assert "bold" in result

    def test_url_with_query_string_and_ampersand(self, adapter):
        """Ampersand in URL query string must not be escaped."""
        result = adapter.format_message("[link](https://x.com?a=1&b=2)")
        assert result == "<https://x.com?a=1&b=2|link>"

    def test_emoji_shortcodes_passthrough(self, adapter):
        """Emoji shortcodes like :smile: pass through unchanged."""
        assert adapter.format_message(":smile: hello :wave:") == ":smile: hello :wave:"


# ---------------------------------------------------------------------------
# TestEditMessage
# ---------------------------------------------------------------------------


class TestEditMessage:
    """Verify that edit_message() applies mrkdwn formatting before sending."""

    @pytest.mark.asyncio
    async def test_edit_message_formats_bold(self, adapter):
        """edit_message converts **bold** to Slack *bold*."""
        adapter._app.client.chat_update = AsyncMock(return_value={"ok": True})
        await adapter.edit_message("C123", "1234.5678", "**hello world**")
        kwargs = adapter._app.client.chat_update.call_args.kwargs
        assert kwargs["text"] == "*hello world*"

    @pytest.mark.asyncio
    async def test_edit_message_formats_links(self, adapter):
        """edit_message converts markdown links to Slack format."""
        adapter._app.client.chat_update = AsyncMock(return_value={"ok": True})
        await adapter.edit_message("C123", "1234.5678", "[click](https://example.com)")
        kwargs = adapter._app.client.chat_update.call_args.kwargs
        assert kwargs["text"] == "<https://example.com|click>"

    @pytest.mark.asyncio
    async def test_edit_message_preserves_blockquotes(self, adapter):
        """edit_message preserves blockquote > markers."""
        adapter._app.client.chat_update = AsyncMock(return_value={"ok": True})
        await adapter.edit_message("C123", "1234.5678", "> quoted text")
        kwargs = adapter._app.client.chat_update.call_args.kwargs
        assert kwargs["text"] == "> quoted text"

    @pytest.mark.asyncio
    async def test_edit_message_escapes_control_chars(self, adapter):
        """edit_message escapes & < > in plain text."""
        adapter._app.client.chat_update = AsyncMock(return_value={"ok": True})
        await adapter.edit_message("C123", "1234.5678", "AT&T < 5 > 3")
        kwargs = adapter._app.client.chat_update.call_args.kwargs
        assert kwargs["text"] == "AT&amp;T &lt; 5 &gt; 3"


# ---------------------------------------------------------------------------
# TestEditMessageStreamingPipeline
# ---------------------------------------------------------------------------


class TestEditMessageStreamingPipeline:
    """E2E: verify that sequential streaming edits all go through format_message.

    Simulates the GatewayStreamConsumer pattern where edit_message is called
    repeatedly with progressively longer accumulated text.  Every call must
    produce properly formatted mrkdwn in the chat_update payload.
    """

    @pytest.mark.asyncio
    async def test_edit_message_formats_streaming_updates(self, adapter):
        """Simulates streaming: multiple edits, each should be formatted."""
        adapter._app.client.chat_update = AsyncMock(return_value={"ok": True})

        # First streaming update — bold
        result1 = await adapter.edit_message("C123", "ts1", "**Processing**...")
        assert result1.success is True
        kwargs1 = adapter._app.client.chat_update.call_args.kwargs
        assert kwargs1["text"] == "*Processing*..."

        # Second streaming update — bold + link
        result2 = await adapter.edit_message(
            "C123", "ts1", "**Done!** See [results](https://example.com)"
        )
        assert result2.success is True
        kwargs2 = adapter._app.client.chat_update.call_args.kwargs
        assert kwargs2["text"] == "*Done!* See <https://example.com|results>"

    @pytest.mark.asyncio
    async def test_edit_message_formats_code_and_bold(self, adapter):
        """Streaming update with code block and bold — code must be preserved."""
        adapter._app.client.chat_update = AsyncMock(return_value={"ok": True})

        content = "**Result:**\n```python\nprint('hello')\n```"
        result = await adapter.edit_message("C123", "ts1", content)
        assert result.success is True
        kwargs = adapter._app.client.chat_update.call_args.kwargs
        assert kwargs["text"].startswith("*Result:*")
        assert "```python\nprint('hello')\n```" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_edit_message_formats_blockquote_in_stream(self, adapter):
        """Streaming update with blockquote — '>' marker must survive."""
        adapter._app.client.chat_update = AsyncMock(return_value={"ok": True})

        content = "> **Important:** do this\nnormal line"
        result = await adapter.edit_message("C123", "ts1", content)
        assert result.success is True
        kwargs = adapter._app.client.chat_update.call_args.kwargs
        assert kwargs["text"].startswith("> *Important:*")
        assert "normal line" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_edit_message_formats_progressive_accumulation(self, adapter):
        """Simulate real streaming: text grows with each edit, all formatted."""
        adapter._app.client.chat_update = AsyncMock(return_value={"ok": True})

        updates = [
            ("**Step 1**", "*Step 1*"),
            ("**Step 1**\n**Step 2**", "*Step 1*\n*Step 2*"),
            (
                "**Step 1**\n**Step 2**\nSee [docs](https://docs.example.com)",
                "*Step 1*\n*Step 2*\nSee <https://docs.example.com|docs>",
            ),
        ]

        for raw, expected in updates:
            result = await adapter.edit_message("C123", "ts1", raw)
            assert result.success is True
            kwargs = adapter._app.client.chat_update.call_args.kwargs
            assert kwargs["text"] == expected, f"Failed for input: {raw!r}"

        # Total edit count should match number of updates
        assert adapter._app.client.chat_update.call_count == len(updates)

    @pytest.mark.asyncio
    async def test_edit_message_formats_bold_italic(self, adapter):
        """Bold+italic ***text*** is formatted as *_text_* in edited messages."""
        adapter._app.client.chat_update = AsyncMock(return_value={"ok": True})
        await adapter.edit_message("C123", "ts1", "***important*** update")
        kwargs = adapter._app.client.chat_update.call_args.kwargs
        assert "*_important_*" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_edit_message_does_not_double_escape(self, adapter):
        """Pre-escaped entities in edited messages must not get double-escaped."""
        adapter._app.client.chat_update = AsyncMock(return_value={"ok": True})
        await adapter.edit_message("C123", "ts1", "5 &gt; 3 and &amp; entity")
        kwargs = adapter._app.client.chat_update.call_args.kwargs
        assert "&amp;gt;" not in kwargs["text"]
        assert "&amp;amp;" not in kwargs["text"]
        assert "&gt;" in kwargs["text"]
        assert "&amp;" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_edit_message_formats_url_with_parens(self, adapter):
        """Wikipedia-style URL with parens survives edit pipeline."""
        adapter._app.client.chat_update = AsyncMock(return_value={"ok": True})
        await adapter.edit_message("C123", "ts1", "See [Foo](https://en.wikipedia.org/wiki/Foo_(bar))")
        kwargs = adapter._app.client.chat_update.call_args.kwargs
        assert "<https://en.wikipedia.org/wiki/Foo_(bar)|Foo>" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_edit_message_not_connected(self, adapter):
        """edit_message returns failure when adapter is not connected."""
        adapter._app = None
        result = await adapter.edit_message("C123", "ts1", "**hello**")
        assert result.success is False
        assert "Not connected" in result.error


# ---------------------------------------------------------------------------
# TestReactions
# ---------------------------------------------------------------------------


class TestReactions:
    """Test emoji reaction methods."""

    @pytest.mark.asyncio
    async def test_add_reaction_calls_api(self, adapter):
        adapter._app.client.reactions_add = AsyncMock()
        result = await adapter._add_reaction("C123", "ts1", "eyes")
        assert result is True
        adapter._app.client.reactions_add.assert_called_once_with(
            channel="C123", timestamp="ts1", name="eyes"
        )

    @pytest.mark.asyncio
    async def test_add_reaction_handles_error(self, adapter):
        adapter._app.client.reactions_add = AsyncMock(side_effect=Exception("already_reacted"))
        result = await adapter._add_reaction("C123", "ts1", "eyes")
        assert result is False

    @pytest.mark.asyncio
    async def test_remove_reaction_calls_api(self, adapter):
        adapter._app.client.reactions_remove = AsyncMock()
        result = await adapter._remove_reaction("C123", "ts1", "eyes")
        assert result is True

    @pytest.mark.asyncio
    async def test_reactions_in_message_flow(self, adapter):
        """Reactions should be bracketed around actual processing via hooks."""
        adapter._app.client.reactions_add = AsyncMock()
        adapter._app.client.reactions_remove = AsyncMock()
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Tyler"}}
        })

        event = {
            "text": "hello",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "im",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)

        # _handle_slack_message should register the message for reactions
        assert "1234567890.000001" in adapter._reacting_message_ids

        # Simulate the base class calling on_processing_start
        from gateway.platforms.base import MessageEvent, MessageType, SessionSource
        from gateway.config import Platform
        source = SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="dm",
            user_id="U_USER",
        )
        msg_event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            message_id="1234567890.000001",
        )
        await adapter.on_processing_start(msg_event)

        add_calls = adapter._app.client.reactions_add.call_args_list
        assert len(add_calls) == 1
        assert add_calls[0].kwargs["name"] == "eyes"

        # Simulate the base class calling on_processing_complete
        from gateway.platforms.base import ProcessingOutcome
        await adapter.on_processing_complete(msg_event, ProcessingOutcome.SUCCESS)

        add_calls = adapter._app.client.reactions_add.call_args_list
        remove_calls = adapter._app.client.reactions_remove.call_args_list
        assert len(add_calls) == 2
        assert add_calls[1].kwargs["name"] == "white_check_mark"
        assert len(remove_calls) == 1
        assert remove_calls[0].kwargs["name"] == "eyes"

        # Message ID should be cleaned up
        assert "1234567890.000001" not in adapter._reacting_message_ids

    @pytest.mark.asyncio
    async def test_reactions_failure_outcome(self, adapter):
        """Failed processing should add :x: instead of :white_check_mark:."""
        adapter._app.client.reactions_add = AsyncMock()
        adapter._app.client.reactions_remove = AsyncMock()

        from gateway.platforms.base import MessageEvent, MessageType, SessionSource, ProcessingOutcome
        from gateway.config import Platform
        source = SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="dm",
            user_id="U_USER",
        )
        adapter._reacting_message_ids.add("1234567890.000002")
        msg_event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            message_id="1234567890.000002",
        )
        await adapter.on_processing_complete(msg_event, ProcessingOutcome.FAILURE)

        add_calls = adapter._app.client.reactions_add.call_args_list
        remove_calls = adapter._app.client.reactions_remove.call_args_list
        assert len(add_calls) == 1
        assert add_calls[0].kwargs["name"] == "x"
        assert len(remove_calls) == 1
        assert remove_calls[0].kwargs["name"] == "eyes"

    @pytest.mark.asyncio
    async def test_reactions_skipped_for_non_dm_non_mention(self, adapter):
        """Non-DM, non-mention messages should not get reactions."""
        adapter._app.client.reactions_add = AsyncMock()
        adapter._app.client.reactions_remove = AsyncMock()
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Tyler"}}
        })

        event = {
            "text": "hello",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "channel",
            "ts": "1234567890.000003",
        }
        await adapter._handle_slack_message(event)

        # Should NOT register for reactions when not mentioned in a channel
        assert "1234567890.000003" not in adapter._reacting_message_ids
        adapter._app.client.reactions_add.assert_not_called()
        adapter._app.client.reactions_remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_reactions_disabled_via_env(self, adapter, monkeypatch):
        """SLACK_REACTIONS=false should suppress all reaction lifecycle."""
        monkeypatch.setenv("SLACK_REACTIONS", "false")
        adapter._app.client.reactions_add = AsyncMock()
        adapter._app.client.reactions_remove = AsyncMock()
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Tyler"}}
        })

        event = {
            "text": "hello",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "im",
            "ts": "1234567890.000004",
        }
        await adapter._handle_slack_message(event)

        # Should NOT register for reactions when toggle is off
        assert "1234567890.000004" not in adapter._reacting_message_ids

        # Hooks should also be no-ops when disabled
        from gateway.platforms.base import MessageEvent, MessageType, SessionSource, ProcessingOutcome
        from gateway.config import Platform
        source = SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="dm",
            user_id="U_USER",
        )
        msg_event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            message_id="1234567890.000004",
        )
        # Force-add to verify hooks respect the toggle independently
        adapter._reacting_message_ids.add("1234567890.000004")
        await adapter.on_processing_start(msg_event)
        await adapter.on_processing_complete(msg_event, ProcessingOutcome.SUCCESS)

        adapter._app.client.reactions_add.assert_not_called()
        adapter._app.client.reactions_remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_reactions_enabled_by_default(self, adapter):
        """SLACK_REACTIONS defaults to true (matches existing behavior)."""
        assert adapter._reactions_enabled() is True


# ---------------------------------------------------------------------------
# TestThreadReplyHandling
# ---------------------------------------------------------------------------


class TestThreadReplyHandling:
    """Test thread reply processing without explicit bot mentions."""

    @pytest.fixture()
    def mock_session_store(self):
        """Create a mock session store with entries dict."""
        store = MagicMock()
        store._entries = {}
        store._ensure_loaded = MagicMock()
        store.config = MagicMock()
        store.config.group_sessions_per_user = True
        return store

    @pytest.fixture()
    def adapter_with_session_store(self, mock_session_store):
        """Create an adapter with a mock session store attached."""
        config = PlatformConfig(enabled=True, token="***")
        a = SlackAdapter(config)
        a._app = MagicMock()
        a._app.client = AsyncMock()
        a._bot_user_id = "U_BOT"
        a._team_bot_user_ids = {"T_TEAM": "U_BOT"}
        a._running = True
        a.handle_message = AsyncMock()
        a.set_session_store(mock_session_store)
        return a

    @pytest.mark.asyncio
    async def test_thread_reply_without_mention_no_session_ignored(
        self, adapter_with_session_store, mock_session_store
    ):
        """Thread replies without mention should be ignored if no active session."""
        mock_session_store._entries = {}  # No active sessions

        event = {
            "text": "Just replying in the thread",
            "user": "U_USER",
            "channel": "C123",
            "ts": "123.456",
            "thread_ts": "123.000",  # Different from ts - this is a reply
            "channel_type": "channel",
            "team": "T_TEAM",
        }
        await adapter_with_session_store._handle_slack_message(event)
        adapter_with_session_store.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_thread_reply_without_mention_with_session_processed(
        self, adapter_with_session_store, mock_session_store
    ):
        """Thread replies without mention should be processed if there's an active session."""
        # Simulate an active session for this thread
        session_key = "agent:main:slack:group:C123:123.000:U_USER"
        mock_session_store._entries = {session_key: MagicMock()}

        event = {
            "text": "Follow-up question",
            "user": "U_USER",
            "channel": "C123",
            "ts": "123.456",
            "thread_ts": "123.000",  # Reply in thread 123.000
            "channel_type": "channel",
            "team": "T_TEAM",
        }
        await adapter_with_session_store._handle_slack_message(event)
        adapter_with_session_store.handle_message.assert_called_once()

        # Verify the text is passed through unchanged (no mention stripping needed)
        msg_event = adapter_with_session_store.handle_message.call_args[0][0]
        assert msg_event.text == "Follow-up question"

    @pytest.mark.asyncio
    async def test_thread_reply_with_mention_strips_bot_id(
        self, adapter_with_session_store, mock_session_store
    ):
        """Thread replies with @mention should still strip the bot ID."""
        # Even with a session, mentions should be stripped
        session_key = "agent:main:slack:group:C123:123.000:U_USER"
        mock_session_store._entries = {session_key: MagicMock()}

        event = {
            "text": "<@U_BOT> thanks for the help",
            "user": "U_USER",
            "channel": "C123",
            "ts": "123.456",
            "thread_ts": "123.000",
            "channel_type": "channel",
            "team": "T_TEAM",
        }
        await adapter_with_session_store._handle_slack_message(event)
        adapter_with_session_store.handle_message.assert_called_once()

        msg_event = adapter_with_session_store.handle_message.call_args[0][0]
        assert "<@U_BOT>" not in msg_event.text
        assert msg_event.text == "thanks for the help"

    @pytest.mark.asyncio
    async def test_top_level_message_requires_mention_even_with_session(
        self, adapter_with_session_store, mock_session_store
    ):
        """Top-level channel messages should require mention even if session exists."""
        # Session exists but this is a top-level message (no thread_ts)
        session_key = "agent:main:slack:group:C123:123.000:U_USER"
        mock_session_store._entries = {session_key: MagicMock()}

        event = {
            "text": "New question without mention",
            "user": "U_USER",
            "channel": "C123",
            "ts": "456.789",
            # No thread_ts - this is a top-level message
            "channel_type": "channel",
            "team": "T_TEAM",
        }
        await adapter_with_session_store._handle_slack_message(event)
        adapter_with_session_store.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_session_store_ignores_thread_replies(
        self, adapter
    ):
        """If no session store is attached, thread replies without mention should be ignored."""
        # adapter fixture has no session store attached
        event = {
            "text": "Thread reply without mention",
            "user": "U_USER",
            "channel": "C123",
            "ts": "123.456",
            "thread_ts": "123.000",
            "channel_type": "channel",
            "team": "T_TEAM",
        }
        await adapter._handle_slack_message(event)
        adapter.handle_message.assert_not_called()


# ---------------------------------------------------------------------------
# TestAssistantThreadLifecycle
# ---------------------------------------------------------------------------


class TestAssistantThreadLifecycle:
    """Slack Assistant lifecycle events should seed session/user context."""

    @pytest.fixture()
    def mock_session_store(self):
        store = MagicMock()
        store._entries = {}
        store._ensure_loaded = MagicMock()
        store.config = MagicMock()
        store.config.group_sessions_per_user = True
        store.get_or_create_session = MagicMock()
        return store

    @pytest.fixture()
    def assistant_adapter(self, mock_session_store):
        config = PlatformConfig(enabled=True, token="***")
        a = SlackAdapter(config)
        a._app = MagicMock()
        a._app.client = AsyncMock()
        a._bot_user_id = "U_BOT"
        a._team_bot_user_ids = {"T_TEAM": "U_BOT"}
        a._running = True
        a.handle_message = AsyncMock()
        a.set_session_store(mock_session_store)
        return a

    @pytest.mark.asyncio
    async def test_lifecycle_event_seeds_session_store(self, assistant_adapter, mock_session_store):
        event = {
            "type": "assistant_thread_started",
            "team_id": "T_TEAM",
            "assistant_thread": {
                "channel_id": "D123",
                "thread_ts": "171.000",
                "user_id": "U_USER",
                "context": {"channel_id": "C_ORIGIN"},
            },
        }

        await assistant_adapter._handle_assistant_thread_lifecycle_event(event)

        assert assistant_adapter._assistant_threads[("D123", "171.000")]["user_id"] == "U_USER"
        mock_session_store.get_or_create_session.assert_called_once()
        source = mock_session_store.get_or_create_session.call_args[0][0]
        assert source.chat_id == "D123"
        assert source.chat_type == "dm"
        assert source.user_id == "U_USER"
        assert source.thread_id == "171.000"
        assert source.chat_topic == "C_ORIGIN"

    @pytest.mark.asyncio
    async def test_message_uses_cached_assistant_thread_identity(self, assistant_adapter):
        assistant_adapter._assistant_threads[("D123", "171.000")] = {
            "channel_id": "D123",
            "thread_ts": "171.000",
            "user_id": "U_USER",
            "team_id": "T_TEAM",
        }
        assistant_adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Tyler"}}
        })
        assistant_adapter._app.client.reactions_add = AsyncMock()
        assistant_adapter._app.client.reactions_remove = AsyncMock()

        event = {
            "text": "hello from assistant dm",
            "channel": "D123",
            "channel_type": "im",
            "thread_ts": "171.000",
            "ts": "171.111",
            "team": "T_TEAM",
        }

        await assistant_adapter._handle_slack_message(event)

        msg_event = assistant_adapter.handle_message.call_args[0][0]
        assert msg_event.source.user_id == "U_USER"
        assert msg_event.source.thread_id == "171.000"
        assert msg_event.source.user_name == "Tyler"

    def test_assistant_threads_cache_eviction(self, assistant_adapter):
        """Cache should evict oldest entries when exceeding the size limit."""
        assistant_adapter._ASSISTANT_THREADS_MAX = 10
        # Fill to the limit
        for i in range(10):
            assistant_adapter._cache_assistant_thread_metadata({
                "channel_id": f"D{i}",
                "thread_ts": f"{i}.000",
                "user_id": f"U{i}",
            })
        assert len(assistant_adapter._assistant_threads) == 10

        # Adding one more should trigger eviction (down to max // 2 = 5)
        assistant_adapter._cache_assistant_thread_metadata({
            "channel_id": "D999",
            "thread_ts": "999.000",
            "user_id": "U999",
        })
        assert len(assistant_adapter._assistant_threads) <= 10
        # The newest entry must survive eviction
        assert ("D999", "999.000") in assistant_adapter._assistant_threads


# ---------------------------------------------------------------------------
# TestUserNameResolution
# ---------------------------------------------------------------------------


class TestUserNameResolution:
    """Test user identity resolution."""

    @pytest.mark.asyncio
    async def test_resolves_display_name(self, adapter):
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Tyler", "real_name": "Tyler B"}}
        })
        name = await adapter._resolve_user_name("U123")
        assert name == "Tyler"

    @pytest.mark.asyncio
    async def test_falls_back_to_real_name(self, adapter):
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "", "real_name": "Tyler B"}}
        })
        name = await adapter._resolve_user_name("U123")
        assert name == "Tyler B"

    @pytest.mark.asyncio
    async def test_caches_result(self, adapter):
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Tyler"}}
        })
        await adapter._resolve_user_name("U123")
        await adapter._resolve_user_name("U123")
        # Only one API call despite two lookups
        assert adapter._app.client.users_info.call_count == 1

    @pytest.mark.asyncio
    async def test_handles_api_error(self, adapter):
        adapter._app.client.users_info = AsyncMock(side_effect=Exception("rate limited"))
        name = await adapter._resolve_user_name("U123")
        assert name == "U123"  # Falls back to user_id

    @pytest.mark.asyncio
    async def test_user_name_in_message_source(self, adapter):
        """Message source should include resolved user name."""
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Tyler"}}
        })
        adapter._app.client.reactions_add = AsyncMock()
        adapter._app.client.reactions_remove = AsyncMock()

        event = {
            "text": "hello",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "im",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)

        # Check the source in the MessageEvent passed to handle_message
        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.source.user_name == "Tyler"


# ---------------------------------------------------------------------------
# TestSlashCommands — expanded command set
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Test slash command routing."""

    @pytest.mark.asyncio
    async def test_compact_maps_to_compress(self, adapter):
        command = {"text": "compact", "user_id": "U1", "channel_id": "C1"}
        await adapter._handle_slash_command(command)
        msg = adapter.handle_message.call_args[0][0]
        assert msg.text == "/compress"

    @pytest.mark.asyncio
    async def test_resume_command(self, adapter):
        command = {"text": "resume my session", "user_id": "U1", "channel_id": "C1"}
        await adapter._handle_slash_command(command)
        msg = adapter.handle_message.call_args[0][0]
        assert msg.text == "/resume my session"

    @pytest.mark.asyncio
    async def test_background_command(self, adapter):
        command = {"text": "background run tests", "user_id": "U1", "channel_id": "C1"}
        await adapter._handle_slash_command(command)
        msg = adapter.handle_message.call_args[0][0]
        assert msg.text == "/background run tests"

    @pytest.mark.asyncio
    async def test_usage_command(self, adapter):
        command = {"text": "usage", "user_id": "U1", "channel_id": "C1"}
        await adapter._handle_slash_command(command)
        msg = adapter.handle_message.call_args[0][0]
        assert msg.text == "/usage"

    @pytest.mark.asyncio
    async def test_reasoning_command(self, adapter):
        command = {"text": "reasoning", "user_id": "U1", "channel_id": "C1"}
        await adapter._handle_slash_command(command)
        msg = adapter.handle_message.call_args[0][0]
        assert msg.text == "/reasoning"


# ---------------------------------------------------------------------------
# TestMessageSplitting
# ---------------------------------------------------------------------------


class TestMessageSplitting:
    """Test that long messages are split before sending."""

    @pytest.mark.asyncio
    async def test_long_message_split_into_chunks(self, adapter):
        """Messages over MAX_MESSAGE_LENGTH should be split."""
        long_text = "x" * 45000  # Over Slack's 40k API limit
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "ts1"}
        )
        await adapter.send("C123", long_text)
        # Should have been called multiple times
        assert adapter._app.client.chat_postMessage.call_count >= 2

    @pytest.mark.asyncio
    async def test_short_message_single_send(self, adapter):
        """Short messages should be sent in one call."""
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "ts1"}
        )
        await adapter.send("C123", "hello world")
        assert adapter._app.client.chat_postMessage.call_count == 1

    @pytest.mark.asyncio
    async def test_send_preserves_blockquote_formatting(self, adapter):
        """Blockquote '>' markers must survive format → chunk → send pipeline."""
        adapter._app.client.chat_postMessage = AsyncMock(return_value={"ts": "ts1"})
        await adapter.send("C123", "> quoted text\nnormal text")
        kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        sent_text = kwargs["text"]
        assert sent_text.startswith("> quoted text")
        assert "normal text" in sent_text

    @pytest.mark.asyncio
    async def test_send_formats_bold_italic(self, adapter):
        """Bold+italic ***text*** is formatted as *_text_* in sent messages."""
        adapter._app.client.chat_postMessage = AsyncMock(return_value={"ts": "ts1"})
        await adapter.send("C123", "***important*** update")
        kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert "*_important_*" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_explicitly_enables_mrkdwn(self, adapter):
        adapter._app.client.chat_postMessage = AsyncMock(return_value={"ts": "ts1"})
        await adapter.send("C123", "**hello**")
        kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert kwargs.get("mrkdwn") is True

    @pytest.mark.asyncio
    async def test_send_does_not_double_escape_entities(self, adapter):
        """Pre-escaped &amp; in sent messages must not become &amp;amp;."""
        adapter._app.client.chat_postMessage = AsyncMock(return_value={"ts": "ts1"})
        await adapter.send("C123", "Use &amp; for ampersand")
        kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert "&amp;amp;" not in kwargs["text"]
        assert "&amp;" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_formats_url_with_parens(self, adapter):
        """Wikipedia-style URL with parens survives send pipeline."""
        adapter._app.client.chat_postMessage = AsyncMock(return_value={"ts": "ts1"})
        await adapter.send("C123", "See [Foo](https://en.wikipedia.org/wiki/Foo_(bar))")
        kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert "<https://en.wikipedia.org/wiki/Foo_(bar)|Foo>" in kwargs["text"]


# ---------------------------------------------------------------------------
# TestReplyBroadcast
# ---------------------------------------------------------------------------


class TestReplyBroadcast:
    """Test reply_broadcast config option."""

    @pytest.mark.asyncio
    async def test_broadcast_disabled_by_default(self, adapter):
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "ts1"}
        )
        await adapter.send("C123", "hi", metadata={"thread_id": "parent_ts"})
        kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert "reply_broadcast" not in kwargs

    @pytest.mark.asyncio
    async def test_broadcast_enabled_via_config(self, adapter):
        adapter.config.extra["reply_broadcast"] = True
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "ts1"}
        )
        await adapter.send("C123", "hi", metadata={"thread_id": "parent_ts"})
        kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert kwargs.get("reply_broadcast") is True


# ---------------------------------------------------------------------------
# TestFallbackPreservesThreadContext
# ---------------------------------------------------------------------------

class TestFallbackPreservesThreadContext:
    """Bug fix: file upload fallbacks lost thread context (metadata) when
    calling super() without metadata, causing replies to appear outside
    the thread."""

    @pytest.mark.asyncio
    async def test_send_image_file_fallback_preserves_thread(self, adapter, tmp_path):
        test_file = tmp_path / "photo.jpg"
        test_file.write_bytes(b"\xff\xd8\xff\xe0")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=Exception("upload failed")
        )
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "msg_ts"}
        )

        metadata = {"thread_id": "parent_ts_123"}
        await adapter.send_image_file(
            chat_id="C123",
            image_path=str(test_file),
            caption="test image",
            metadata=metadata,
        )

        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert call_kwargs.get("thread_ts") == "parent_ts_123"

    @pytest.mark.asyncio
    async def test_send_video_fallback_preserves_thread(self, adapter, tmp_path):
        test_file = tmp_path / "clip.mp4"
        test_file.write_bytes(b"\x00\x00\x00\x1c")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=Exception("upload failed")
        )
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "msg_ts"}
        )

        metadata = {"thread_id": "parent_ts_456"}
        await adapter.send_video(
            chat_id="C123",
            video_path=str(test_file),
            metadata=metadata,
        )

        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert call_kwargs.get("thread_ts") == "parent_ts_456"

    @pytest.mark.asyncio
    async def test_send_document_fallback_preserves_thread(self, adapter, tmp_path):
        test_file = tmp_path / "report.pdf"
        test_file.write_bytes(b"%PDF-1.4")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=Exception("upload failed")
        )
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "msg_ts"}
        )

        metadata = {"thread_id": "parent_ts_789"}
        await adapter.send_document(
            chat_id="C123",
            file_path=str(test_file),
            caption="report",
            metadata=metadata,
        )

        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert call_kwargs.get("thread_ts") == "parent_ts_789"

    @pytest.mark.asyncio
    async def test_send_image_file_fallback_includes_caption(self, adapter, tmp_path):
        test_file = tmp_path / "photo.jpg"
        test_file.write_bytes(b"\xff\xd8\xff\xe0")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=Exception("upload failed")
        )
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "msg_ts"}
        )

        await adapter.send_image_file(
            chat_id="C123",
            image_path=str(test_file),
            caption="important screenshot",
        )

        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert "important screenshot" in call_kwargs["text"]


# ---------------------------------------------------------------------------
# TestSendImageSSRFGuards
# ---------------------------------------------------------------------------

class TestSendImageSSRFGuards:
    """send_image should reject redirects that land on private/internal hosts."""

    @pytest.mark.asyncio
    async def test_send_image_blocks_private_redirect_target(self, adapter):
        redirect_response = MagicMock()
        redirect_response.is_redirect = True
        redirect_response.next_request = MagicMock(
            url="http://169.254.169.254/latest/meta-data"
        )

        client_kwargs = {}
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def fake_get(_url):
            for hook in client_kwargs["event_hooks"]["response"]:
                await hook(redirect_response)

        mock_client.get = AsyncMock(side_effect=fake_get)
        adapter._app.client.files_upload_v2 = AsyncMock(return_value={"ok": True})
        adapter._app.client.chat_postMessage = AsyncMock(return_value={"ts": "reply_ts"})

        def fake_async_client(*args, **kwargs):
            client_kwargs.update(kwargs)
            return mock_client

        def fake_is_safe_url(url):
            return url == "https://public.example/image.png"

        with (
            patch("tools.url_safety.is_safe_url", side_effect=fake_is_safe_url),
            patch("httpx.AsyncClient", side_effect=fake_async_client),
        ):
            result = await adapter.send_image(
                chat_id="C123",
                image_url="https://public.example/image.png",
                caption="see this",
            )

        assert result.success
        assert client_kwargs["follow_redirects"] is True
        assert client_kwargs["event_hooks"]["response"]
        adapter._app.client.files_upload_v2.assert_not_awaited()
        adapter._app.client.chat_postMessage.assert_awaited_once()
        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert "see this" in call_kwargs["text"]
        assert "https://public.example/image.png" in call_kwargs["text"]


# ---------------------------------------------------------------------------
# TestProgressMessageThread
# ---------------------------------------------------------------------------

class TestProgressMessageThread:
    """Verify that progress messages go to the correct thread.

    Issue #2954: For Slack DM top-level messages, source.thread_id is None
    but the final reply is threaded under the user's message via reply_to.
    Progress messages must use the same thread anchor (the original message's
    ts) so they appear in the thread instead of the DM root.
    """

    @pytest.mark.asyncio
    async def test_dm_toplevel_progress_uses_message_ts_as_thread(self, adapter):
        """Progress messages for a top-level DM should go into the reply thread."""
        # Simulate a top-level DM: no thread_ts in the event
        event = {
            "channel": "D_DM",
            "channel_type": "im",
            "user": "U_USER",
            "text": "Hello bot",
            "ts": "1234567890.000001",
            # No thread_ts — this is a top-level DM
        }

        captured_events = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: captured_events.append(e))

        # Patch _resolve_user_name to avoid async Slack API call
        with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="testuser")):
            await adapter._handle_slack_message(event)

        assert len(captured_events) == 1
        msg_event = captured_events[0]
        source = msg_event.source

        # With default dm_top_level_threads_as_sessions=True, source.thread_id
        # should equal the message ts so each DM thread gets its own session.
        assert source.thread_id == "1234567890.000001", (
            "source.thread_id must equal the message ts for top-level DMs "
            "so each reply thread gets its own session"
        )

        # The message_id should be the event's ts — this is what the gateway
        # passes as event_message_id so progress messages can thread correctly
        assert msg_event.message_id == "1234567890.000001", (
            "message_id must equal the event ts so _run_agent can use it as "
            "the fallback thread anchor for progress messages"
        )

        # Verify that the Slack send() method correctly threads a message
        # when metadata contains thread_id equal to the original ts
        adapter._app.client.chat_postMessage = AsyncMock(return_value={"ts": "reply_ts"})
        result = await adapter.send(
            chat_id="D_DM",
            content="⚙️ working...",
            metadata={"thread_id": msg_event.message_id},
        )
        assert result.success
        call_kwargs = adapter._app.client.chat_postMessage.call_args[1]
        assert call_kwargs.get("thread_ts") == "1234567890.000001", (
            "send() must pass thread_ts when metadata has thread_id, "
            "ensuring progress messages land in the thread"
        )

    @pytest.mark.asyncio
    async def test_dm_toplevel_shares_session_when_disabled(self, adapter):
        """Opting out restores legacy single-session-per-DM-channel behavior."""
        adapter.config.extra["dm_top_level_threads_as_sessions"] = False

        event = {
            "channel": "D_DM",
            "channel_type": "im",
            "user": "U_USER",
            "text": "Hello bot",
            "ts": "1234567890.000001",
        }

        captured_events = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: captured_events.append(e))

        with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="testuser")):
            await adapter._handle_slack_message(event)

        assert len(captured_events) == 1
        msg_event = captured_events[0]
        source = msg_event.source

        assert source.thread_id is None, (
            "source.thread_id must stay None when "
            "dm_top_level_threads_as_sessions is disabled"
        )

    @pytest.mark.asyncio
    async def test_channel_mention_progress_uses_thread_ts(self, adapter):
        """Progress messages for a channel @mention should go into the reply thread."""
        # Simulate an @mention in a channel: the event ts becomes the thread anchor
        event = {
            "channel": "C_CHAN",
            "channel_type": "channel",
            "user": "U_USER",
            "text": f"<@U_BOT> help me",
            "ts": "2000000000.000001",
            # No thread_ts — top-level channel message
        }

        captured_events = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: captured_events.append(e))

        with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="testuser")):
            await adapter._handle_slack_message(event)

        assert len(captured_events) == 1
        msg_event = captured_events[0]
        source = msg_event.source

        # For channel @mention: thread_id should equal the event ts (fallback)
        assert source.thread_id == "2000000000.000001", (
            "source.thread_id must equal the event ts for channel messages "
            "so each @mention starts its own thread"
        )
        assert msg_event.message_id == "2000000000.000001"

    @pytest.mark.asyncio
    async def test_build_source_propagates_message_id(self, adapter):
        """source.message_id must equal the Slack message ts so run.py can read it
        for _thread_metadata fallback (Bug: build_source() was not passing message_id=ts)."""
        event = {
            "channel": "C_CHAN",
            "channel_type": "channel",
            "user": "U_USER",
            "text": "<@U_BOT> hello",
            "ts": "3000000000.000001",
        }
        captured_events = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: captured_events.append(e))

        with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="testuser")):
            await adapter._handle_slack_message(event)

        assert len(captured_events) == 1
        source = captured_events[0].source
        assert source.message_id == "3000000000.000001", (
            "source.message_id must equal the event ts; "
            "build_source() must propagate message_id=ts"
        )


# ---------------------------------------------------------------------------
# TestNixiSendOnlyMode
# ---------------------------------------------------------------------------

class TestNixiSendOnlyMode:
    """Verify NIXI_MODE send-only behaviour: Socket Mode disabled, send() works."""

    @pytest.mark.asyncio
    async def test_connect_nixi_mode_skips_socket_mode(self):
        """In NIXI_MODE, connect() should init _primary_client, skip AsyncApp/Socket Mode."""
        config = PlatformConfig(enabled=True, token="xoxb-fake-token")
        adapter = SlackAdapter(config)

        mock_web_client = AsyncMock()
        mock_web_client.auth_test = AsyncMock(return_value={
            "team_id": "T_NIXI",
            "user_id": "U_BOT",
            "user": "nixibot",
            "team": "NixiTeam",
        })

        with patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
             patch.dict(os.environ, {"NIXI_MODE": "1", "SLACK_BOT_TOKEN": "xoxb-fake-token"}), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)):
            result = await adapter.connect()

        assert result is True
        assert adapter._primary_client is not None
        assert adapter._app is None  # AsyncApp should NOT be created in NIXI_MODE
        assert adapter._handler is None
        assert "T_NIXI" in adapter._team_clients
        assert adapter._bot_user_id == "U_BOT"

    @pytest.mark.asyncio
    async def test_connect_nixi_mode_multi_workspace(self):
        """In NIXI_MODE with comma-separated tokens, all workspaces should be registered."""
        config = PlatformConfig(enabled=True, token="xoxb-token1,xoxb-token2")
        adapter = SlackAdapter(config)

        call_count = 0
        auth_responses = [
            {"team_id": "T_TEAM1", "user_id": "U_BOT1", "user": "bot1", "team": "Team1"},
            {"team_id": "T_TEAM2", "user_id": "U_BOT2", "user": "bot2", "team": "Team2"},
        ]

        def make_client(token):
            nonlocal call_count
            mock_client = AsyncMock()
            mock_client.auth_test = AsyncMock(return_value=auth_responses[call_count])
            call_count += 1
            return mock_client

        with patch.object(_slack_mod, "AsyncWebClient", side_effect=make_client), \
             patch.dict(os.environ, {"NIXI_MODE": "1"}), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)):
            result = await adapter.connect()

        assert result is True
        assert "T_TEAM1" in adapter._team_clients
        assert "T_TEAM2" in adapter._team_clients
        assert adapter._bot_user_id == "U_BOT1"  # first token is primary

    @pytest.mark.asyncio
    async def test_get_client_fallback_to_primary_in_nixi_mode(self):
        """_get_client() should return _primary_client as fallback when _app is None."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        mock_primary = AsyncMock()
        adapter._primary_client = mock_primary
        adapter._app = None

        # unknown chat_id → no team mapping → should fall back to _primary_client
        client = adapter._get_client("C_UNKNOWN")
        assert client is mock_primary

    def test_get_client_fallback_to_app_client_in_normal_mode(self):
        """_get_client() should fall back to _app.client in normal mode."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        mock_app_client = AsyncMock()
        adapter._app = MagicMock()
        adapter._app.client = mock_app_client
        adapter._primary_client = None

        client = adapter._get_client("C_UNKNOWN")
        assert client is mock_app_client

    @pytest.mark.asyncio
    async def test_send_works_in_nixi_mode(self):
        """send() should work when _app is None but _primary_client exists."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        mock_client = AsyncMock()
        mock_client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1234.5"})
        adapter._primary_client = mock_client
        adapter._app = None
        adapter._running = True

        result = await adapter.send(chat_id="C_TEST", content="Hello from Nixi")
        assert result.success

    @pytest.mark.asyncio
    async def test_send_blocked_when_no_app_and_no_primary(self):
        """send() should fail when both _app and _primary_client are None."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        adapter._app = None
        adapter._primary_client = None

        result = await adapter.send(chat_id="C_TEST", content="Hello")
        assert not result.success
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_disconnect_nixi_mode_skips_handler_cleanup(self):
        """disconnect() in NIXI_MODE should skip socket handler cleanup, release lock."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        adapter._primary_client = AsyncMock()
        adapter._app = None
        adapter._handler = None
        adapter._running = True
        # Set lock identity so _release_platform_lock actually calls release_scoped_lock
        adapter._platform_lock_scope = "slack-bot-token"
        adapter._platform_lock_identity = "xoxb-fake"

        with patch("gateway.status.release_scoped_lock") as mock_release:
            await adapter.disconnect()

        assert adapter._primary_client is None
        mock_release.assert_called_once()

    @pytest.mark.asyncio
    async def test_normal_mode_unchanged(self):
        """Without NIXI_MODE, connect() should use normal AsyncApp/Socket Mode path."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        mock_app = MagicMock()
        mock_web_client = AsyncMock()
        mock_web_client.auth_test = AsyncMock(return_value={
            "team_id": "T_NORMAL",
            "user_id": "U_BOT",
            "user": "normalbot",
            "team": "NormalTeam",
        })
        mock_handler = MagicMock()
        mock_handler.start_async = AsyncMock()

        with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
             patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
             patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=mock_handler), \
             patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}, clear=False), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)):
            # Ensure NIXI_MODE is NOT set
            os.environ.pop("NIXI_MODE", None)
            result = await adapter.connect()

        assert adapter._primary_client is None  # Should stay None in normal mode
        assert adapter._app is not None


# ---------------------------------------------------------------------------
# TestSocketHealthMonitor
# ---------------------------------------------------------------------------

class TestSocketHealthMonitor:
    """Verify Socket Mode connection lifecycle and health monitoring."""

    def test_init_has_socket_connected_and_health_monitor(self):
        """__init__ should initialize _socket_connected=False and _health_monitor_task=None."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        assert adapter._socket_connected is False
        assert adapter._health_monitor_task is None

    @pytest.mark.asyncio
    async def test_connect_registers_on_close_listener(self):
        """connect() should register an on_close listener on the handler's client."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        mock_app = MagicMock()
        mock_web_client = AsyncMock()
        mock_web_client.auth_test = AsyncMock(return_value={
            "team_id": "T_TEST",
            "user_id": "U_BOT",
            "user": "testbot",
            "team": "TestTeam",
        })

        mock_handler_client = MagicMock()
        mock_handler_client.on_close_listeners = []
        mock_handler = MagicMock()
        mock_handler.client = mock_handler_client
        mock_handler.start_async = AsyncMock()

        with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
             patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
             patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=mock_handler), \
             patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}, clear=False), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)):
            os.environ.pop("NIXI_MODE", None)
            result = await adapter.connect()

        assert result is True
        # The on_close listener should have been registered
        assert len(mock_handler_client.on_close_listeners) >= 1

    @pytest.mark.asyncio
    async def test_connect_no_longer_sets_running_true_immediately(self):
        """connect() should NOT set _running=True immediately — _mark_connected() does it via health monitor."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        mock_app = MagicMock()
        mock_web_client = AsyncMock()
        mock_web_client.auth_test = AsyncMock(return_value={
            "team_id": "T_TEST",
            "user_id": "U_BOT",
            "user": "testbot",
            "team": "TestTeam",
        })

        mock_handler_client = MagicMock()
        mock_handler_client.on_close_listeners = []
        mock_handler_client.is_connected = AsyncMock(return_value=False)
        mock_handler = MagicMock()
        mock_handler.client = mock_handler_client
        mock_handler.start_async = AsyncMock()

        with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
             patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
             patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=mock_handler), \
             patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}, clear=False), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)):
            os.environ.pop("NIXI_MODE", None)
            result = await adapter.connect()

        assert result is True
        # _running should be False because _mark_connected() hasn't been called yet
        # (health monitor hasn't detected connection)
        assert adapter._running is False

    @pytest.mark.asyncio
    async def test_connect_creates_health_monitor_task(self):
        """connect() should create a health monitor task."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        mock_app = MagicMock()
        mock_web_client = AsyncMock()
        mock_web_client.auth_test = AsyncMock(return_value={
            "team_id": "T_TEST",
            "user_id": "U_BOT",
            "user": "testbot",
            "team": "TestTeam",
        })

        mock_handler_client = MagicMock()
        mock_handler_client.on_close_listeners = []
        mock_handler = MagicMock()
        mock_handler.client = mock_handler_client
        mock_handler.start_async = AsyncMock()

        with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
             patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
             patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=mock_handler), \
             patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}, clear=False), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)):
            os.environ.pop("NIXI_MODE", None)
            result = await adapter.connect()

        assert result is True
        assert adapter._health_monitor_task is not None
        # Clean up the health monitor task
        adapter._health_monitor_task.cancel()

    @pytest.mark.asyncio
    async def test_on_close_listener_marks_disconnected(self):
        """The on_close listener should set _socket_connected=False and call _mark_disconnected()."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)
        adapter._running = True
        adapter._socket_connected = True

        mock_app = MagicMock()
        mock_web_client = AsyncMock()
        mock_web_client.auth_test = AsyncMock(return_value={
            "team_id": "T_TEST",
            "user_id": "U_BOT",
            "user": "testbot",
            "team": "TestTeam",
        })

        captured_listeners = []
        mock_handler_client = MagicMock()
        mock_handler_client.on_close_listeners = captured_listeners
        mock_handler = MagicMock()
        mock_handler.client = mock_handler_client
        mock_handler.start_async = AsyncMock()

        with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
             patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
             patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=mock_handler), \
             patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}, clear=False), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
             patch("gateway.status.write_runtime_status"):
            os.environ.pop("NIXI_MODE", None)
            result = await adapter.connect()

        assert result is True
        assert len(captured_listeners) == 1

        # Simulate WebSocket close
        adapter._socket_connected = True
        adapter._running = True
        await captured_listeners[0]()

        assert adapter._socket_connected is False
        assert adapter._running is False

    @pytest.mark.asyncio
    async def test_health_monitor_detects_connection(self):
        """Health monitor should call _mark_connected() when is_connected() returns True."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        adapter._handler = MagicMock()
        adapter._handler.client.is_connected = AsyncMock(return_value=True)
        adapter._app = MagicMock()
        adapter._socket_connected = False
        adapter._running = True
        adapter._socket_mode_task = asyncio.create_task(asyncio.sleep(100))  # Not done

        with patch.object(adapter, "_mark_connected") as mock_mark_connected:
            await adapter._socket_health_monitor()
            # _mark_connected should be called because transition from False→True
            mock_mark_connected.assert_called_once()

        adapter._socket_mode_task.cancel()

    @pytest.mark.asyncio
    async def test_health_monitor_detects_disconnection(self):
        """Health monitor should call _mark_disconnected() when is_connected() returns False but we thought connected."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        adapter._handler = MagicMock()
        adapter._handler.client.is_connected = AsyncMock(return_value=False)
        adapter._app = MagicMock()
        adapter._socket_connected = True
        adapter._running = True
        adapter._socket_mode_task = asyncio.create_task(asyncio.sleep(100))  # Not done

        with patch.object(adapter, "_mark_disconnected") as mock_mark_disconnected:
            await adapter._socket_health_monitor()
            # _mark_disconnected should be called because transition from True→False
            mock_mark_disconnected.assert_called_once()

        adapter._socket_mode_task.cancel()

    @pytest.mark.asyncio
    async def test_health_monitor_signals_fatal_error_on_dead_task(self):
        """Health monitor should set fatal error when socket mode task is done but _running is True."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        # Create a task that completes immediately
        async def _quick_done():
            pass
        adapter._socket_mode_task = asyncio.create_task(_quick_done())
        await adapter._socket_mode_task  # Ensure it's done
        adapter._running = True

        with patch.object(adapter, "_set_fatal_error") as mock_fatal:
            await adapter._socket_health_monitor()
            mock_fatal.assert_called_once_with(
                "socket_mode_dead",
                "Socket Mode task terminated",
                retryable=True,
            )

    @pytest.mark.asyncio
    async def test_disconnect_cancels_health_monitor(self):
        """disconnect() should cancel the health monitor task."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        # Simulate a connected adapter
        adapter._handler = MagicMock()
        adapter._handler.close_async = AsyncMock()
        adapter._running = True
        adapter._socket_connected = True
        adapter._health_monitor_task = asyncio.create_task(asyncio.sleep(100))
        adapter._platform_lock_scope = "slack-app-token"
        adapter._platform_lock_identity = "xapp-fake"

        with patch("gateway.status.release_scoped_lock"):
            await adapter.disconnect()

        # Health monitor task should be cancelled and set to None
        assert adapter._health_monitor_task is None
        assert adapter._socket_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_sets_socket_connected_false(self):
        """disconnect() should set _socket_connected=False and _running=False."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        adapter._handler = MagicMock()
        adapter._handler.close_async = AsyncMock()
        adapter._running = True
        adapter._socket_connected = True
        adapter._platform_lock_scope = "slack-app-token"
        adapter._platform_lock_identity = "xapp-fake"

        with patch("gateway.status.release_scoped_lock"):
            await adapter.disconnect()

        assert adapter._socket_connected is False
        assert adapter._running is False
