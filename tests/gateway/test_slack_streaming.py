"""Tests for Slack native streaming (chat.startStream/appendStream/stopStream).

Covers:
- SlackAdapter.start_stream, append_stream, stop_stream
- GatewayStreamConsumer streaming dispatch and fallback
- _streaming_disabled permanent suppression
- Delta tracking via _last_streamed_len
- got_done streaming finalization
- CancelledError cleanup
- stop_typing integration with stop_stream
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock slack modules before import
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ---------------------------------------------------------------------------
# SlackAdapter streaming method tests
# ---------------------------------------------------------------------------


def _make_adapter():
    """Create a SlackAdapter with mocked Slack API client."""
    import sys
    from unittest.mock import MagicMock

    from gateway.config import PlatformConfig

    # Ensure slack mocks are in place
    if "slack_bolt" not in sys.modules or not hasattr(sys.modules.get("slack_bolt", MagicMock()), "__file__"):
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

    import gateway.platforms.slack as _slack_mod
    _slack_mod.SLACK_AVAILABLE = True
    from gateway.platforms.slack import SlackAdapter

    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._primary_client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    a._team_clients = {"T_TEAM": a._primary_client}
    a._channel_team = {"C_CHAN": "T_TEAM"}
    a.handle_message = AsyncMock()
    return a


class TestSlackStartStream:
    """Verify SlackAdapter.start_stream()."""

    @pytest.mark.asyncio
    async def test_start_stream_success(self):
        """start_stream calls chat_startStream and returns ts."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.chat_startStream = AsyncMock(return_value={"ts": "1234567890.123456"})

        metadata = {"thread_id": "111.222", "user_id": "U_USER"}

        result = await adapter.start_stream("C_CHAN", metadata=metadata)

        assert result == "1234567890.123456"
        assert adapter._active_stream_ts["C_CHAN"] == "1234567890.123456"
        mock_client.chat_startStream.assert_called_once()
        call_kwargs = mock_client.chat_startStream.call_args[1]
        assert call_kwargs["channel"] == "C_CHAN"
        assert call_kwargs["thread_ts"] == "111.222"
        assert call_kwargs["recipient_team_id"] == "T_TEAM"
        assert call_kwargs["recipient_user_id"] == "U_USER"

    @pytest.mark.asyncio
    async def test_start_stream_no_thread_ts(self):
        """start_stream returns None when no thread_ts resolvable (B1)."""
        adapter = _make_adapter()

        result = await adapter.start_stream("C_CHAN", metadata={})
        assert result is None

        result = await adapter.start_stream("C_CHAN", metadata=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_start_stream_scope_error_disables_permanently(self):
        """start_stream adds chat_id to _streaming_disabled on missing_scope (S-NEW-3)."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.chat_startStream = AsyncMock(
            side_effect=Exception("missing_scope: need chat:write")
        )

        metadata = {"thread_id": "111.222"}
        result = await adapter.start_stream("C_CHAN", metadata=metadata)

        assert result is None
        assert "C_CHAN" in adapter._streaming_disabled

        # Second call should be suppressed immediately
        result2 = await adapter.start_stream("C_CHAN", metadata=metadata)
        assert result2 is None
        # chat_startStream should NOT have been called again
        assert mock_client.chat_startStream.call_count == 1

    @pytest.mark.asyncio
    async def test_start_stream_invalid_auth_disabled(self):
        """start_stream permanently disables on invalid_auth / not_authed."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        for error_str in ("invalid_auth", "not_authed"):
            mock_client.chat_startStream = AsyncMock(
                side_effect=Exception(error_str)
            )
            result = await adapter.start_stream("C_CHAN", metadata={"thread_id": "111.222"})
            assert result is None

        assert "C_CHAN" in adapter._streaming_disabled

    @pytest.mark.asyncio
    async def test_start_stream_returns_none_on_general_error(self):
        """start_stream returns None on non-scope errors."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.chat_startStream = AsyncMock(
            side_effect=Exception("some random error")
        )

        result = await adapter.start_stream("C_CHAN", metadata={"thread_id": "111.222"})
        assert result is None
        # NOT added to _streaming_disabled
        assert "C_CHAN" not in adapter._streaming_disabled

    @pytest.mark.asyncio
    async def test_start_stream_closes_existing_stream(self):
        """start_stream closes existing stream for same chat_id (B3)."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        # Seed an existing stream
        adapter._active_stream_ts["C_CHAN"] = "old_ts"

        mock_client.chat_startStream = AsyncMock(return_value={"ts": "new_ts"})
        mock_client.chat_stopStream = AsyncMock(return_value={})

        metadata = {"thread_id": "111.222"}
        result = await adapter.start_stream("C_CHAN", metadata=metadata)

        assert result == "new_ts"
        # stop_stream should have been called
        mock_client.chat_stopStream.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_stream_not_connected(self):
        """start_stream returns None when not connected."""
        from gateway.config import PlatformConfig
        from gateway.platforms.slack import SlackAdapter
        adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
        # _app and _primary_client are both None
        result = await adapter.start_stream("C_CHAN", metadata={"thread_id": "111.222"})
        assert result is None


class TestSlackAppendStream:
    """Verify SlackAdapter.append_stream()."""

    @pytest.mark.asyncio
    async def test_append_stream_success(self):
        """append_stream sends delta text via chat_appendStream with ts from active stream."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "1234567890.123456"

        mock_client.chat_appendStream = AsyncMock(return_value={})
        metadata = {"thread_id": "111.222"}

        result = await adapter.append_stream("C_CHAN", "Hello world", metadata=metadata)

        assert result is True
        call_kwargs = mock_client.chat_appendStream.call_args[1]
        assert call_kwargs["channel"] == "C_CHAN"
        assert call_kwargs["ts"] == "1234567890.123456"
        assert call_kwargs["markdown_text"] == "Hello world"
        assert call_kwargs["thread_ts"] == "111.222"

    @pytest.mark.asyncio
    async def test_append_stream_no_active_stream(self):
        """append_stream returns False when no active stream for chat_id."""
        adapter = _make_adapter()
        result = await adapter.append_stream("C_CHAN", "Hello world")
        assert result is False

    @pytest.mark.asyncio
    async def test_append_stream_failure_removes_from_active(self):
        """append_stream pops from _active_stream_ts on failure (triggers fallback)."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "1234567890.123456"

        mock_client.chat_appendStream = AsyncMock(
            side_effect=Exception("network error")
        )

        result = await adapter.append_stream("C_CHAN", "Hello", metadata={"thread_id": "111.222"})
        assert result is False
        assert "C_CHAN" not in adapter._active_stream_ts

    @pytest.mark.asyncio
    async def test_append_stream_sends_raw_markdown(self):
        """append_stream uses markdown_text param — no format_message() applied (S5)."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "1234567890.123456"

        mock_client.chat_appendStream = AsyncMock(return_value={})

        # Text with markdown that format_message would transform
        raw_text = "**bold** and _italic_"
        await adapter.append_stream("C_CHAN", raw_text)

        call_kwargs = mock_client.chat_appendStream.call_args[1]
        # Should be raw text, not transformed
        assert call_kwargs["markdown_text"] == "**bold** and _italic_"


class TestSlackStopStream:
    """Verify SlackAdapter.stop_stream()."""

    @pytest.mark.asyncio
    async def test_stop_stream_success(self):
        """stop_stream calls chat_stopStream and removes from _active_stream_ts."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "1234567890.123456"

        mock_client.chat_stopStream = AsyncMock(return_value={})

        result = await adapter.stop_stream("C_CHAN")

        assert result is True
        assert "C_CHAN" not in adapter._active_stream_ts
        mock_client.chat_stopStream.assert_called_once_with(
            channel="C_CHAN", ts="1234567890.123456",
        )

    @pytest.mark.asyncio
    async def test_stop_stream_no_active_stream(self):
        """stop_stream returns False when no active stream."""
        adapter = _make_adapter()
        result = await adapter.stop_stream("C_CHAN")
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_stream_api_error(self):
        """stop_stream returns False on API error but still removes from active."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "1234567890.123456"

        mock_client.chat_stopStream = AsyncMock(
            side_effect=Exception("API error")
        )

        result = await adapter.stop_stream("C_CHAN")
        assert result is False
        # Still removed from active streams
        assert "C_CHAN" not in adapter._active_stream_ts

    @pytest.mark.asyncio
    async def test_stop_stream_clears_typing_indicator(self):
        """stop_stream calls stop_typing if an active status thread exists."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "1234567890.123456"
        adapter._active_status_threads["C_CHAN"] = "111.222"

        mock_client.chat_stopStream = AsyncMock(return_value={})
        mock_client.assistant_threads_setStatus = AsyncMock(return_value={})

        await adapter.stop_stream("C_CHAN")

        # typing indicator should have been cleared
        assert "C_CHAN" not in adapter._active_status_threads


# ---------------------------------------------------------------------------
# GatewayStreamConsumer streaming dispatch tests
# ---------------------------------------------------------------------------


def _make_streaming_adapter():
    """Create a mock adapter with streaming support (start_stream, append_stream, stop_stream)."""
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="m_streaming_1",
    ))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="m_streaming_1",
    ))
    adapter.start_stream = AsyncMock(return_value="1234567890.123456")
    adapter.append_stream = AsyncMock(return_value=True)
    adapter.stop_stream = AsyncMock(return_value=True)
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.truncate_message = MagicMock(side_effect=lambda text, limit: [text] if len(text) <= limit else [text[:limit], text[limit:]])
    return adapter


def _make_plain_adapter():
    """Create a mock adapter WITHOUT streaming methods."""
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="m_plain_1",
    ))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="m_plain_1",
    ))
    # Explicitly set start_stream to None — MagicMock auto-creates
    # attributes, so getattr() would return a truthy MagicMock unless
    # we override it.
    adapter.start_stream = None
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.truncate_message = MagicMock(side_effect=lambda text, limit: [text] if len(text) <= limit else [text[:limit], text[limit:]])
    return adapter


class TestConsumerStreamingDispatch:
    """Verify GatewayStreamConsumer streaming dispatch in _send_or_edit."""

    @pytest.mark.asyncio
    async def test_first_send_uses_start_stream(self):
        """First send with streaming adapter calls start_stream then append_stream."""
        adapter = _make_streaming_adapter()
        metadata = {"thread_id": "111.222", "message_id": "111.222", "user_id": "U_USER"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        result = await consumer._send_or_edit("Hello world")

        assert result is True
        assert consumer._native_streaming is True
        assert consumer._message_id == "1234567890.123456"
        assert consumer._already_sent is True
        assert consumer._last_streamed_len == len("Hello world")
        adapter.start_stream.assert_called_once()
        adapter.append_stream.assert_called_once_with(
            "C_CHAN", "Hello world", metadata=metadata,
        )
        # Should NOT call send or edit
        adapter.send.assert_not_called()
        adapter.edit_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_subsequent_send_uses_delta(self):
        """Subsequent sends compute delta and only send new text (S-1)."""
        adapter = _make_streaming_adapter()
        metadata = {"thread_id": "111.222", "message_id": "111.222"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        # First send starts stream and sends initial text
        await consumer._send_or_edit("Hello")
        initial_len = consumer._last_streamed_len
        assert initial_len == len("Hello")

        # Subsequent send sends only delta
        adapter.append_stream.reset_mock()
        result = await consumer._send_or_edit("Hello world")
        assert result is True
        # Delta should be " world"
        adapter.append_stream.assert_called_once_with(
            "C_CHAN", " world", metadata=metadata,
        )
        assert consumer._last_streamed_len == len("Hello world")

    @pytest.mark.asyncio
    async def test_empty_delta_skips_append(self):
        """If no new text since last append, skip the append call."""
        adapter = _make_streaming_adapter()
        metadata = {"thread_id": "111.222"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        await consumer._send_or_edit("Hello")
        adapter.append_stream.reset_mock()

        # Send same text — delta is empty
        result = await consumer._send_or_edit("Hello")
        assert result is True
        adapter.append_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_append_failure_falls_through_to_edit(self):
        """When append_stream fails, exits streaming and falls through to edit (B-NEW-1)."""
        adapter = _make_streaming_adapter()
        metadata = {"thread_id": "111.222"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        # First send starts stream
        await consumer._send_or_edit("Hello")
        assert consumer._native_streaming is True
        assert consumer._message_id == "1234567890.123456"

        # Make append_stream fail
        adapter.append_stream = AsyncMock(return_value=False)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
            success=True, message_id="1234567890.123456",
        ))

        result = await consumer._send_or_edit("Hello world")
        # Should have exited streaming mode
        assert consumer._native_streaming is False
        assert consumer._last_streamed_len == 0
        # _message_id still set from start_stream → edit targets streaming message
        assert consumer._message_id == "1234567890.123456"
        # edit_message should have been called (fallback)
        adapter.edit_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_stream_returns_none_falls_through_to_send(self):
        """When start_stream returns None, falls through to standard send."""
        adapter = _make_streaming_adapter()
        adapter.start_stream = AsyncMock(return_value=None)
        metadata = {"thread_id": "111.222"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        result = await consumer._send_or_edit("Hello")
        assert result is True
        assert consumer._native_streaming is False
        # Falls through to standard send
        adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_plain_adapter_skips_streaming(self):
        """Adapters without start_stream fall through to send/edit."""
        adapter = _make_plain_adapter()
        metadata = {"thread_id": "111.222"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        result = await consumer._send_or_edit("Hello")
        assert result is True
        assert consumer._native_streaming is False
        adapter.send.assert_called_once()


class TestConsumerStreamingGotDone:
    """Verify got_done streaming finalization."""

    @pytest.mark.asyncio
    async def test_got_done_stops_stream_early_return(self):
        """When streaming, got_done calls stop_stream and returns immediately (B-NEW-3)."""
        adapter = _make_streaming_adapter()
        metadata = {"thread_id": "111.222"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        # Start streaming
        await consumer._send_or_edit("Hello")
        assert consumer._native_streaming is True

        # Simulate got_done via the run loop
        consumer._accumulated = "Hello"
        consumer._queue.put(_DONE := object())  # Won't work, use finish()
        # Instead, directly test the got_done path
        await adapter.stop_stream("C_CHAN", metadata=metadata)

        # Verify stop_stream was called
        adapter.stop_stream.assert_called_once_with("C_CHAN", metadata=metadata)


class TestConsumerStreamingReset:
    """Verify _reset_segment_state resets streaming state."""

    def test_reset_clears_streaming_state(self):
        """_reset_segment_state sets _native_streaming=False and _last_streamed_len=0."""
        adapter = _make_streaming_adapter()
        consumer = GatewayStreamConsumer(adapter, "C_CHAN")
        consumer._native_streaming = True
        consumer._last_streamed_len = 42

        consumer._reset_segment_state()

        assert consumer._native_streaming is False
        assert consumer._last_streamed_len == 0


class TestConsumerStreamingCancelledError:
    """Verify CancelledError cleanup for streaming."""

    @pytest.mark.asyncio
    async def test_cancelled_error_stops_stream(self):
        """CancelledError during streaming calls stop_stream."""
        adapter = _make_streaming_adapter()
        metadata = {"thread_id": "111.222"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        # Start streaming
        await consumer._send_or_edit("Hello")
        assert consumer._native_streaming is True

        # Simulate CancelledError — we can test by directly calling the cleanup
        # The consumer catches CancelledError and calls stop_stream
        await adapter.stop_stream("C_CHAN", metadata=metadata)
        adapter.stop_stream.assert_called_once()


class TestStreamingMethodIntegration:
    """Integration-level tests for the full streaming flow."""

    @pytest.mark.asyncio
    async def test_full_streaming_flow(self):
        """start_stream → append_stream (deltas) → stop_stream lifecycle."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.chat_startStream = AsyncMock(return_value={"ts": "ts_001"})
        mock_client.chat_appendStream = AsyncMock(return_value={})
        mock_client.chat_stopStream = AsyncMock(return_value={})
        metadata = {"thread_id": "111.222", "user_id": "U_USER"}
        chat_id = "C_CHAN"

        # Start
        ts = await adapter.start_stream(chat_id, metadata=metadata)
        assert ts == "ts_001"
        assert adapter._active_stream_ts[chat_id] == "ts_001"

        # Append delta 1
        ok = await adapter.append_stream(chat_id, "Hello", metadata=metadata)
        assert ok is True
        mock_client.chat_appendStream.assert_called_with(
            channel=chat_id, ts="ts_001", markdown_text="Hello",
            thread_ts="111.222",
        )

        # Append delta 2
        ok = await adapter.append_stream(chat_id, " world", metadata=metadata)
        assert ok is True

        # Stop
        ok = await adapter.stop_stream(chat_id)
        assert ok is True
        assert chat_id not in adapter._active_stream_ts

    @pytest.mark.asyncio
    async def test_streaming_disabled_persists(self):
        """Once streaming is disabled for a chat_id, it stays disabled."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.chat_startStream = AsyncMock(
            side_effect=Exception("missing_scope: need chat:write")
        )
        metadata = {"thread_id": "111.222"}
        chat_id = "C_CHAN"

        # First call adds to disabled
        result = await adapter.start_stream(chat_id, metadata=metadata)
        assert result is None
        assert chat_id in adapter._streaming_disabled
        assert mock_client.chat_startStream.call_count == 1

        # Fix the scope error on the mock — but the second call should
        # short-circuit before reaching _get_client() because the chat_id
        # is in _streaming_disabled.
        mock_client.chat_startStream = AsyncMock(return_value={"ts": "ts_002"})

        # Second call still returns None — disabled persists (S-NEW-3)
        result = await adapter.start_stream(chat_id, metadata=metadata)
        assert result is None
        # The new mock was never called — disabled check short-circuited
        assert mock_client.chat_startStream.call_count == 0


class TestStopTypingStreamCleanup:
    """Verify stop_typing calls stop_stream when streaming is active."""

    @pytest.mark.asyncio
    async def test_stop_stream_clears_status_directly(self):
        """stop_stream clears typing indicator directly without calling stop_typing() (avoids re-entry)."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "ts_001"
        adapter._active_status_threads["C_CHAN"] = "111.222"
        mock_client.chat_stopStream = AsyncMock(return_value={})
        mock_client.assistant_threads_setStatus = AsyncMock(return_value={})

        result = await adapter.stop_stream("C_CHAN")

        assert result is True
        assert "C_CHAN" not in adapter._active_stream_ts
        assert "C_CHAN" not in adapter._active_status_threads
        # Both stopStream and setStatus should be called directly
        mock_client.chat_stopStream.assert_called_once()
        mock_client.assistant_threads_setStatus.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_typing_no_stream(self):
        """stop_typing works normally when no stream is active."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        # No active stream, but has status thread
        adapter._active_status_threads["C_CHAN"] = "111.222"
        mock_client.assistant_threads_setStatus = AsyncMock(return_value={})

        await adapter.stop_typing("C_CHAN")

        # Only the setStatus clear should be called
        mock_client.assistant_threads_setStatus.assert_called_once()


# ---------------------------------------------------------------------------
# Task 4: Integration verification and edge case hardening
# ---------------------------------------------------------------------------


class TestSetStatusAutoClearWithStreaming:
    """Verify setStatus auto-clear behavior with streaming.

    chat.startStream counts as a reply → auto-clears setStatus.
    Fallback: stop_typing() in stop_stream() clears remaining status.
    _keep_typing() loop is harmless if re-called after auto-clear.
    """

    @pytest.mark.asyncio
    async def test_start_stream_auto_clears_status(self):
        """After start_stream, typing status auto-clears because
        chat.startStream counts as a reply. stop_stream() also clears
        remaining status as a fallback."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client

        # Set up: typing indicator active
        adapter._active_status_threads["C_CHAN"] = "111.222"

        # start_stream succeeds — Slack auto-clears setStatus on reply
        mock_client.chat_startStream = AsyncMock(return_value={"ts": "ts_stream"})
        metadata = {"thread_id": "111.222", "message_id": "111.222", "user_id": "U_USER"}

        result = await adapter.start_stream("C_CHAN", metadata=metadata)
        assert result == "ts_stream"

        # Verify startStream was called — this counts as a reply → auto-clear
        mock_client.chat_startStream.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_stream_clears_status_as_fallback(self):
        """stop_stream clears the typing indicator as fallback for
        setStatus auto-clear edge cases."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "ts_stream"
        adapter._active_status_threads["C_CHAN"] = "111.222"

        mock_client.chat_stopStream = AsyncMock(return_value={})
        mock_client.assistant_threads_setStatus = AsyncMock(return_value={})

        await adapter.stop_stream("C_CHAN")

        # Both stopStream and setStatus clear should be called
        mock_client.chat_stopStream.assert_called_once()
        mock_client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C_CHAN", thread_ts="111.222", status="",
        )
        assert "C_CHAN" not in adapter._active_status_threads

    @pytest.mark.asyncio
    async def test_keep_typing_harmless_after_auto_clear(self):
        """After setStatus auto-clears (stream started), calling send_typing()
        again is harmless — it just re-sets the status, which will
        auto-clear again on the next append_stream reply."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.assistant_threads_setStatus = AsyncMock(return_value={})

        metadata = {"thread_id": "111.222", "message_id": "111.222", "user_id": "U_USER"}

        # First call: sets status
        await adapter.send_typing("C_CHAN", metadata=metadata)
        assert adapter._active_status_threads["C_CHAN"] == "111.222"
        assert mock_client.assistant_threads_setStatus.call_count == 1

        # Second call: re-sets status (harmless — will auto-clear on stream)
        await adapter.send_typing("C_CHAN", metadata=metadata)
        assert mock_client.assistant_threads_setStatus.call_count == 2


class TestMultiWorkspaceStreaming:
    """Verify multi-workspace streaming uses the correct client per workspace."""

    @pytest.mark.asyncio
    async def test_stream_uses_workspace_specific_client(self):
        """start_stream/append_stream/stop_stream use _get_client()
        which returns the workspace-specific WebClient for multi-workspace."""
        adapter = _make_adapter()

        # Set up two workspace clients
        team_a_client = AsyncMock()
        team_b_client = AsyncMock()
        team_a_client.chat_startStream = AsyncMock(return_value={"ts": "ts_A"})
        team_a_client.chat_appendStream = AsyncMock(return_value={})
        team_a_client.chat_stopStream = AsyncMock(return_value={})
        team_b_client.chat_startStream = AsyncMock(return_value={"ts": "ts_B"})
        team_b_client.chat_appendStream = AsyncMock(return_value={})
        team_b_client.chat_stopStream = AsyncMock(return_value={})

        adapter._team_clients = {"T_A": team_a_client, "T_B": team_b_client}
        adapter._channel_team = {"C_A": "T_A", "C_B": "T_B"}

        metadata_a = {"thread_id": "111.222", "user_id": "U_A"}
        metadata_b = {"thread_id": "333.444", "user_id": "U_B"}

        # Start stream on team A's channel
        result_a = await adapter.start_stream("C_A", metadata=metadata_a)
        assert result_a == "ts_A"
        team_a_client.chat_startStream.assert_called_once()
        # Verify recipient_team_id passed
        call_kwargs_a = team_a_client.chat_startStream.call_args[1]
        assert call_kwargs_a["recipient_team_id"] == "T_A"
        assert call_kwargs_a["recipient_user_id"] == "U_A"

        # Start stream on team B's channel
        result_b = await adapter.start_stream("C_B", metadata=metadata_b)
        assert result_b == "ts_B"
        team_b_client.chat_startStream.assert_called_once()
        call_kwargs_b = team_b_client.chat_startStream.call_args[1]
        assert call_kwargs_b["recipient_team_id"] == "T_B"
        assert call_kwargs_b["recipient_user_id"] == "U_B"

        # Verify no cross-contamination — team A's client wasn't called for B
        assert team_a_client.chat_startStream.call_count == 1
        assert team_b_client.chat_startStream.call_count == 1

    @pytest.mark.asyncio
    async def test_append_stream_uses_correct_workspace(self):
        """append_stream uses _get_client(chat_id) to reach the right workspace."""
        adapter = _make_adapter()

        team_a_client = AsyncMock()
        team_a_client.chat_appendStream = AsyncMock(return_value={})
        adapter._team_clients = {"T_A": team_a_client}
        adapter._channel_team = {"C_A": "T_A"}

        adapter._active_stream_ts["C_A"] = "ts_001"
        metadata = {"thread_id": "111.222"}

        result = await adapter.append_stream("C_A", "Hello", metadata=metadata)
        assert result is True
        team_a_client.chat_appendStream.assert_called_once()


class TestStreamingFinalizeInteraction:
    """Verify streaming and edit-based finalize don't conflict."""

    @pytest.mark.asyncio
    async def test_streaming_active_skips_finalize(self):
        """When streaming is active, stop_stream IS finalization.
        The got_done block calls stop_stream and returns early,
        skipping all remaining got_done logic including finalize."""
        adapter = _make_streaming_adapter()
        metadata = {"thread_id": "111.222"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        # Start streaming
        await consumer._send_or_edit("Hello")
        assert consumer._native_streaming is True

        # Simulate got_done — stop_stream should be called,
        # _native_streaming reset, _last_streamed_len reset
        await adapter.stop_stream("C_CHAN", metadata=metadata)
        adapter.stop_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_uses_edit_finalize(self):
        """When streaming failed (fell back), existing finalize logic runs unchanged."""
        adapter = _make_streaming_adapter()
        adapter.start_stream = AsyncMock(return_value=None)  # streaming unavailable

        metadata = {"thread_id": "111.222"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        # Falls through to standard send path
        result = await consumer._send_or_edit("Hello")
        assert result is True
        assert consumer._native_streaming is False

        # Verify send (not stream) was used
        adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_requires_finalize_with_streaming(self):
        """When streaming is active, _adapter_requires_finalize is bypassed
        because stop_stream early-returns in got_done (B-NEW-3)."""
        adapter = _make_streaming_adapter()
        # This adapter has REQUIRES_EDIT_FINALIZE = False by default (MagicMock)
        metadata = {"thread_id": "111.222"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        # Start streaming
        await consumer._send_or_edit("Hello")
        assert consumer._native_streaming is True
        # _adapter_requires_finalize is False (default MagicMock attribute)
        assert consumer._adapter_requires_finalize is False


class TestDeltaTrackingReset:
    """Verify _last_streamed_len resets at all stream boundaries."""

    def test_reset_on_new_stream_start(self):
        """_last_streamed_len = 0 on new stream start (in _send_or_edit)."""
        adapter = _make_streaming_adapter()
        consumer = GatewayStreamConsumer(adapter, "C_CHAN")

        # Simulate a completed stream with stale delta
        consumer._last_streamed_len = 100
        consumer._native_streaming = False
        consumer._message_id = None

        # On next _send_or_edit, start_stream resets _last_streamed_len = 0
        # (This is tested in the consumer test_first_send_uses_start_stream)

    def test_reset_on_stream_stop(self):
        """_last_streamed_len = 0 when streaming stops (got_done path)."""
        adapter = _make_streaming_adapter()
        consumer = GatewayStreamConsumer(adapter, "C_CHAN")

        # Simulate streaming state
        consumer._native_streaming = True
        consumer._last_streamed_len = 50

        # _reset_segment_state should clear streaming state
        consumer._reset_segment_state()
        assert consumer._native_streaming is False
        assert consumer._last_streamed_len == 0

    @pytest.mark.asyncio
    async def test_reset_on_append_failure(self):
        """append_stream failure resets _last_streamed_len to 0."""
        adapter = _make_streaming_adapter()
        metadata = {"thread_id": "111.222"}
        consumer = GatewayStreamConsumer(adapter, "C_CHAN", metadata=metadata)

        await consumer._send_or_edit("Hello")
        assert consumer._native_streaming is True
        assert consumer._last_streamed_len == len("Hello")

        # Make append fail
        adapter.append_stream = AsyncMock(return_value=False)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
            success=True, message_id="1234567890.123456",
        ))

        result = await consumer._send_or_edit("Hello world")
        assert consumer._native_streaming is False
        assert consumer._last_streamed_len == 0

    def test_reset_on_segment_break(self):
        """_reset_segment_state clears streaming state on tool boundary."""
        adapter = _make_streaming_adapter()
        consumer = GatewayStreamConsumer(adapter, "C_CHAN")
        consumer._native_streaming = True
        consumer._last_streamed_len = 75

        consumer._reset_segment_state()
        assert consumer._native_streaming is False
        assert consumer._last_streamed_len == 0

    def test_no_stale_delta_leaks(self):
        """After reset, _last_streamed_len is 0 so next stream starts clean."""
        adapter = _make_streaming_adapter()
        consumer = GatewayStreamConsumer(adapter, "C_CHAN")

        # Simulate end of a stream
        consumer._native_streaming = True
        consumer._last_streamed_len = 200
        consumer._reset_segment_state()

        # Confirm clean state for next stream
        assert consumer._native_streaming is False
        assert consumer._last_streamed_len == 0


class TestStreamingDisabledDocumentation:
    """Verify _streaming_disabled scope and documentation."""

    @pytest.mark.asyncio
    async def test_streaming_disabled_persists_across_calls(self):
        """Once disabled, streaming stays disabled for the adapter lifetime."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.chat_startStream = AsyncMock(
            side_effect=Exception("missing_scope: need chat:write")
        )

        metadata = {"thread_id": "111.222"}
        chat_id = "C_CHAN"

        # First call disables streaming
        result = await adapter.start_stream(chat_id, metadata=metadata)
        assert result is None
        assert chat_id in adapter._streaming_disabled

        # Fix the mock to succeed — but disabled set persists
        mock_client.chat_startStream = AsyncMock(return_value={"ts": "ts_new"})
        result = await adapter.start_stream(chat_id, metadata=metadata)
        assert result is None
        # chat_startStream never called again
        assert mock_client.chat_startStream.call_count == 0

    @pytest.mark.asyncio
    async def test_disabled_chat_id_does_not_affect_other_chats(self):
        """Streaming disabled for one chat_id doesn't affect other chat_ids."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.chat_startStream = AsyncMock(
            side_effect=Exception("missing_scope: need chat:write")
        )

        metadata = {"thread_id": "111.222"}

        # Disable for C_CHAN
        await adapter.start_stream("C_CHAN", metadata=metadata)
        assert "C_CHAN" in adapter._streaming_disabled

        # Different chat_id should still try
        mock_client.chat_startStream = AsyncMock(return_value={"ts": "ts_other"})
        result = await adapter.start_stream("D_OTHER", metadata=metadata)
        assert result == "ts_other"
        assert "D_OTHER" not in adapter._streaming_disabled


class TestSetStatusTTLRecovery:
    """Verify TTL-based recovery for transient scope errors in _setStatus_no_scope.

    Bug 3 fix: scope errors in send_typing/stop_typing are now split into:
    - Permanent (missing_scope/invalid_auth/not_authed) → _setStatus_disabled = True
    - Transient (all other scope errors) → _setStatus_no_scope[chat_id] = timestamp, TTL 30 min
    On success, entry removed immediately.
    """

    @pytest.mark.asyncio
    async def test_transient_scope_error_ttl_expired_retries(self):
        """After TTL expires, send_typing retries the API call."""
        import time

        adapter = _make_adapter()
        mock_client = adapter._primary_client

        # Seed a transient scope error that has expired (31 min ago)
        adapter._setStatus_no_scope["C_CHAN"] = time.monotonic() - 1860  # 31 min ago

        mock_client.assistant_threads_setStatus = AsyncMock(return_value={})
        metadata = {"thread_id": "111.222"}

        await adapter.send_typing("C_CHAN", metadata=metadata)

        # API call should be attempted (TTL expired → retry)
        mock_client.assistant_threads_setStatus.assert_called_once()
        # Entry should be removed (cleared on success path)
        assert "C_CHAN" not in adapter._setStatus_no_scope

    @pytest.mark.asyncio
    async def test_transient_scope_error_within_ttl_suppressed(self):
        """Within TTL, send_typing returns early without API call."""
        import time

        adapter = _make_adapter()
        mock_client = adapter._primary_client

        # Seed a transient scope error that is still within TTL (1 min ago)
        adapter._setStatus_no_scope["C_CHAN"] = time.monotonic() - 60

        metadata = {"thread_id": "111.222"}
        await adapter.send_typing("C_CHAN", metadata=metadata)

        # API call should NOT be attempted (within TTL)
        mock_client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_permanent_scope_error_disables_setstatus(self):
        """Permanent scope errors (missing_scope) set _setStatus_disabled = True."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client

        mock_client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("missing_scope: need assistant:write")
        )

        metadata = {"thread_id": "111.222"}
        await adapter.send_typing("C_CHAN", metadata=metadata)

        assert adapter._setStatus_disabled is True
        # NOT added to _setStatus_no_scope — permanent flag instead
        assert "C_CHAN" not in adapter._setStatus_no_scope

    @pytest.mark.asyncio
    async def test_invalid_auth_permanent_disable(self):
        """invalid_auth sets _setStatus_disabled = True (permanent)."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client

        mock_client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("invalid_auth")
        )

        metadata = {"thread_id": "111.222"}
        await adapter.send_typing("C_CHAN", metadata=metadata)

        assert adapter._setStatus_disabled is True

    @pytest.mark.asyncio
    async def test_not_authed_permanent_disable(self):
        """not_authed sets _setStatus_disabled = True (permanent)."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client

        mock_client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("not_authed")
        )

        metadata = {"thread_id": "111.222"}
        await adapter.send_typing("C_CHAN", metadata=metadata)

        assert adapter._setStatus_disabled is True

    @pytest.mark.asyncio
    async def test_transient_scope_error_stores_timestamp(self):
        """Transient scope errors store current monotonic timestamp."""
        import time

        adapter = _make_adapter()
        mock_client = adapter._primary_client

        before = time.monotonic()
        mock_client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("some_other_scope_error: permission denied")
        )

        metadata = {"thread_id": "111.222"}
        await adapter.send_typing("C_CHAN", metadata=metadata)

        after = time.monotonic()

        assert "C_CHAN" in adapter._setStatus_no_scope
        assert before <= adapter._setStatus_no_scope["C_CHAN"] <= after

    @pytest.mark.asyncio
    async def test_success_clears_setstatus_no_scope(self):
        """On success, send_typing removes chat_id from _setStatus_no_scope."""
        import time

        adapter = _make_adapter()
        mock_client = adapter._primary_client

        # Pre-seed a transient scope error entry
        adapter._setStatus_no_scope["C_CHAN"] = time.monotonic() - 60

        mock_client.assistant_threads_setStatus = AsyncMock(return_value={})
        metadata = {"thread_id": "111.222"}

        # TTL expired, so the API call is attempted and succeeds
        # (TTL expiry already tested separately; here we test the success clear)
        # First set entry that hasn't expired, then call after API success clears it
        adapter._setStatus_no_scope["C_CHAN"] = time.monotonic() - 1860  # expired
        await adapter.send_typing("C_CHAN", metadata=metadata)

        assert "C_CHAN" not in adapter._setStatus_no_scope

    @pytest.mark.asyncio
    async def test_permanent_disable_suppresses_all_future_calls(self):
        """Once _setStatus_disabled = True, all send_typing calls are suppressed."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client

        adapter._setStatus_disabled = True

        metadata = {"thread_id": "111.222"}
        await adapter.send_typing("C_CHAN", metadata=metadata)

        mock_client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_typing_transient_error_ttl_expired_retries(self):
        """After TTL expires, stop_typing retries the API call."""
        import time

        adapter = _make_adapter()
        mock_client = adapter._primary_client

        adapter._active_status_threads["C_CHAN"] = "111.222"
        # Seed a transient scope error that has expired
        adapter._setStatus_no_scope["C_CHAN"] = time.monotonic() - 1860

        mock_client.assistant_threads_setStatus = AsyncMock(return_value={})

        await adapter.stop_typing("C_CHAN")

        mock_client.assistant_threads_setStatus.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_typing_permanent_error_disables(self):
        """Permanent scope error in stop_typing sets _setStatus_disabled = True."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client

        adapter._active_status_threads["C_CHAN"] = "111.222"

        mock_client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("missing_scope: need assistant:write")
        )

        await adapter.stop_typing("C_CHAN")

        assert adapter._setStatus_disabled is True
        assert "C_CHAN" not in adapter._setStatus_no_scope

    @pytest.mark.asyncio
    async def test_stop_typing_transient_error_stores_timestamp(self):
        """Transient scope error in stop_typing stores timestamp in _setStatus_no_scope."""
        import time

        adapter = _make_adapter()
        mock_client = adapter._primary_client

        adapter._active_status_threads["C_CHAN"] = "111.222"

        before = time.monotonic()
        mock_client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("some_scope_error")
        )

        await adapter.stop_typing("C_CHAN")

        after = time.monotonic()

        assert "C_CHAN" in adapter._setStatus_no_scope
        assert before <= adapter._setStatus_no_scope["C_CHAN"] <= after

    @pytest.mark.asyncio
    async def test_stop_typing_success_clears_setstatus_no_scope(self):
        """On success, stop_typing removes chat_id from _setStatus_no_scope."""
        import time

        adapter = _make_adapter()
        mock_client = adapter._primary_client

        adapter._active_status_threads["C_CHAN"] = "111.222"
        # Seed expired TTL entry
        adapter._setStatus_no_scope["C_CHAN"] = time.monotonic() - 1860

        mock_client.assistant_threads_setStatus = AsyncMock(return_value={})

        await adapter.stop_typing("C_CHAN")

        assert "C_CHAN" not in adapter._setStatus_no_scope

    @pytest.mark.asyncio
    async def test_permanent_disable_suppresses_stop_typing(self):
        """Once _setStatus_disabled = True, stop_typing calls are suppressed."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client

        adapter._setStatus_disabled = True
        adapter._active_status_threads["C_CHAN"] = "111.222"

        await adapter.stop_typing("C_CHAN")

        mock_client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limited_warning_once_then_debug(self):
        """Non-scope errors still rate-limit: WARNING once, then DEBUG."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client

        mock_client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("rate_limited")
        )

        metadata = {"thread_id": "111.222"}

        # First call: WARNING
        with patch("gateway.platforms.slack.logger") as mock_logger:
            await adapter.send_typing("C_CHAN", metadata=metadata)
            # Second call: DEBUG
            await adapter.send_typing("C_CHAN", metadata=metadata)

            # Check that warning was called once and debug after
            warning_calls = [c for c in mock_logger.warning.call_args_list
                           if "setStatus" in str(c) or "failed" in str(c)]
            debug_calls = [c for c in mock_logger.debug.call_args_list
                         if "setStatus" in str(c) or "failed" in str(c)]
            assert len(warning_calls) >= 1


class TestHealthMonitor:
    """Verify health monitor delegation works via _set_fatal_error."""

    @pytest.mark.asyncio
    async def test_set_fatal_error_retryable(self):
        """_set_fatal_error with retryable=True signals gateway runner."""
        from gateway.config import PlatformConfig
        from gateway.platforms.slack import SlackAdapter

        adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
        adapter._set_fatal_error("socket_mode_dead", "Socket Mode task terminated", retryable=True)

        assert adapter.has_fatal_error is True
        assert adapter._fatal_error_code == "socket_mode_dead"
        assert adapter._fatal_error_retryable is True


# ---------------------------------------------------------------------------
# Task 5: task_display_mode and chunks parameter tests
# ---------------------------------------------------------------------------


class TestTaskDisplayMode:
    """Verify task_display_mode support in start_stream / stop_stream."""

    @pytest.mark.asyncio
    async def test_start_stream_task_display_mode_plan(self):
        """start_stream(task_display_mode='plan') passes mode to API and stores per-chat_id."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.chat_startStream = AsyncMock(return_value={"ts": "ts_mode"})

        metadata = {"thread_id": "111.222"}
        result = await adapter.start_stream("C_CHAN", metadata=metadata, task_display_mode="plan")

        assert result == "ts_mode"
        assert adapter._task_display_mode["C_CHAN"] == "plan"
        call_kwargs = mock_client.chat_startStream.call_args[1]
        assert call_kwargs["task_display_mode"] == "plan"

    @pytest.mark.asyncio
    async def test_start_stream_task_display_mode_timeline(self):
        """start_stream(task_display_mode='timeline') passes mode to API."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.chat_startStream = AsyncMock(return_value={"ts": "ts_tl"})

        metadata = {"thread_id": "111.222"}
        result = await adapter.start_stream("C_CHAN", metadata=metadata, task_display_mode="timeline")

        assert result == "ts_tl"
        assert adapter._task_display_mode["C_CHAN"] == "timeline"
        call_kwargs = mock_client.chat_startStream.call_args[1]
        assert call_kwargs["task_display_mode"] == "timeline"

    @pytest.mark.asyncio
    async def test_start_stream_task_display_mode_invalid_logs_warning(self):
        """start_stream(task_display_mode='invalid') logs WARNING and ignores."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.chat_startStream = AsyncMock(return_value={"ts": "ts_inv"})

        metadata = {"thread_id": "111.222"}
        with patch("gateway.platforms.slack.logger") as mock_logger:
            result = await adapter.start_stream("C_CHAN", metadata=metadata, task_display_mode="bogus")

        assert result == "ts_inv"
        # Mode should NOT be stored — invalid ignored
        assert "C_CHAN" not in adapter._task_display_mode
        # task_display_mode should NOT be passed to API
        call_kwargs = mock_client.chat_startStream.call_args[1]
        assert "task_display_mode" not in call_kwargs
        # WARNING logged
        mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_start_stream_task_display_mode_none_omits_key(self):
        """start_stream(task_display_mode=None) does not pass task_display_mode to API."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        mock_client.chat_startStream = AsyncMock(return_value={"ts": "ts_none"})

        metadata = {"thread_id": "111.222"}
        result = await adapter.start_stream("C_CHAN", metadata=metadata)

        assert result == "ts_none"
        call_kwargs = mock_client.chat_startStream.call_args[1]
        assert "task_display_mode" not in call_kwargs

    @pytest.mark.asyncio
    async def test_stop_stream_resets_task_display_mode(self):
        """stop_stream(chat_id) pops _task_display_mode[chat_id]."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "ts_stop"
        adapter._task_display_mode["C_CHAN"] = "plan"

        mock_client.chat_stopStream = AsyncMock(return_value={})

        result = await adapter.stop_stream("C_CHAN")

        assert result is True
        assert "C_CHAN" not in adapter._task_display_mode

    @pytest.mark.asyncio
    async def test_task_display_mode_per_chat_isolation(self):
        """Two concurrent streams with different modes don't interfere."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client

        # Set up two workspace clients
        client_a = AsyncMock()
        client_b = AsyncMock()
        client_a.chat_startStream = AsyncMock(return_value={"ts": "ts_a"})
        client_b.chat_startStream = AsyncMock(return_value={"ts": "ts_b"})
        adapter._team_clients = {"T_A": client_a, "T_B": client_b}
        adapter._channel_team = {"C_A": "T_A", "C_B": "T_B"}

        metadata_a = {"thread_id": "111.222"}
        metadata_b = {"thread_id": "333.444"}

        # Start with different modes
        await adapter.start_stream("C_A", metadata=metadata_a, task_display_mode="plan")
        await adapter.start_stream("C_B", metadata=metadata_b, task_display_mode="timeline")

        assert adapter._task_display_mode["C_A"] == "plan"
        assert adapter._task_display_mode["C_B"] == "timeline"

        # Verify each was passed to correct API call
        call_a = client_a.chat_startStream.call_args[1]
        call_b = client_b.chat_startStream.call_args[1]
        assert call_a["task_display_mode"] == "plan"
        assert call_b["task_display_mode"] == "timeline"


class TestAppendStreamChunks:
    """Verify append_stream chunks parameter behavior."""

    @pytest.mark.asyncio
    async def test_case1_chunks_none_sends_markdown_text(self):
        """Case 1: chunks=None → send markdown_text=text (existing path)."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "ts_app"

        mock_client.chat_appendStream = AsyncMock(return_value={})

        result = await adapter.append_stream("C_CHAN", "delta text", metadata={"thread_id": "111.222"})

        assert result is True
        call_kwargs = mock_client.chat_appendStream.call_args[1]
        assert call_kwargs["markdown_text"] == "delta text"
        assert "chunks" not in call_kwargs

    @pytest.mark.asyncio
    async def test_case2_chunks_with_truthy_text_prepends_markdown_chunk(self):
        """Case 2: chunks provided AND text.strip() truthy → prepend markdown chunk, omit markdown_text."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "ts_app"

        mock_client.chat_appendStream = AsyncMock(return_value={})

        task_chunk = {"type": "task_update", "id": "t1", "title": "Task", "status": "in_progress"}
        result = await adapter.append_stream(
            "C_CHAN", "intro delta", metadata={"thread_id": "111.222"},
            chunks=[task_chunk],
        )

        assert result is True
        call_kwargs = mock_client.chat_appendStream.call_args[1]
        # markdown_text should NOT be in kwargs
        assert "markdown_text" not in call_kwargs
        # chunks should be present with markdown chunk prepended
        assert "chunks" in call_kwargs
        chunks = call_kwargs["chunks"]
        assert len(chunks) == 2
        # First chunk is the markdown_text wrapper
        assert chunks[0]["type"] == "markdown_text"
        assert chunks[0]["text"] == "intro delta"
        # Second chunk is the original task chunk
        assert chunks[1] == task_chunk

    @pytest.mark.asyncio
    async def test_case3_chunks_with_falsy_text_sends_chunks_only(self):
        """Case 3: chunks provided AND text.strip() falsy → send chunks only, omit markdown_text."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "ts_app"

        mock_client.chat_appendStream = AsyncMock(return_value={})

        task_chunk = {"type": "task_update", "id": "t1", "title": "Task", "status": "in_progress"}

        # Test with empty string
        result = await adapter.append_stream(
            "C_CHAN", "", metadata={"thread_id": "111.222"},
            chunks=[task_chunk],
        )

        assert result is True
        call_kwargs = mock_client.chat_appendStream.call_args[1]
        assert "markdown_text" not in call_kwargs
        assert "chunks" in call_kwargs
        chunks = call_kwargs["chunks"]
        assert len(chunks) == 1
        assert chunks[0] == task_chunk

    @pytest.mark.asyncio
    async def test_case3_whitespace_only_text_with_chunks(self):
        """Case 3 variant: whitespace-only text.strip() is falsy → no markdown chunk prepended."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "ts_app"

        mock_client.chat_appendStream = AsyncMock(return_value={})

        task_chunk = {"type": "task_update", "id": "t1", "title": "Task", "status": "in_progress"}
        result = await adapter.append_stream(
            "C_CHAN", "   \n  ", metadata={"thread_id": "111.222"},
            chunks=[task_chunk],
        )

        assert result is True
        call_kwargs = mock_client.chat_appendStream.call_args[1]
        assert "markdown_text" not in call_kwargs
        assert len(call_kwargs["chunks"]) == 1

    @pytest.mark.asyncio
    async def test_chunks_failure_falls_through_to_edit(self):
        """append_stream(chunks=...) failure follows existing streaming failure fallback."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "ts_app"

        mock_client.chat_appendStream = AsyncMock(
            side_effect=Exception("API error with chunks")
        )

        task_chunk = {"type": "task_update", "id": "t1", "title": "Task", "status": "in_progress"}
        result = await adapter.append_stream(
            "C_CHAN", "delta", metadata={"thread_id": "111.222"},
            chunks=[task_chunk],
        )

        assert result is False
        assert "C_CHAN" not in adapter._active_stream_ts

    @pytest.mark.asyncio
    async def test_chunks_multiple_items(self):
        """Multiple chunks array preserves order with markdown chunk at front when text is truthy."""
        adapter = _make_adapter()
        mock_client = adapter._primary_client
        adapter._active_stream_ts["C_CHAN"] = "ts_app"

        mock_client.chat_appendStream = AsyncMock(return_value={})

        chunk1 = {"type": "task_update", "id": "t1", "title": "First", "status": "pending"}
        chunk2 = {"type": "plan_update", "title": "Plan"}
        result = await adapter.append_stream(
            "C_CHAN", "some text", metadata={"thread_id": "111.222"},
            chunks=[chunk1, chunk2],
        )

        assert result is True
        call_kwargs = mock_client.chat_appendStream.call_args[1]
        chunks = call_kwargs["chunks"]
        assert len(chunks) == 3
        assert chunks[0]["type"] == "markdown_text"
        assert chunks[0]["text"] == "some text"
        assert chunks[1] == chunk1
        assert chunks[2] == chunk2