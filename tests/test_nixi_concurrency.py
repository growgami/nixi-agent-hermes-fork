"""Tests for NixiAdapter concurrency semaphore — bounds concurrent handle_message() calls.

Verifies:
- Semaphore limits parallel handle_message() invocations to NIXI_CONCURRENCY_LIMIT
- Semaphore releases on successful handle_message()
- Semaphore releases on handle_message() exceptions
- Default concurrency limit is 10
- Config extra takes precedence over env var
"""

import asyncio
import os
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


# ─── Semaphore initialization ────────────────────────────────────────────────


class TestConcurrencySemaphoreInit:
    """Tests for NIXI_CONCURRENCY_LIMIT config resolution and semaphore creation."""

    def test_default_concurrency_limit_is_10(self):
        """Without config or env var, concurrency limit defaults to 10."""
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {}, clear=True):
            # Remove env var if present
            os.environ.pop("NIXI_CONCURRENCY_LIMIT", None)
            adapter = NixiAdapter(_make_config())
            assert adapter._concurrency_limit == 10

    def test_concurrency_limit_from_config_extra(self):
        """Config extra NIXI_CONCURRENCY_LIMIT takes precedence over default."""
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NIXI_CONCURRENCY_LIMIT", None)
            adapter = NixiAdapter(_make_config(NIXI_CONCURRENCY_LIMIT="5"))
            assert adapter._concurrency_limit == 5

    def test_concurrency_limit_from_env_var(self):
        """Env var NIXI_CONCURRENCY_LIMIT is used when config extra not set."""
        from nixi.gateway_adapter import NixiAdapter

        # Config WITHOUT NIXI_CONCURRENCY_LIMIT in extra — falls back to env var
        config = PlatformConfig(
            enabled=True,
            extra={
                "internal_secret": "test-secret-12345678",
                "team_id": "T_TEAM_TEST",
                "host": "127.0.0.1",
                "port": 0,
            },
        )
        with patch.dict(os.environ, {"NIXI_CONCURRENCY_LIMIT": "3"}, clear=False):
            adapter = NixiAdapter(config)
            assert adapter._concurrency_limit == 3

    def test_config_extra_overrides_env_var(self):
        """Config extra takes precedence over env var."""
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {"NIXI_CONCURRENCY_LIMIT": "3"}, clear=False):
            adapter = NixiAdapter(_make_config(NIXI_CONCURRENCY_LIMIT="7"))
            assert adapter._concurrency_limit == 7

    def test_invalid_concurrency_limit_falls_back_to_default(self):
        """Non-integer env var falls back to default 10."""
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {"NIXI_CONCURRENCY_LIMIT": "not_a_number"}, clear=False):
            adapter = NixiAdapter(_make_config())
            assert adapter._concurrency_limit == 10

    def test_semaphore_created_with_correct_limit(self):
        """Semaphore internal counter matches the concurrency limit."""
        from nixi.gateway_adapter import NixiAdapter

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NIXI_CONCURRENCY_LIMIT", None)
            adapter = NixiAdapter(_make_config(NIXI_CONCURRENCY_LIMIT="5"))
            assert adapter._concurrency_semaphore._value == 5


# ─── Semaphore limiting behavior ──────────────────────────────────────────────


class TestConcurrencySemaphoreLimitsParallelTasks:
    """Tests that the semaphore bounds concurrent handle_message() calls."""

    @pytest.mark.asyncio
    async def test_at_most_n_concurrent_handle_message_calls(self):
        """With concurrency limit 5, at most 5 handle_message() calls run simultaneously."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config(NIXI_CONCURRENCY_LIMIT="5"))

        concurrency_tracker = {"count": 0, "max_concurrent": 0}
        events = [asyncio.Event() for _ in range(20)]

        async def mock_handle_message(event):
            concurrency_tracker["count"] += 1
            concurrency_tracker["max_concurrent"] = max(
                concurrency_tracker["max_concurrent"],
                concurrency_tracker["count"],
            )
            # Block until the test signals us
            idx = int(event.text.split("_")[-1])
            await events[idx].wait()
            concurrency_tracker["count"] -= 1

        adapter.handle_message = mock_handle_message

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            # Dispatch 20 events
            for i in range(20):
                await adapter._dispatch_event(
                    event_data={"event": {"text": f"msg_{i}", "channel": f"C{i}"}},
                    user_id=f"U{i}",
                    user_name=f"User{i}",
                )

            # Give tasks time to start — at most 5 should be running
            await asyncio.sleep(0.3)

            # Assert: at most 5 concurrent calls
            assert concurrency_tracker["max_concurrent"] <= 5, (
                f"Expected max 5 concurrent, got {concurrency_tracker['max_concurrent']}"
            )

            # Release all events to let tasks complete
            for e in events:
                e.set()

            # Wait for all tasks to finish
            await asyncio.sleep(0.5)

    @pytest.mark.asyncio
    async def test_all_events_processed_with_semaphore(self):
        """All events eventually get processed even with semaphore limiting."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config(NIXI_CONCURRENCY_LIMIT="3"))

        processed = []

        async def mock_handle_message(event):
            processed.append(event.text)
            await asyncio.sleep(0.01)

        adapter.handle_message = mock_handle_message

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            for i in range(10):
                await adapter._dispatch_event(
                    event_data={"event": {"text": f"msg_{i}", "channel": f"C{i}"}},
                    user_id=f"U{i}",
                    user_name=f"User{i}",
                )

            # Wait for all tasks to complete
            await asyncio.sleep(1.0)

            assert len(processed) == 10
            for i in range(10):
                assert f"msg_{i}" in processed


# ─── Semaphore release on success ────────────────────────────────────────────


class TestConcurrencySemaphoreReleasesOnSuccess:
    """Tests that the semaphore is released after successful handle_message()."""

    @pytest.mark.asyncio
    async def test_semaphore_released_after_success(self):
        """After handle_message() completes successfully, semaphore slot is freed."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config(NIXI_CONCURRENCY_LIMIT="2"))

        call_count = {"value": 0}

        async def mock_handle_message(event):
            call_count["value"] += 1
            await asyncio.sleep(0.05)

        adapter.handle_message = mock_handle_message

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            # Dispatch 2 events (fills the semaphore)
            for i in range(2):
                await adapter._dispatch_event(
                    event_data={"event": {"text": f"msg_{i}", "channel": f"C{i}"}},
                    user_id=f"U{i}",
                    user_name=f"User{i}",
                )

            # Wait for them to complete
            await asyncio.sleep(0.3)

            # Semaphore should be fully released now
            assert adapter._concurrency_semaphore._value == 2

            # Dispatch 2 more — should succeed
            for i in range(2, 4):
                await adapter._dispatch_event(
                    event_data={"event": {"text": f"msg_{i}", "channel": f"C{i}"}},
                    user_id=f"U{i}",
                    user_name=f"User{i}",
                )

            await asyncio.sleep(0.3)

            assert call_count["value"] == 4
            assert adapter._concurrency_semaphore._value == 2


# ─── Semaphore release on exception ──────────────────────────────────────────


class TestConcurrencySemaphoreReleasesOnException:
    """Tests that the semaphore is released even when handle_message() raises."""

    @pytest.mark.asyncio
    async def test_semaphore_released_after_exception(self):
        """After handle_message() raises, semaphore slot is freed."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config(NIXI_CONCURRENCY_LIMIT="2"))

        call_count = {"value": 0}

        async def failing_handle_message(event):
            call_count["value"] += 1
            raise RuntimeError("Simulated LLM failure")

        adapter.handle_message = failing_handle_message

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            # Dispatch 2 events (fills the semaphore, both will fail)
            for i in range(2):
                await adapter._dispatch_event(
                    event_data={"event": {"text": f"msg_{i}", "channel": f"C{i}"}},
                    user_id=f"U{i}",
                    user_name=f"User{i}",
                )

            # Wait for tasks to complete (they'll log the exception)
            await asyncio.sleep(0.3)

            # Both call slots should be released
            assert adapter._concurrency_semaphore._value == 2
            assert call_count["value"] == 2

    @pytest.mark.asyncio
    async def test_semaphore_release_on_exception_allows_subsequent_tasks(self):
        """After failed tasks release the semaphore, new tasks can proceed."""
        from nixi.gateway_adapter import NixiAdapter

        adapter = NixiAdapter(_make_config(NIXI_CONCURRENCY_LIMIT="2"))

        success_count = {"value": 0}

        async def sometimes_failing_handle(event):
            idx = int(event.text.split("_")[-1])
            if idx < 2:
                raise RuntimeError("First two fail")
            else:
                success_count["value"] += 1
                await asyncio.sleep(0.01)

        adapter.handle_message = sometimes_failing_handle

        with patch("nixi.gateway_adapter.load_overlay", return_value=""):
            # Dispatch 2 failing events
            for i in range(2):
                await adapter._dispatch_event(
                    event_data={"event": {"text": f"msg_{i}", "channel": f"C{i}"}},
                    user_id=f"U{i}",
                    user_name=f"User{i}",
                )

            await asyncio.sleep(0.2)

            # Semaphore should be fully released after failures
            assert adapter._concurrency_semaphore._value == 2

            # Dispatch 2 succeeding events
            for i in range(2, 4):
                await adapter._dispatch_event(
                    event_data={"event": {"text": f"msg_{i}", "channel": f"C{i}"}},
                    user_id=f"U{i}",
                    user_name=f"User{i}",
                )

            await asyncio.sleep(0.5)

            assert success_count["value"] == 2
            assert adapter._concurrency_semaphore._value == 2


# ─── Connect logs concurrency limit ────────────────────────────────────────


class TestConcurrencyLimitLogged:
    """Tests that connect() logs the concurrency limit."""

    @pytest.mark.asyncio
    async def test_connect_logs_concurrency_limit(self, caplog):
        """connect() should log the NIXI_CONCURRENCY_LIMIT at INFO level."""
        from nixi.gateway_adapter import NixiAdapter

        config = _make_config(NIXI_CONCURRENCY_LIMIT="3")
        adapter = NixiAdapter(config)
        adapter._message_handler = AsyncMock(return_value="test")

        try:
            with caplog.at_level("INFO", logger="nixi.gateway_adapter"):
                await adapter.connect()
                # Check that the concurrency limit is logged
                log_messages = [rec.message for rec in caplog.records]
                assert any("3" in msg and "concurrency" in msg.lower() for msg in log_messages), (
                    f"Expected concurrency limit log, got: {log_messages}"
                )
        finally:
            await adapter.disconnect()