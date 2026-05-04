"""Integration tests for realtime message extraction pipeline.

Tests the complete flow from realtime_messages table through the extraction
pipeline: reading unprocessed messages, marking them extracted, and verifying
ExtractionBatcher reads from realtime_messages when source='realtime'.

These tests use an in-memory/temp SQLite database and DO NOT require an LLM
or a running ingester — they test the Python side only.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nixi.db import (
    ensure_realtime_schema,
    ensure_schema,
    get_connection,
    get_realtime_unprocessed,
    get_realtime_unprocessed_channels,
    mark_extracted,
)
from nixi.extraction.batch import ExtractionBatcher
from nixi.models import RealtimeMessage


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def realtime_db(tmp_path: Path):
    """Temp database with both realtime_messages and nixi_extraction_log."""
    db_path = tmp_path / "nixi_state.db"
    ensure_realtime_schema(db_path)
    ensure_schema(db_path)
    conn = get_connection(db_path)
    yield conn
    conn.close()


def _insert_realtime_message(
    conn,
    slack_ts="1700000000.000100",
    channel_id="C_RT_TEST",
    user_id="U_USER01",
    text="Hello from realtime",
    thread_ts=None,
    parent_ts=None,
    is_bot=0,
    channel_type="channel",
    event_id="Ev_RT_001",
    client_msg_id=None,
    team_id="T_TEAM01",
    timestamp="2023-11-14T22:13:20Z",
):
    """Helper to insert a row into realtime_messages."""
    conn.execute(
        """INSERT INTO realtime_messages
           (slack_ts, channel_id, user_id, text, thread_ts, parent_ts,
            is_bot, channel_type, event_id, client_msg_id, team_id, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (slack_ts, channel_id, user_id, text, thread_ts, parent_ts,
         is_bot, channel_type, event_id, client_msg_id, team_id, timestamp),
    )
    conn.commit()


# ── get_realtime_unprocessed ──────────────────────────────────────────────────


class TestRealtimeUnprocessedIntegration:
    """Integration tests for get_realtime_unprocessed with full schema."""

    def test_returns_all_unprocessed_messages(self, realtime_db):
        """All messages in realtime_messages are returned when none are extracted."""
        for i in range(5):
            _insert_realtime_message(
                realtime_db,
                slack_ts=f"1700000000.{i:06d}",
                channel_id="C_RT_TEST",
                event_id=f"Ev_RT_{i:03d}",
            )

        result = get_realtime_unprocessed(realtime_db, "C_RT_TEST")
        assert len(result) == 5
        # Verify all fields are present
        for msg in result:
            assert "slack_ts" in msg
            assert "channel_id" in msg
            assert "text" in msg
            assert "event_id" in msg
            assert "timestamp" in msg

    def test_excludes_messages_in_extraction_log(self, realtime_db):
        """Messages marked in extraction_log are excluded from unprocessed."""
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000100",
            channel_id="C_RT_TEST",
            event_id="Ev_RT_001",
        )
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000200",
            channel_id="C_RT_TEST",
            event_id="Ev_RT_002",
        )

        # Mark first as extracted
        mark_extracted(realtime_db, "C_RT_TEST", ["1700000000.000100"], "batch-int-1")

        result = get_realtime_unprocessed(realtime_db, "C_RT_TEST")
        assert len(result) == 1
        assert result[0]["slack_ts"] == "1700000000.000200"

    def test_shared_extraction_log_dedup_across_sources(self, realtime_db):
        """A message extracted from scraped_messages won't be re-extracted from realtime.

        The nixi_extraction_log uses (channel_id, slack_ts) as the dedup key,
        shared between scraped and realtime sources.
        """
        # Insert into scraped_messages via db helpers
        from nixi.db import insert_messages
        from nixi.models import ScrapedMessage

        msg = ScrapedMessage(
            slack_ts="1700000000.000100",
            channel_id="C_RT_TEST",
            channel_name="test-channel",
            user_id="U_USER01",
            user_name="TestUser",
            text="Same message in both tables",
            thread_ts=None,
            parent_ts=None,
            is_bot=False,
            source_file="test",
            timestamp="2023-11-14T22:13:20Z",
        )
        insert_messages(realtime_db, [msg])

        # Same message in realtime_messages (same channel_id + slack_ts)
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000100",
            channel_id="C_RT_TEST",
            event_id="Ev_RT_CROSS",
        )

        # Mark extracted from scraped_messages
        mark_extracted(realtime_db, "C_RT_TEST", ["1700000000.000100"], "batch-cross-1")

        # Realtime unprocessed should also exclude it (shared extraction log)
        result = get_realtime_unprocessed(realtime_db, "C_RT_TEST")
        assert len(result) == 0

    def test_returns_messages_ordered_by_timestamp(self, realtime_db):
        """Messages are returned in timestamp order (oldest first)."""
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000001.000200",
            channel_id="C_RT_TEST",
            event_id="Ev_RT_LATE",
            timestamp="2023-11-14T22:13:21Z",
        )
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000100",
            channel_id="C_RT_TEST",
            event_id="Ev_RT_EARLY",
            timestamp="2023-11-14T22:13:20Z",
        )

        result = get_realtime_unprocessed(realtime_db, "C_RT_TEST")
        assert len(result) == 2
        assert result[0]["slack_ts"] == "1700000000.000100"
        assert result[1]["slack_ts"] == "1700000001.000200"

    def test_limit_parameter_caps_results(self, realtime_db):
        """Limit parameter caps the number of returned messages."""
        for i in range(10):
            _insert_realtime_message(
                realtime_db,
                slack_ts=f"1700000000.{i:06d}",
                channel_id="C_RT_TEST",
                event_id=f"Ev_RT_{i:03d}",
            )

        result = get_realtime_unprocessed(realtime_db, "C_RT_TEST", limit=3)
        assert len(result) == 3

    def test_filters_by_channel(self, realtime_db):
        """Only returns messages for the specified channel."""
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000100",
            channel_id="C_CHANNEL_A",
            event_id="Ev_A",
        )
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000200",
            channel_id="C_CHANNEL_B",
            event_id="Ev_B",
        )

        result_a = get_realtime_unprocessed(realtime_db, "C_CHANNEL_A")
        result_b = get_realtime_unprocessed(realtime_db, "C_CHANNEL_B")
        assert len(result_a) == 1
        assert len(result_b) == 1
        assert result_a[0]["channel_id"] == "C_CHANNEL_A"


# ── get_realtime_unprocessed_channels ────────────────────────────────────────


class TestRealtimeUnprocessedChannelsIntegration:
    """Integration tests for get_realtime_unprocessed_channels."""

    def test_returns_channels_with_unprocessed_messages(self, realtime_db):
        """Returns distinct channel_ids that have unprocessed realtime messages."""
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000100",
            channel_id="C_CHANNEL_A",
            event_id="Ev_A",
        )
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000200",
            channel_id="C_CHANNEL_B",
            event_id="Ev_B",
        )

        channels = get_realtime_unprocessed_channels(realtime_db)
        assert "C_CHANNEL_A" in channels
        assert "C_CHANNEL_B" in channels

    def test_extracts_fully_extracted_channels(self, realtime_db):
        """Channels where all messages are extracted are not returned."""
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000100",
            channel_id="C_FULLY_EXTRACTED",
            event_id="Ev_EX1",
        )
        mark_extracted(realtime_db, "C_FULLY_EXTRACTED", ["1700000000.000100"], "batch-ex-1")

        channels = get_realtime_unprocessed_channels(realtime_db)
        assert "C_FULLY_EXTRACTED" not in channels

    def test_separates_channels_independently(self, realtime_db):
        """Extracting one channel's messages doesn't affect another channel."""
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000100",
            channel_id="C_CH_A",
            event_id="Ev_A",
        )
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000200",
            channel_id="C_CH_B",
            event_id="Ev_B",
        )

        # Extract channel A's messages
        mark_extracted(realtime_db, "C_CH_A", ["1700000000.000100"], "batch-sep-1")

        channels = get_realtime_unprocessed_channels(realtime_db)
        assert "C_CH_A" not in channels
        assert "C_CH_B" in channels


# ── mark_extracted (shared between sources) ────────────────────────────────────


class TestMarkExtractedIntegration:
    """Integration tests for mark_extracted with realtime messages."""

    def test_mark_extracted_prevents_reextraction(self, realtime_db):
        """After marking, get_realtime_unprocessed returns 0 for that channel."""
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000000.000100",
            channel_id="C_RT_TEST",
            event_id="Ev_RT_001",
        )

        mark_extracted(realtime_db, "C_RT_TEST", ["1700000000.000100"], "batch-mark-1")

        result = get_realtime_unprocessed(realtime_db, "C_RT_TEST")
        assert len(result) == 0

    def test_mark_multiple_messages(self, realtime_db):
        """mark_extracted can mark multiple slack_ts at once."""
        for i in range(3):
            _insert_realtime_message(
                realtime_db,
                slack_ts=f"1700000000.{i:06d}",
                channel_id="C_RT_TEST",
                event_id=f"Ev_RT_{i:03d}",
            )

        mark_extracted(
            realtime_db,
            "C_RT_TEST",
            ["1700000000.000000", "1700000000.000001", "1700000000.000002"],
            "batch-mark-multi",
        )

        # Verify all are in extraction log
        cursor = realtime_db.execute("SELECT COUNT(*) FROM nixi_extraction_log")
        assert cursor.fetchone()["COUNT(*)"] == 3

    def test_incremental_extraction(self, realtime_db):
        """Extracting in batches: first batch marks some, second batch picks up remainder."""
        for i in range(5):
            _insert_realtime_message(
                realtime_db,
                slack_ts=f"1700000000.{i:06d}",
                channel_id="C_RT_TEST",
                event_id=f"Ev_RT_{i:03d}",
            )

        # First extraction batch
        mark_extracted(
            realtime_db,
            "C_RT_TEST",
            ["1700000000.000000", "1700000000.000001", "1700000000.000002"],
            "batch-incremental-1",
        )

        # Should have 2 unprocessed remaining
        result = get_realtime_unprocessed(realtime_db, "C_RT_TEST")
        assert len(result) == 2

        # Second extraction batch
        mark_extracted(
            realtime_db,
            "C_RT_TEST",
            ["1700000000.000003", "1700000000.000004"],
            "batch-incremental-2",
        )

        # All extracted
        result = get_realtime_unprocessed(realtime_db, "C_RT_TEST")
        assert len(result) == 0


# ── ExtractionBatcher with source='realtime' ──────────────────────────────────


class TestExtractionBatcherRealtimeIntegration:
    """Integration tests for ExtractionBatcher with source='realtime'."""

    def test_realtime_source_routes_to_realtime_queries(self, realtime_db, tmp_path):
        """ExtractionBatcher(source='realtime') reads from realtime_messages table."""
        from nixi.config import NixiConfig

        # Insert sufficient messages to exceed min_messages threshold
        for i in range(25):
            _insert_realtime_message(
                realtime_db,
                slack_ts=f"1700000000.{i:06d}",
                channel_id="C_RT_BATCH",
                event_id=f"Ev_BATCH_{i:03d}",
                user_id="U_BATCH_USER",
            )

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="## Extracted\n- Test data")

        config = NixiConfig(
            log_dir=tmp_path / "logs",
            output_dir=tmp_path / "output",
            extraction_batch_size=50,
        )

        with patch("nixi.extraction.batch.write_org_facts"), \
             patch("nixi.extraction.batch.write_rules"), \
             patch("nixi.extraction.batch.write_employee_info"), \
             patch("nixi.extraction.batch.write_channel_skill"):
            batcher = ExtractionBatcher(config, realtime_db, mock_llm, source="realtime", min_messages=20)
            # extract_channel is async, so we need to run it
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                batcher.extract_channel("C_RT_BATCH")
            )

        assert result is not None
        assert result.get("skipped") is False
        assert result.get("message_count") == 25

        # Verify messages are marked in extraction log
        unprocessed = get_realtime_unprocessed(realtime_db, "C_RT_BATCH")
        assert len(unprocessed) == 0

    def test_realtime_source_below_threshold_skips(self, realtime_db, tmp_path):
        """ExtractionBatcher(source='realtime') skips channels below min_messages."""
        from nixi.config import NixiConfig

        # Only 5 messages — below default threshold
        for i in range(5):
            _insert_realtime_message(
                realtime_db,
                slack_ts=f"1700000001.{i:06d}",
                channel_id="C_RT_SMALL",
                event_id=f"Ev_SMALL_{i:03d}",
            )

        mock_llm = AsyncMock()
        config = NixiConfig(
            log_dir=tmp_path / "logs",
            output_dir=tmp_path / "output",
        )

        batcher = ExtractionBatcher(config, realtime_db, mock_llm, source="realtime", min_messages=20)
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            batcher.extract_channel("C_RT_SMALL")
        )

        assert result is not None
        assert result.get("skipped") is True

    def test_realtime_source_uses_shared_extraction_log(self, realtime_db, tmp_path):
        """Realtime extraction marks in shared nixi_extraction_log table."""
        from nixi.config import NixiConfig

        for i in range(25):
            _insert_realtime_message(
                realtime_db,
                slack_ts=f"1700000002.{i:06d}",
                channel_id="C_RT_SHARED",
                event_id=f"Ev_SHARED_{i:03d}",
            )

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="## Extracted\n- Shared log test")

        config = NixiConfig(
            log_dir=tmp_path / "logs",
            output_dir=tmp_path / "output",
        )

        with patch("nixi.extraction.batch.write_org_facts"), \
             patch("nixi.extraction.batch.write_rules"), \
             patch("nixi.extraction.batch.write_employee_info"), \
             patch("nixi.extraction.batch.write_channel_skill"):
            batcher = ExtractionBatcher(config, realtime_db, mock_llm, source="realtime", min_messages=20)
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                batcher.extract_channel("C_RT_SHARED")
            )

        # Verify extraction_log has entries with the correct channel_id
        cursor = realtime_db.execute(
            "SELECT channel_id, COUNT(*) as cnt FROM nixi_extraction_log GROUP BY channel_id"
        )
        rows = {row["channel_id"]: row["cnt"] for row in cursor.fetchall()}
        assert "C_RT_SHARED" in rows
        assert rows["C_RT_SHARED"] == 25

    def test_realtime_source_preserves_all_fields(self, realtime_db, tmp_path):
        """Messages read from realtime_messages preserve all relevant fields."""
        _insert_realtime_message(
            realtime_db,
            slack_ts="1700000003.000100",
            channel_id="C_RT_FIELDS",
            user_id="U_FIELDS_USER",
            text="Test message with all fields",
            thread_ts="1700000003.000050",
            parent_ts="1700000003.000050",
            is_bot=1,
            channel_type="group",
            event_id="Ev_FIELDS_001",
            client_msg_id="client-uuid-fields",
            team_id="T_FIELDS_TEAM",
            timestamp="2023-11-14T22:13:23Z",
        )

        result = get_realtime_unprocessed(realtime_db, "C_RT_FIELDS")
        assert len(result) == 1
        msg = result[0]

        # All fields must be present as dict keys
        assert msg["slack_ts"] == "1700000003.000100"
        assert msg["channel_id"] == "C_RT_FIELDS"
        assert msg["user_id"] == "U_FIELDS_USER"
        assert msg["text"] == "Test message with all fields"
        assert msg["thread_ts"] == "1700000003.000050"
        assert msg["parent_ts"] == "1700000003.000050"
        assert msg["is_bot"] == 1
        assert msg["channel_type"] == "group"
        assert msg["event_id"] == "Ev_FIELDS_001"
        assert msg["client_msg_id"] == "client-uuid-fields"
        assert msg["team_id"] == "T_FIELDS_TEAM"