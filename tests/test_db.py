"""Tests for nixi.db — database helpers for nixi_state.db.

Covers: schema creation, message insertion, get_unprocessed,
get_unprocessed_channels, mark_extracted, build_user_map.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from nixi.db import (
    build_user_map,
    count_by_channel,
    ensure_schema,
    get_connection,
    get_unprocessed,
    get_unprocessed_channels,
    insert_messages,
    mark_extracted,
)
from nixi.models import ScrapedMessage, UserMap


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Temp database path."""
    return tmp_path / "nixi_state.db"


@pytest.fixture
def conn(db_path: Path):
    """Schema-initialized database connection."""
    ensure_schema(db_path)
    connection = get_connection(db_path)
    yield connection
    connection.close()


def _make_message(
    slack_ts: str = "1766766571.412779",
    channel_id: str = "C06M81FSKFF",
    channel_name: str = "C06M81FSKFF",
    user_id: str | None = None,
    user_name: str = "Kuro",
    text: str = "hello world",
    thread_ts: str | None = None,
    parent_ts: str | None = None,
    is_bot: bool = False,
    source_file: str = "C06M81FSKFF",
    timestamp: str | None = None,
) -> ScrapedMessage:
    """Helper to build test ScrapedMessage records."""
    if timestamp is None:
        timestamp = datetime.fromtimestamp(float(slack_ts), tz=timezone.utc).isoformat()
    return ScrapedMessage(
        slack_ts=slack_ts,
        channel_id=channel_id,
        channel_name=channel_name,
        user_id=user_id,
        user_name=user_name,
        text=text,
        thread_ts=thread_ts,
        parent_ts=parent_ts,
        is_bot=is_bot,
        source_file=source_file,
        timestamp=timestamp,
    )


# ── ensure_schema ──────────────────────────────────────────────────────────────

class TestEnsureSchema:
    def test_creates_both_tables_and_indexes(self, db_path: Path):
        """ensure_schema creates scraped_messages, nixi_extraction_log, and composite indexes."""
        ensure_schema(db_path)
        conn = get_connection(db_path)

        # Verify both tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "scraped_messages" in tables
        assert "nixi_extraction_log" in tables

        # Verify indexes exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        )
        indexes = {row["name"] for row in cursor.fetchall()}
        assert "idx_sm_channel_ts" in indexes
        assert "idx_sm_bot" in indexes
        assert "idx_extraction_batch" in indexes
        assert "idx_el_channel_ts" in indexes

        conn.close()

    def test_idempotent_schema_creation(self, db_path: Path):
        """Running ensure_schema twice does not raise."""
        ensure_schema(db_path)
        ensure_schema(db_path)  # Should not raise

    def test_creates_parent_directory(self, tmp_path: Path):
        """ensure_schema creates db parent directory if missing."""
        db_path = tmp_path / "nested" / "dir" / "nixi_state.db"
        ensure_schema(db_path)
        assert db_path.parent.is_dir()


# ── insert_messages ────────────────────────────────────────────────────────────

class TestInsertMessages:
    def test_batch_insert(self, conn):
        """Batch INSERT inserts all messages."""
        msgs = [
            _make_message(slack_ts="1766766571.000001", user_name="Kuro"),
            _make_message(slack_ts="1766766571.000002", user_name="Jin"),
        ]
        inserted = insert_messages(conn, msgs)
        assert inserted == 2

        cursor = conn.execute("SELECT COUNT(*) FROM scraped_messages")
        assert cursor.fetchone()["COUNT(*)"] == 2

    def test_insert_or_ignore_duplicate(self, conn):
        """INSERT OR IGNORE: re-inserting same (channel_id, slack_ts) inserts 0 new rows."""
        msg = _make_message(slack_ts="1766766571.412779")
        insert_messages(conn, [msg])
        # Re-insert
        inserted = insert_messages(conn, [msg])
        assert inserted == 0

        cursor = conn.execute("SELECT COUNT(*) FROM scraped_messages")
        assert cursor.fetchone()["COUNT(*)"] == 1

    def test_insert_empty_list(self, conn):
        """insert_messages with empty list returns 0."""
        result = insert_messages(conn, [])
        assert result == 0

    def test_insert_with_null_user_id(self, conn):
        """Messages with user_id=NULL are inserted correctly."""
        msg = _make_message(user_id=None, user_name="Kuro")
        insert_messages(conn, [msg])
        cursor = conn.execute("SELECT user_id, user_name FROM scraped_messages")
        row = cursor.fetchone()
        assert row["user_id"] is None
        assert row["user_name"] == "Kuro"

    def test_insert_with_raw_uid(self, conn):
        """Raw UID posters: user_id and user_name both set to UID."""
        msg = _make_message(user_id="U09NDP0R44Q", user_name="U09NDP0R44Q")
        insert_messages(conn, [msg])
        cursor = conn.execute("SELECT user_id, user_name FROM scraped_messages")
        row = cursor.fetchone()
        assert row["user_id"] == "U09NDP0R44Q"
        assert row["user_name"] == "U09NDP0R44Q"

    def test_insert_bot_message(self, conn):
        """Bot messages tagged with is_bot=1."""
        msg = _make_message(user_name="Toothless", is_bot=True)
        insert_messages(conn, [msg])
        cursor = conn.execute("SELECT is_bot FROM scraped_messages")
        assert cursor.fetchone()["is_bot"] == 1

    def test_insert_thread_message(self, conn):
        """Thread replies have parent_ts set."""
        msg = _make_message(
            slack_ts="1766775007.615089",
            parent_ts="1766766571.412779",
            thread_ts="1766766571.412779",
        )
        insert_messages(conn, [msg])
        cursor = conn.execute("SELECT parent_ts, thread_ts FROM scraped_messages")
        row = cursor.fetchone()
        assert row["parent_ts"] == "1766766571.412779"
        assert row["thread_ts"] == "1766766571.412779"


# ── get_unprocessed ────────────────────────────────────────────────────────────

class TestGetUnprocessed:
    def test_excludes_extracted_messages(self, conn):
        """Messages in extraction_log are excluded from unprocessed."""
        msgs = [
            _make_message(slack_ts="1766766571.000001"),
            _make_message(slack_ts="1766766571.000002"),
        ]
        insert_messages(conn, msgs)

        # Mark first as extracted
        mark_extracted(conn, "C06M81FSKFF", ["1766766571.000001"], "batch-1")

        unprocessed = get_unprocessed(conn, "C06M81FSKFF")
        assert len(unprocessed) == 1
        assert unprocessed[0]["slack_ts"] == "1766766571.000002"

    def test_returns_all_when_none_extracted(self, conn):
        """All messages returned when none are in extraction_log."""
        msgs = [
            _make_message(slack_ts="1766766571.000001"),
            _make_message(slack_ts="1766766571.000002"),
        ]
        insert_messages(conn, msgs)

        unprocessed = get_unprocessed(conn, "C06M81FSKFF")
        assert len(unprocessed) == 2

    def test_limit_parameter(self, conn):
        """Limit parameter caps the result count."""
        for i in range(5):
            insert_messages(conn, [_make_message(slack_ts=f"1766766571.{i:06d}")])

        unprocessed = get_unprocessed(conn, "C06M81FSKFF", limit=3)
        assert len(unprocessed) == 3

    def test_filters_by_channel(self, conn):
        """Only returns messages for the requested channel."""
        insert_messages(conn, [_make_message(slack_ts="1766766571.000001", channel_id="C_CHANNEL1")])
        insert_messages(conn, [_make_message(slack_ts="1766766571.000002", channel_id="C_CHANNEL2")])

        unprocessed = get_unprocessed(conn, "C_CHANNEL1")
        assert len(unprocessed) == 1
        assert unprocessed[0]["channel_id"] == "C_CHANNEL1"


# ── get_unprocessed_channels ────────────────────────────────────────────────────

class TestGetUnprocessedChannels:
    def test_returns_channels_with_no_extraction(self, conn):
        """Returns channel_ids that have no extraction log entries."""
        insert_messages(conn, [_make_message(channel_id="C_CH1")])
        insert_messages(conn, [_make_message(channel_id="C_CH2")])

        channels = get_unprocessed_channels(conn)
        assert "C_CH1" in channels
        assert "C_CH2" in channels

    def test_excludes_fully_extracted_channels(self, conn):
        """A channel with all messages extracted is not returned."""
        insert_messages(conn, [_make_message(channel_id="C_CH1")])
        mark_extracted(conn, "C_CH1", ["1766766571.412779"], "batch-1")

        channels = get_unprocessed_channels(conn)
        assert "C_CH1" not in channels


# ── count_by_channel ──────────────────────────────────────────────────────────

class TestCountByChannel:
    def test_counts_per_channel(self, conn):
        """Returns dict mapping channel_id to message count."""
        for i in range(3):
            insert_messages(conn, [_make_message(slack_ts=f"1766766571.{i:06d}", channel_id="C_CH1")])
        for i in range(2):
            insert_messages(conn, [_make_message(slack_ts=f"1766766572.{i:06d}", channel_id="C_CH2")])

        counts = count_by_channel(conn)
        assert counts["C_CH1"] == 3
        assert counts["C_CH2"] == 2

    def test_empty_db(self, conn):
        """Returns empty dict for no messages."""
        assert count_by_channel(conn) == {}


# ── mark_extracted ─────────────────────────────────────────────────────────────

class TestMarkExtracted:
    def test_marks_multiple_timestamps(self, conn):
        """Inserts multiple extraction log entries."""
        insert_messages(conn, [
            _make_message(slack_ts="1766766571.000001"),
            _make_message(slack_ts="1766766571.000002"),
        ])
        mark_extracted(conn, "C06M81FSKFF", ["1766766571.000001", "1766766571.000002"], "batch-1")

        cursor = conn.execute("SELECT COUNT(*) FROM nixi_extraction_log")
        assert cursor.fetchone()["COUNT(*)"] == 2


# ── build_user_map ─────────────────────────────────────────────────────────────

class TestBuildUserMap:
    def test_returns_user_map_with_both_mappings(self, conn):
        """build_user_map returns UserMap with name_to_id and id_to_name."""
        # Direct user_id correlations
        msgs = [
            _make_message(slack_ts="1766766571.000001", user_name="Kuro", user_id="U04K8NLDCG0"),
            _make_message(slack_ts="1766766571.000002", user_name="Jin", user_id="U073D278H62"),
        ]
        insert_messages(conn, msgs)

        user_map = build_user_map(conn)
        assert isinstance(user_map, UserMap)
        assert user_map.name_to_id["Kuro"] == "U04K8NLDCG0"
        assert user_map.id_to_name["U04K8NLDCG0"] == "Kuro"

    def test_cooccurrence_heuristic(self, conn):
        """Self-mention heuristic maps display_name → user_id via co-occurrence."""
        # Kuro mentions <@U04K8NLDCG0> in multiple messages — but Kuro IS U04K8NLDCG0
        # Let's use a different scenario: display_name "Riya" mentions <@U5555555555>
        # enough times to meet threshold
        msgs = []
        for i in range(4):  # Above default threshold of 3
            msgs.append(_make_message(
                slack_ts=f"176676657{i}.00000{i}",
                user_name="Riya",
                user_id=None,  # Not yet resolved
                text=f"hey <@U5555555555> check this",
            ))
        insert_messages(conn, msgs)

        user_map = build_user_map(conn, cooccurrence_threshold=3)
        # Riya → U5555555555 via co-occurrence (4 mentions ≥ 3 threshold)
        assert user_map.name_to_id.get("Riya") == "U5555555555"
        assert user_map.id_to_name.get("U5555555555") == "Riya"

    def test_below_threshold_not_mapped(self, conn):
        """Co-occurrences below threshold are not mapped."""
        msgs = [
            _make_message(
                slack_ts="1766766571.000001",
                user_name="Casual",
                user_id=None,
                text="just once <@U9999999999>",
            ),
        ]
        insert_messages(conn, msgs)

        user_map = build_user_map(conn, cooccurrence_threshold=3)
        # Only 1 mention, below threshold of 3
        assert "Casual" not in user_map.name_to_id or user_map.name_to_id["Casual"] is None

    def test_id_to_name_populated(self, conn):
        """id_to_name section populated from direct correlations."""
        insert_messages(conn, [
            _make_message(slack_ts="1766766571.000001", user_name="Alice", user_id="U1111111111"),
        ])
        user_map = build_user_map(conn)
        assert user_map.id_to_name["U1111111111"] == "Alice"


# ── get_connection ─────────────────────────────────────────────────────────────

class TestGetConnection:
    def test_wal_mode_enabled(self, db_path: Path):
        """get_connection returns connection with WAL mode."""
        ensure_schema(db_path)
        conn = get_connection(db_path)
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()["journal_mode"]
        # WAL mode is uppercase on some SQLite versions, lowercase on others
        assert mode.lower() == "wal"
        conn.close()

    def test_row_factory_is_dict(self, db_path: Path):
        """Row factory returns dict-like rows."""
        ensure_schema(db_path)
        conn = get_connection(db_path)
        insert_messages(conn, [_make_message()])
        cursor = conn.execute("SELECT slack_ts FROM scraped_messages")
        row = cursor.fetchone()
        assert row["slack_ts"] == "1766766571.412779"
        conn.close()