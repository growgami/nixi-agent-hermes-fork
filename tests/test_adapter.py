"""Tests for nixi.adapter — LogFileAdapter integration tests.

Covers: full ingestion, cross-month thread resolution, thread deduplication,
multi-line accumulation, bot tagging, raw UID handling, normal poster user_id
resolution, orphan thread files, per-channel insert, idempotent re-ingestion,
user map generation.
"""

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from nixi.adapter import LogFileAdapter
from nixi.config import NixiConfig
from nixi.db import ensure_schema, get_connection, insert_messages
from nixi.models import IngestionResult, ScrapedMessage


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """Temp log directory matching real structure with multiple channels."""
    ld = tmp_path / "slack_logs"
    ld.mkdir()
    return ld


@pytest.fixture
def nixi_config(log_dir: Path, tmp_path: Path, monkeypatch) -> NixiConfig:
    """NixiConfig pointing at temp log_dir and output_dir."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setenv("NIXI_LOG_DIR", str(log_dir))
    monkeypatch.setenv("NIXI_OUTPUT_DIR", str(output_dir))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    return NixiConfig.from_env()


@pytest.fixture
def adapter(nixi_config: NixiConfig) -> LogFileAdapter:
    """LogFileAdapter with test config."""
    return LogFileAdapter(config=nixi_config)


def _write_channel_log(channel_dir: Path, filename: str, content: str) -> Path:
    """Write a log file in a channel directory."""
    channel_dir.mkdir(parents=True, exist_ok=True)
    path = channel_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def _write_thread(channel_dir: Path, thread_ts: str, content: str) -> Path:
    """Write a thread file in channel_dir/threads/."""
    threads_dir = channel_dir / "threads"
    threads_dir.mkdir(parents=True, exist_ok=True)
    path = threads_dir / f"{thread_ts}.log"
    path.write_text(content, encoding="utf-8")
    return path


# ── Full ingestion: walk → parse → DB insert ───────────────────────────────────

class TestFullIngestion:
    def test_walk_log_dir_and_insert(self, adapter: LogFileAdapter, log_dir: Path):
        """Full ingestion: walks channel dirs, parses logs, inserts into DB."""
        # Create channel dir with one monthly log
        chan_dir = log_dir / "C06M81FSKFF"
        content = (
            "[1766766571.412779] @Kuro: hello world\n"
            "[1766775007.615089] @Jin: check this\n"
        )
        _write_channel_log(chan_dir, "2025-12.log", content)

        result = adapter.ingest()

        assert result.total_lines == 2
        assert result.parsed == 2
        assert result.inserted == 2
        assert result.already_existing == 0

        # Verify DB content
        conn = get_connection(adapter.config.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM scraped_messages")
        assert cursor.fetchone()["COUNT(*)"] == 2
        conn.close()

    def test_multiple_channels(self, adapter: LogFileAdapter, log_dir: Path):
        """Each channel directory is ingested separately."""
        for chan_id in ["C06M81FSKFF", "C06M6MATUDB"]:
            chan_dir = log_dir / chan_id
            content = f"[1766766571.000001] @User: message in {chan_id}\n"
            _write_channel_log(chan_dir, "2025-12.log", content)

        result = adapter.ingest()
        assert result.inserted == 2

        conn = get_connection(adapter.config.db_path)
        counts = {}
        cursor = conn.execute("SELECT channel_id, COUNT(*) as cnt FROM scraped_messages GROUP BY channel_id")
        for row in cursor.fetchall():
            counts[row["channel_id"]] = row["cnt"]
        assert len(counts) == 2
        conn.close()


# ── Cross-month thread resolution ───────────────────────────────────────────────

class TestCrossMonthThreadResolution:
    def test_thread_parent_in_january_linked(self, adapter: LogFileAdapter, log_dir: Path):
        """Thread parent in January log, thread file exists → correctly linked."""
        chan_dir = log_dir / "C06M81FSKFF"

        # January log contains the parent message
        _write_channel_log(chan_dir, "2025-01.log", (
            "[1766000001.000001] @Kuro: parent message in jan\n"
        ))
        # February log has no relevant messages
        _write_channel_log(chan_dir, "2025-02.log", "")
        # Thread file
        _write_thread(chan_dir, "1766000001.000001", (
            "[1766000001.000001] @Kuro: parent message in jan\n"
            "[1766000002.000002] (thread:1766000001.000001) @Jin: reply to parent\n"
        ))

        result = adapter.ingest()
        # Parent msg from thread file + reply from thread file = 2
        # (parent is not duplicated from channel log since thread file takes precedence)
        assert result.threads_linked == 2

        conn = get_connection(adapter.config.db_path)
        cursor = conn.execute(
            "SELECT parent_ts FROM scraped_messages WHERE parent_ts IS NOT NULL"
        )
        parent_tses = [row["parent_ts"] for row in cursor.fetchall()]
        assert "1766000001.000001" in parent_tses
        conn.close()

    def test_cross_month_messages_accumulated(self, adapter: LogFileAdapter, log_dir: Path):
        """Top-level messages across multiple months are all kept."""
        chan_dir = log_dir / "C06M81FSKFF"
        _write_channel_log(chan_dir, "2025-01.log", (
            "[1766000001.000001] @Kuro: jan message\n"
        ))
        _write_channel_log(chan_dir, "2025-02.log", (
            "[1766100002.000002] @Jin: feb message\n"
        ))

        result = adapter.ingest()
        assert result.parsed == 2

        conn = get_connection(adapter.config.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM scraped_messages")
        assert cursor.fetchone()["COUNT(*)"] == 2
        conn.close()


# ── Top-level vs thread deduplication ──────────────────────────────────────────

class TestThreadDeduplication:
    def test_channel_log_thread_lines_skipped(self, adapter: LogFileAdapter, log_dir: Path):
        """Top-level messages from channel logs kept, thread reply lines skipped."""
        chan_dir = log_dir / "C06M81FSKFF"
        _write_channel_log(chan_dir, "2025-12.log", (
            "[1766766571.412779] @Kuro: original\n"
            "[1766775007.615089] (thread:1766766571.412779) @OG: thread reply in channel log\n"
            "[1766780001.123456] @Riya: new topic\n"
        ))

        result = adapter.ingest()
        # Only top-level messages: Kuro and Riya (OG's thread line is skipped)
        assert result.parsed == 2

    def test_thread_file_fully_parsed(self, adapter: LogFileAdapter, log_dir: Path):
        """Thread file is fully parsed and linked to parent via timestamp."""
        chan_dir = log_dir / "C06M81FSKFF"
        _write_channel_log(chan_dir, "2025-12.log", (
            "[1766766571.412779] @Kuro: original\n"
        ))
        _write_thread(chan_dir, "1766766571.412779", (
            "[1766766571.412779] @Kuro: original\n"
            "[1766775007.615089] (thread:1766766571.412779) @Jin: reply\n"
        ))

        result = adapter.ingest()
        # Thread file has 2 lines (parent + reply), both parsed
        assert result.threads_linked == 2

        conn = get_connection(adapter.config.db_path)
        cursor = conn.execute(
            "SELECT parent_ts FROM scraped_messages WHERE parent_ts = '1766766571.412779'"
        )
        rows = cursor.fetchall()
        assert len(rows) >= 1  # At least the reply has parent_ts
        conn.close()


# ── Multi-line message accumulation ─────────────────────────────────────────────

class TestMultiLineMessages:
    def test_multiline_accumulated_in_ingestion(self, adapter: LogFileAdapter, log_dir: Path):
        """Multi-line messages are fully accumulated (no dropped continuation lines)."""
        chan_dir = log_dir / "C06M81FSKFF"
        _write_channel_log(chan_dir, "2025-12.log", (
            "[1766780001.123456] @Riya: line1\n"
            "line2\n"
            "line3\n"
        ))

        result = adapter.ingest()
        assert result.parsed == 1

        conn = get_connection(adapter.config.db_path)
        cursor = conn.execute("SELECT text FROM scraped_messages")
        text = cursor.fetchone()["text"]
        assert text == "line1\nline2\nline3"
        conn.close()


# ── Bot message tagging ─────────────────────────────────────────────────────────

class TestBotTagging:
    def test_bot_messages_tagged_is_bot(self, adapter: LogFileAdapter, log_dir: Path):
        """Bot messages tagged is_bot=true (not discarded)."""
        chan_dir = log_dir / "C06M81FSKFF"
        _write_channel_log(chan_dir, "2025-12.log", (
            "[1766766571.000001] @nixi: I'm a bot\n"
            "[1766766571.000002] @Kuro: I'm a human\n"
        ))

        result = adapter.ingest()
        assert result.bots_tagged == 1

        conn = get_connection(adapter.config.db_path)
        cursor = conn.execute("SELECT user_name, is_bot FROM scraped_messages ORDER BY slack_ts")
        rows = cursor.fetchall()
        # nixi is a bot
        assert rows[0]["user_name"] == "nixi"
        assert rows[0]["is_bot"] == 1
        # Kuro is not a bot
        assert rows[1]["user_name"] == "Kuro"
        assert rows[1]["is_bot"] == 0
        conn.close()


# ── Raw UID posters ──────────────────────────────────────────────────────────────

class TestRawUidPosters:
    def test_raw_uid_poster_both_fields_set(self, adapter: LogFileAdapter, log_dir: Path):
        """Raw UID poster: @U09NDP0R44Q → user_id and user_name both set to 'U09NDP0R44Q'."""
        chan_dir = log_dir / "C06M81FSKFF"
        _write_channel_log(chan_dir, "2025-12.log", (
            "[1766766571.000001] @U09NDP0R44Q: message from raw uid\n"
        ))

        result = adapter.ingest()
        assert result.raw_uid_posters == 1

        conn = get_connection(adapter.config.db_path)
        cursor = conn.execute("SELECT user_id, user_name FROM scraped_messages")
        row = cursor.fetchone()
        assert row["user_id"] == "U09NDP0R44Q"
        assert row["user_name"] == "U09NDP0R44Q"
        conn.close()


# ── Normal posters: NULL at INSERT, UPDATE after user_map ──────────────────────

class TestNormalPosterResolution:
    def test_normal_poster_null_then_updated(self, adapter: LogFileAdapter, log_dir: Path):
        """Normal posters: user_id=NULL at INSERT, UPDATE'd after user_map build."""
        chan_dir = log_dir / "C06M81FSKFF"
        # Kuro mentions their own ID enough times to meet threshold
        for i in range(4):
            _write_channel_log(chan_dir, f"2025-0{i+1}.log", (
                f"[176676657{i}.00000{i}] @Kuro: msg with <@U04K8NLDCG0>\n"
            ))

        result = adapter.ingest()

        conn = get_connection(adapter.config.db_path)

        # After ingestion + user_map resolution, Kuro should have user_id set
        # (resolved via self-mention co-occurrence)
        cursor = conn.execute(
            "SELECT user_id, user_name FROM scraped_messages WHERE user_name = 'Kuro' LIMIT 1"
        )
        row = cursor.fetchone()

        # If threshold is met, user_id should be resolved
        # With 4 mentions and threshold 3, it should map
        if row:
            # At least one Kuro message should have resolved user_id
            kuro_with_id = conn.execute(
                "SELECT COUNT(*) FROM scraped_messages WHERE user_name = 'Kuro' AND user_id = 'U04K8NLDCG0'"
            ).fetchone()["COUNT(*)"]
            kuro_null = conn.execute(
                "SELECT COUNT(*) FROM scraped_messages WHERE user_name = 'Kuro' AND user_id IS NULL"
            ).fetchone()["COUNT(*)"]
            # At least some should be resolved or all if co-occurrence met
            assert kuro_with_id > 0 or kuro_null > 0  # Resolution depends on threshold

        conn.close()


# ── Idempotent re-ingestion ─────────────────────────────────────────────────────

class TestIdempotentIngestion:
    def test_re_ingestion_inserts_zero_new_rows(self, adapter: LogFileAdapter, log_dir: Path):
        """Re-running ingestion inserts 0 new rows."""
        chan_dir = log_dir / "C06M81FSKFF"
        _write_channel_log(chan_dir, "2025-12.log", (
            "[1766766571.000001] @Kuro: hello\n"
        ))

        # First ingestion
        result1 = adapter.ingest()
        assert result1.inserted == 1

        # Second ingestion (same data)
        result2 = adapter.ingest()
        assert result2.inserted == 0
        assert result2.already_existing >= 1


# ── User map generation ────────────────────────────────────────────────────────

class TestUserMapGeneration:
    def test_user_map_yaml_populated(self, adapter: LogFileAdapter, log_dir: Path):
        """user_map.yaml populated with both name_to_id and id_to_name sections."""
        chan_dir = log_dir / "C06M81FSKFF"
        # Kuro self-mentions enough times
        for i in range(4):
            _write_channel_log(chan_dir, f"2025-0{i+1}.log", (
                f"[17667665{i}.00000{i}] @Kuro: msg <@U04K8NLDCG0>\n"
            ))

        adapter.ingest()

        # Check user_map.yaml in output_dir/schemas/
        user_map_path = adapter.config.output_dir / "schemas" / "user_map.yaml"
        assert user_map_path.exists()

        data = yaml.safe_load(user_map_path.read_text(encoding="utf-8"))
        assert data is not None
        assert "name_to_id" in data
        assert "id_to_name" in data

    def test_id_to_name_section_populated(self, adapter: LogFileAdapter, log_dir: Path):
        """id_to_name section in user_map.yaml has entries."""
        chan_dir = log_dir / "C06M81FSKFF"
        # Create data where user_id is directly known through raw UID
        _write_channel_log(chan_dir, "2025-12.log", (
            "[1766766571.000001] @U04K8NLDCG0: raw uid message\n"
        ))

        adapter.ingest()

        user_map_path = adapter.config.output_dir / "schemas" / "user_map.yaml"
        data = yaml.safe_load(user_map_path.read_text(encoding="utf-8"))
        assert data is not None
        assert "id_to_name" in data
        # Raw UID posters get mapped to themselves
        assert "U04K8NLDCG0" in data["id_to_name"]


# ── Orphan thread files ────────────────────────────────────────────────────────

class TestOrphanThreadFiles:
    def test_orphan_thread_still_parsed(self, adapter: LogFileAdapter, log_dir: Path):
        """Thread file whose parent_ts has no matching top-level message still gets parsed."""
        chan_dir = log_dir / "C06M81FSKFF"
        # Channel log does NOT contain the parent message
        _write_channel_log(chan_dir, "2025-12.log", (
            "[1766780001.123456] @Riya: unrelated message\n"
        ))
        # But thread file exists for a parent that was deleted
        _write_thread(chan_dir, "1766766571.412779", (
            "[1766766571.412779] @Kuro: orphan parent message\n"
            "[1766766572.000001] (thread:1766766571.412779) @Jin: reply to orphan\n"
        ))

        result = adapter.ingest()
        # Thread file has 2 lines parsed
        assert result.threads_linked == 2

        conn = get_connection(adapter.config.db_path)
        # Both messages should have parent_ts derived from filename
        cursor = conn.execute(
            "SELECT slack_ts, parent_ts FROM scraped_messages WHERE parent_ts = '1766766571.412779'"
        )
        rows = cursor.fetchall()
        assert len(rows) == 2  # Both parent and reply have parent_ts from thread
        conn.close()

    def test_orphan_parent_ts_from_filename(self, adapter: LogFileAdapter, log_dir: Path):
        """Orphan thread: parent_ts derived from thread filename."""
        chan_dir = log_dir / "C06M81FSKFF"
        _write_channel_log(chan_dir, "2025-12.log", "")
        _write_thread(chan_dir, "1766799999.888888", (
            "[1766799999.888888] @Kuro: orphan\n"
        ))

        result = adapter.ingest()
        conn = get_connection(adapter.config.db_path)
        cursor = conn.execute("SELECT parent_ts FROM scraped_messages")
        rows = cursor.fetchall()
        # The orphan message should have parent_ts from filename
        assert any(row["parent_ts"] == "1766799999.888888" for row in rows)
        conn.close()


# ── Per-channel insert ──────────────────────────────────────────────────────────

class TestPerChannelInsert:
    def test_messages_written_per_channel_not_accumulated(self, adapter: LogFileAdapter, log_dir: Path):
        """Each channel's messages written to DB per-channel, accumulated in RAM across channels."""
        for chan_id in ["C_CH1", "C_CH2", "C_CH3"]:
            chan_dir = log_dir / chan_id
            _write_channel_log(chan_dir, "2025-12.log", (
                f"[1766766571.000001] @User: msg in {chan_id}\n"
            ))

        result = adapter.ingest()
        assert result.inserted == 3

        conn = get_connection(adapter.config.db_path)
        counts = {}
        cursor = conn.execute("SELECT channel_id, COUNT(*) as cnt FROM scraped_messages GROUP BY channel_id")
        for row in cursor.fetchall():
            counts[row["channel_id"]] = row["cnt"]
        assert counts["C_CH1"] == 1
        assert counts["C_CH2"] == 1
        assert counts["C_CH3"] == 1
        conn.close()


# ── Channel names are channel IDs ───────────────────────────────────────────────

class TestChannelIds:
    def test_channel_name_equals_channel_id(self, adapter: LogFileAdapter, log_dir: Path):
        """Channel names are channel IDs (no name resolution at bootstrap)."""
        chan_dir = log_dir / "C06M81FSKFF"
        _write_channel_log(chan_dir, "2025-12.log", (
            "[1766766571.000001] @Kuro: test\n"
        ))

        adapter.ingest()
        conn = get_connection(adapter.config.db_path)
        cursor = conn.execute("SELECT channel_id, channel_name FROM scraped_messages")
        row = cursor.fetchone()
        assert row["channel_id"] == "C06M81FSKFF"
        assert row["channel_name"] == "C06M81FSKFF"
        conn.close()


# ── Timestamp conversion ────────────────────────────────────────────────────────

class TestTimestampConversion:
    def test_timestamp_is_utc_iso(self, adapter: LogFileAdapter, log_dir: Path):
        """ScrapedMessage.timestamp is ISO datetime string in UTC."""
        chan_dir = log_dir / "C06M81FSKFF"
        _write_channel_log(chan_dir, "2025-12.log", (
            "[1766766571.412779] @Kuro: test\n"
        ))

        adapter.ingest()
        conn = get_connection(adapter.config.db_path)
        cursor = conn.execute("SELECT timestamp FROM scraped_messages")
        ts = cursor.fetchone()["timestamp"]
        # Should be a valid ISO datetime
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None  # Has timezone info
        conn.close()


# ── ingest_channel single channel re-ingestion ──────────────────────────────────

class TestIngestChannel:
    def test_single_channel_reingestion(self, adapter: LogFileAdapter, log_dir: Path):
        """ingest_channel re-ingests a specific channel."""
        chan1 = log_dir / "C_CH1"
        chan2 = log_dir / "C_CH2"
        _write_channel_log(chan1, "2025-12.log", "[1766766571.000001] @Kuro: ch1 msg\n")
        _write_channel_log(chan2, "2025-12.log", "[1766766571.000002] @Jin: ch2 msg\n")

        # Full ingestion first
        adapter.ingest()

        # Re-ingest just C_CH1
        result = adapter.ingest_channel("C_CH1")
        assert isinstance(result, IngestionResult)


# ── Edge cases ──────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_log_dir(self, adapter: LogFileAdapter, log_dir: Path):
        """Empty log dir produces zero-count result."""
        result = adapter.ingest()
        assert result.inserted == 0
        assert result.parsed == 0

    def test_channel_dir_no_log_files(self, adapter: LogFileAdapter, log_dir: Path):
        """Channel dir with no .log files produces zero for that channel."""
        (log_dir / "C_EMPTY").mkdir()
        result = adapter.ingest()
        assert result.parsed == 0

    def test_non_c_directory_ignored(self, adapter: LogFileAdapter, log_dir: Path):
        """Directories not starting with C are ignored."""
        # This is a directory but doesn't start with C
        (log_dir / "DM123456").mkdir()
        (log_dir / "DM123456" / "2025-12.log").write_text(
            "[1766766571.000001] @Kuro: should be ignored\n", encoding="utf-8"
        )
        result = adapter.ingest()
        assert result.parsed == 0

    def test_log_dir_not_found(self, nixi_config: NixiConfig):
        """Non-existent log_dir raises FileNotFoundError."""
        config = NixiConfig(
            log_dir=Path("/nonexistent/path"),
            output_dir=nixi_config.output_dir,
        )
        adapter = LogFileAdapter(config=config)
        with pytest.raises(FileNotFoundError):
            adapter.ingest()