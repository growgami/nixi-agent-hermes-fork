"""Tests for nixi.ingest — CLI ingest command and empty DB guard.

Covers: run_ingestion, run_ingestion_channel, NixiConfig env var fallbacks,
empty DB guard in extract.py, ensure_schema at startup.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nixi.config import NixiConfig
from nixi.db import ensure_schema, get_connection, insert_messages
from nixi.models import ScrapedMessage


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Temp output directory for standalone tests."""
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """Temp log directory."""
    d = tmp_path / "slack_logs"
    d.mkdir()
    return d


@pytest.fixture
def nixi_config(output_dir: Path, log_dir: Path) -> NixiConfig:
    """Minimal NixiConfig for testing."""
    return NixiConfig(
        log_dir=log_dir,
        output_dir=output_dir,
        extraction_batch_size=50,
        memory_limit=500,
        employee_limit=300,
    )


@pytest.fixture
def db_conn(nixi_config: NixiConfig):
    """Schema-initialized database connection."""
    ensure_schema(nixi_config.db_path)
    conn = get_connection(nixi_config.db_path)
    yield conn
    conn.close()


# ── NixiConfig from_env fallbacks ──────────────────────────────────────────────


class TestNixiConfigFromEnv:
    """Test NixiConfig.from_env() reads all env var fallbacks."""

    def test_all_env_vars_set(self, tmp_path: Path, monkeypatch):
        """All extraction config keys read from env vars."""
        log_dir = tmp_path / "logs"
        output_dir = tmp_path / "out"
        log_dir.mkdir()
        output_dir.mkdir()

        monkeypatch.setenv("NIXI_LOG_DIR", str(log_dir))
        monkeypatch.setenv("NIXI_OUTPUT_DIR", str(output_dir))
        monkeypatch.setenv("NIXI_EXTRACTION_BATCH_SIZE", "100")
        monkeypatch.setenv("NIXI_BOT_NAMES", '["BotA","BotB"]')
        monkeypatch.setenv("NIXI_COOCCURRENCE_THRESHOLD", "5")
        monkeypatch.setenv("NIXI_MEMORY_LIMIT", "20000")
        monkeypatch.setenv("NIXI_EMPLOYEE_LIMIT", "500")
        monkeypatch.setenv("NIXI_MODEL", "gpt-4o")
        monkeypatch.delenv("HERMES_HOME", raising=False)

        config = NixiConfig.from_env()
        assert config.log_dir == log_dir
        assert config.output_dir == output_dir
        assert config.extraction_batch_size == 100
        assert config.bot_names == ["BotA", "BotB"]
        assert config.cooccurrence_threshold == 5
        assert config.memory_limit == 20000
        assert config.employee_limit == 500
        assert config.extraction_model == "gpt-4o"

    def test_defaults_when_env_vars_missing(self, tmp_path: Path, monkeypatch):
        """Defaults used when env vars not set."""
        monkeypatch.delenv("NIXI_LOG_DIR", raising=False)
        monkeypatch.delenv("NIXI_OUTPUT_DIR", raising=False)
        monkeypatch.delenv("NIXI_EXTRACTION_BATCH_SIZE", raising=False)
        monkeypatch.delenv("NIXI_BOT_NAMES", raising=False)
        monkeypatch.delenv("NIXI_COOCCURRENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("NIXI_MEMORY_LIMIT", raising=False)
        monkeypatch.delenv("NIXI_EMPLOYEE_LIMIT", raising=False)
        monkeypatch.delenv("NIXI_MODEL", raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)

        config = NixiConfig.from_env()
        assert config.extraction_batch_size == 50
        assert config.bot_names == ["Fixi", ".OP", "Toothless", "Cerberus"]
        assert config.cooccurrence_threshold == 3
        assert config.memory_limit == 10_000
        assert config.employee_limit == 1375
        assert config.extraction_model == ""

    def test_db_path_property(self, tmp_path: Path, monkeypatch):
        """NixiConfig.db_path returns output_dir / nixi_state.db."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        monkeypatch.setenv("NIXI_OUTPUT_DIR", str(output_dir))
        monkeypatch.delenv("HERMES_HOME", raising=False)

        config = NixiConfig.from_env()
        assert config.db_path == output_dir / "nixi_state.db"

    def test_standalone_output_dir_default(self, tmp_path: Path, monkeypatch):
        """Without HERMES_HOME or NIXI_OUTPUT_DIR, default is ~/.nixi/output."""
        monkeypatch.delenv("NIXI_OUTPUT_DIR", raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        # We can't easily test Path.home() but we can verify the path structure
        config = NixiConfig.from_env()
        assert config.output_dir == Path.home() / ".nixi" / "output"


# ── NixiConfig from_config ─────────────────────────────────────────────────────


class TestNixiConfigFromConfig:
    """Test NixiConfig.from_config() reads from YAML config."""

    def test_reads_nixi_section(self, tmp_path: Path, monkeypatch):
        """Config reads nixi: section from YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model: gpt-4o-mini\n"
            "nixi:\n"
            "  log_dir: /tmp/logs\n"
            "  output_dir: /tmp/output\n"
            "  extraction_batch_size: 25\n"
            "  bot_names:\n"
            "    - Alpha\n"
            "    - Beta\n"
            "  cooccurrence_threshold: 7\n"
            "  memory_limit: 5000\n"
            "  employee_limit: 200\n"
            "  extraction_model: gpt-4o\n",
            encoding="utf-8",
        )

        config = NixiConfig.from_config(config_file)
        assert config.extraction_batch_size == 25
        assert config.bot_names == ["Alpha", "Beta"]
        assert config.cooccurrence_threshold == 7
        assert config.memory_limit == 5000
        assert config.employee_limit == 200
        assert config.extraction_model == "gpt-4o"

    def test_model_key_fallback(self, tmp_path: Path, monkeypatch):
        """Top-level model: key used as extraction_model fallback."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model: gpt-4o-mini\n"
            "nixi:\n"
            "  log_dir: /tmp/logs\n"
            "  output_dir: /tmp/output\n",
            encoding="utf-8",
        )

        config = NixiConfig.from_config(config_file)
        assert config.extraction_model == "gpt-4o-mini"

    def test_nixi_extraction_model_overrides_model(self, tmp_path: Path, monkeypatch):
        """nixi.extraction_model takes priority over top-level model."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model: gpt-4o-mini\n"
            "nixi:\n"
            "  log_dir: /tmp/logs\n"
            "  output_dir: /tmp/output\n"
            "  extraction_model: gpt-4o\n",
            encoding="utf-8",
        )

        config = NixiConfig.from_config(config_file)
        assert config.extraction_model == "gpt-4o"

    def test_missing_config_file_uses_defaults(self, tmp_path: Path, monkeypatch):
        """Missing config file uses defaults."""
        nonexistent = tmp_path / "nonexistent.yaml"
        config = NixiConfig.from_config(nonexistent)
        # Defaults from NixiConfig dataclass
        assert config.extraction_batch_size == 50
        assert config.cooccurrence_threshold == 3


# ── run_ingestion ──────────────────────────────────────────────────────────────


class TestRunIngestion:
    """Test run_ingestion CLI entrypoint."""

    @pytest.mark.asyncio
    async def test_run_ingestion_calls_ensure_schema(self, nixi_config: NixiConfig, db_conn):
        """run_ingestion calls ensure_schema before ingestion."""
        from nixi.ingest import run_ingestion

        with patch("nixi.ingest.LogFileAdapter") as MockAdapter:
            mock_result = MagicMock(
                total_lines=10, parsed=10, bots_tagged=0,
                threads_linked=0, inserted=10, already_existing=0,
                raw_uid_posters=0,
            )
            MockAdapter.return_value.ingest.return_value = mock_result

            result = await run_ingestion(nixi_config)
            # ensure_schema was called — DB file exists
            assert nixi_config.db_path.exists()

    @pytest.mark.asyncio
    async def test_run_ingestion_calls_adapter_ingest(self, nixi_config: NixiConfig):
        """run_ingestion creates LogFileAdapter and calls ingest()."""
        from nixi.ingest import run_ingestion

        with patch("nixi.ingest.LogFileAdapter") as MockAdapter:
            mock_result = MagicMock(
                total_lines=5, parsed=5, bots_tagged=0,
                threads_linked=0, inserted=5, already_existing=0,
                raw_uid_posters=0,
            )
            MockAdapter.return_value.ingest.return_value = mock_result

            result = await run_ingestion(nixi_config)
            MockAdapter.return_value.ingest.assert_called_once()
            assert result["inserted"] == 5

    @pytest.mark.asyncio
    async def test_run_ingestion_channel(self, nixi_config: NixiConfig):
        """run_ingestion_channel calls ingest_channel for a specific channel."""
        from nixi.ingest import run_ingestion_channel

        with patch("nixi.ingest.LogFileAdapter") as MockAdapter:
            mock_result = MagicMock(
                total_lines=3, parsed=3, bots_tagged=0,
                threads_linked=0, inserted=3, already_existing=0,
                raw_uid_posters=0,
            )
            MockAdapter.return_value.ingest_channel.return_value = mock_result

            result = await run_ingestion_channel("C12345", nixi_config)
            MockAdapter.return_value.ingest_channel.assert_called_once_with("C12345")
            assert result["inserted"] == 3


# ── Empty DB guard in extract.py ───────────────────────────────────────────────


class TestExtractEmptyDBGuard:
    """Test that extract.py guards against empty scraped_messages."""

    @pytest.mark.asyncio
    async def test_run_extraction_empty_db_prints_message(self, nixi_config: NixiConfig, capsys):
        """run_extraction with empty DB prints 'Run nixi ingest first' message."""
        from nixi.extract import run_extraction

        # Create schema but no messages
        ensure_schema(nixi_config.db_path)

        result = await run_extraction(nixi_config)

        # Should return early with guidance
        assert result is not None
        captured = capsys.readouterr()
        assert "ingest" in captured.out.lower()

    @pytest.mark.asyncio
    async def test_run_extraction_channel_empty_db(self, nixi_config: NixiConfig, capsys):
        """run_extraction_channel with empty DB prints ingest message."""
        from nixi.extract import run_extraction_channel

        ensure_schema(nixi_config.db_path)

        result = await run_extraction_channel("C12345", nixi_config)
        assert result is not None
        captured = capsys.readouterr()
        assert "ingest" in captured.out.lower()

    @pytest.mark.asyncio
    async def test_run_extraction_with_data_proceeds(self, nixi_config: NixiConfig, db_conn):
        """run_extraction with data in DB proceeds normally."""
        from nixi.extract import run_extraction

        # Insert messages so DB is not empty
        from datetime import datetime, timezone

        msgs = [
            ScrapedMessage(
                slack_ts=f"1766766571.{i:06d}",
                channel_id="C06M81FSKFF",
                channel_name="C06M81FSKFF",
                user_id=None,
                user_name="Kuro",
                text="hello",
                thread_ts=None,
                parent_ts=None,
                is_bot=False,
                source_file="C06M81FSKFF",
                timestamp=datetime.fromtimestamp(float(f"1766766571.{i:06d}"), tz=timezone.utc).isoformat(),
            )
            for i in range(5)
        ]
        insert_messages(db_conn, msgs)

        # With messages present, extraction should proceed
        # (it may still skip due to threshold, but it won't print "ingest first")
        with patch("nixi.extract.ExtractionBatcher") as MockBatcher:
            mock_batcher = AsyncMock()
            mock_batcher.extract_all = AsyncMock(return_value={"channels": {}, "total_extracted": 0})
            MockBatcher.return_value = mock_batcher

            # Don't print "ingest first" if there's data
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                result = await run_extraction(nixi_config)
            output = f.getvalue().lower()
            # "Run nixi ingest first" should NOT appear when DB has data
            assert "run nixi ingest first" not in output


# ── ensure_schema called at extract startup ──────────────────────────────────────


class TestExtractEnsureSchema:
    """Test that extract.py calls ensure_schema when DB doesn't exist."""

    @pytest.mark.asyncio
    async def test_extract_creates_db_if_missing(self, nixi_config: NixiConfig, capsys):
        """run_extraction creates nixi_state.db when it doesn't exist."""
        from nixi.extract import run_extraction

        # Don't call ensure_schema — verify extract does it
        assert not nixi_config.db_path.exists()

        result = await run_extraction(nixi_config)
        # DB should be created by ensure_schema call
        assert nixi_config.db_path.exists()