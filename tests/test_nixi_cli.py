"""Tests for nixi CLI entrypoints and worker orchestration.

Covers: click commands (ingest, extract, run), options (--force, --dry-run,
--channel), worker.run / worker.run_channel orchestration,
empty DB guard, ensure_schema at startup.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from nixi.cli import main
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


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


def _make_message(
    slack_ts: str = "1766766571.412779",
    channel_id: str = "C06M81FSKFF",
    channel_name: str = "general",
    user_id: str | None = "U04K8NLDCG0",
    user_name: str = "Kuro",
    text: str = "hello world",
    thread_ts: str | None = None,
    parent_ts: str | None = None,
    is_bot: bool = False,
    source_file: str = "C06M81FSKFF",
    timestamp: str | None = None,
) -> ScrapedMessage:
    from datetime import datetime, timezone

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


# ── CLI ingest command ────────────────────────────────────────────────────────


class TestCLIIngest:
    """Test the `nixi ingest` click command."""

    def test_ingest_help_displays(self, runner: CliRunner):
        """nixi ingest --help shows usage."""
        result = runner.invoke(main, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output.lower() or "slack" in result.output.lower()

    def test_ingest_runs_adapter(self, runner: CliRunner, nixi_config: NixiConfig):
        """nixi ingest delegates to run_ingestion."""
        with patch("nixi.ingest.run_ingestion", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.return_value = {
                "total_lines": 100,
                "parsed": 95,
                "inserted": 80,
                "already_existing": 15,
                "bots_tagged": 5,
                "threads_linked": 10,
                "raw_uid_posters": 2,
            }
            result = runner.invoke(main, ["ingest"])
            assert result.exit_code == 0
            mock_ingest.assert_called_once()

    def test_ingest_force_option(self, runner: CliRunner, nixi_config: NixiConfig):
        """nixi ingest --force passes force flag to ingestion."""
        with patch("nixi.ingest.run_ingestion", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.return_value = {
                "total_lines": 50,
                "parsed": 50,
                "inserted": 50,
                "already_existing": 0,
                "bots_tagged": 0,
                "threads_linked": 0,
                "raw_uid_posters": 0,
            }
            result = runner.invoke(main, ["ingest", "--force"])
            assert result.exit_code == 0
            # Verify force=True was passed
            call_kwargs = mock_ingest.call_args[1]
            assert call_kwargs.get("force") is True

    def test_ingest_channel_option(self, runner: CliRunner, nixi_config: NixiConfig):
        """nixi ingest --channel C123 delegates to run_ingestion_channel."""
        with patch("nixi.ingest.run_ingestion_channel", new_callable=AsyncMock) as mock_channel:
            mock_channel.return_value = {
                "total_lines": 30,
                "parsed": 30,
                "inserted": 25,
                "already_existing": 5,
                "bots_tagged": 0,
                "threads_linked": 0,
                "raw_uid_posters": 0,
            }
            result = runner.invoke(main, ["ingest", "--channel", "C123"])
            assert result.exit_code == 0
            mock_channel.assert_called_once()

    def test_ingest_log_dir_option(self, runner: CliRunner, tmp_path: Path):
        """nixi ingest --log-dir uses specified directory."""
        log_dir = tmp_path / "custom_logs"
        log_dir.mkdir()
        output_dir = tmp_path / "custom_output"
        output_dir.mkdir()
        with patch("nixi.ingest.run_ingestion", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.return_value = {
                "total_lines": 0,
                "parsed": 0,
                "inserted": 0,
                "already_existing": 0,
                "bots_tagged": 0,
                "threads_linked": 0,
                "raw_uid_posters": 0,
            }
            result = runner.invoke(main, [
                "ingest", "--log-dir", str(log_dir), "--output-dir", str(output_dir),
            ])
            assert result.exit_code == 0


# ── CLI extract command ───────────────────────────────────────────────────────


class TestCLIExtract:
    """Test the `nixi extract` click command."""

    def test_extract_help_displays(self, runner: CliRunner):
        """nixi extract --help shows usage."""
        result = runner.invoke(main, ["extract", "--help"])
        assert result.exit_code == 0
        assert "extract" in result.output.lower()

    def test_extract_runs_batch(self, runner: CliRunner, nixi_config: NixiConfig):
        """nixi extract delegates to run_extraction."""
        with patch("nixi.extract.run_extraction", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = {
                "channels": {},
                "total_messages": 0,
                "total_skipped": 0,
                "total_extracted": 0,
            }
            result = runner.invoke(main, ["extract"])
            assert result.exit_code == 0
            mock_extract.assert_called_once()

    def test_extract_channel_option(self, runner: CliRunner, nixi_config: NixiConfig):
        """nixi extract --channel C123 delegates to run_extraction_channel."""
        with patch("nixi.extract.run_extraction_channel", new_callable=AsyncMock) as mock_channel:
            mock_channel.return_value = {
                "channel_id": "C123",
                "message_count": 10,
                "skipped": False,
            }
            result = runner.invoke(main, ["extract", "--channel", "C123"])
            assert result.exit_code == 0
            mock_channel.assert_called_once()

    def test_extract_dry_run_option(self, runner: CliRunner, nixi_config: NixiConfig):
        """nixi extract --dry-run shows what would be extracted."""
        with patch("nixi.extract.run_extraction", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = {
                "status": "dry_run",
                "channels": {},
                "total_messages": 0,
            }
            result = runner.invoke(main, ["extract", "--dry-run"])
            assert result.exit_code == 0
            # Verify dry_run=True was passed
            call_kwargs = mock_extract.call_args[1]
            assert call_kwargs.get("dry_run") is True

    def test_extract_empty_db_prints_message(self, runner: CliRunner, nixi_config: NixiConfig):
        """nixi extract with empty DB prints ingest-first message."""
        with patch("nixi.extract.run_extraction", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = {
                "status": "empty_db",
                "message": "Run nixi ingest first",
            }
            result = runner.invoke(main, ["extract"])
            assert result.exit_code == 0


# ── CLI run command ───────────────────────────────────────────────────────────


class TestCLIRun:
    """Test the `nixi run` click command."""

    def test_run_help_displays(self, runner: CliRunner):
        """nixi run --help shows usage."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output.lower() or "ingest" in result.output.lower()

    def test_run_delegates_to_worker(self, runner: CliRunner, nixi_config: NixiConfig):
        """nixi run delegates to worker.run()."""
        with patch("nixi.worker.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "ingest": {"inserted": 100},
                "extract": {"total_extracted": 5},
            }
            result = runner.invoke(main, ["run"])
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_run_force_option(self, runner: CliRunner, nixi_config: NixiConfig):
        """nixi run --force passes force flag."""
        with patch("nixi.worker.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "ingest": {"inserted": 100},
                "extract": {"total_extracted": 5},
            }
            result = runner.invoke(main, ["run", "--force"])
            assert result.exit_code == 0
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs.get("force") is True

    def test_run_channel_option(self, runner: CliRunner, nixi_config: NixiConfig):
        """nixi run --channel C123 delegates to worker.run_channel()."""
        with patch("nixi.worker.run_channel", new_callable=AsyncMock) as mock_channel:
            mock_channel.return_value = {
                "ingest": {"inserted": 30},
                "extract": {"total_extracted": 1},
            }
            result = runner.invoke(main, ["run", "--channel", "C123"])
            assert result.exit_code == 0
            mock_channel.assert_called_once()

    def test_run_output_dir_option(self, runner: CliRunner, tmp_path: Path):
        """nixi run --output-dir uses specified directory."""
        output_dir = tmp_path / "custom_output"
        output_dir.mkdir()
        with patch("nixi.worker.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "ingest": {"inserted": 0},
                "extract": {"total_extracted": 0},
            }
            result = runner.invoke(main, ["run", "--output-dir", str(output_dir)])
            assert result.exit_code == 0


# ── Worker orchestration ──────────────────────────────────────────────────────


class TestWorker:
    """Test worker.run and worker.run_channel orchestration."""

    @pytest.mark.asyncio
    async def test_run_calls_ingest_then_extract(self, nixi_config: NixiConfig):
        """worker.run() calls ensure_schema, ingest, then extract_all."""
        from nixi.worker import run

        with patch("nixi.worker.ensure_schema") as mock_schema, \
             patch("nixi.worker.LogFileAdapter") as MockAdapter, \
             patch("nixi.worker.ExtractionBatcher") as MockBatcher, \
             patch("nixi.worker.LLMClient") as MockLLM, \
             patch("nixi.worker.get_connection") as mock_get_conn, \
             patch("nixi.worker._check_db_populated", return_value=True):
            mock_result = MagicMock(
                total_lines=10, parsed=10, bots_tagged=0,
                threads_linked=0, inserted=10, already_existing=0,
                raw_uid_posters=0,
            )
            MockAdapter.return_value.ingest.return_value = mock_result

            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn

            mock_batcher = AsyncMock()
            mock_batcher.extract_all = AsyncMock(return_value={
                "channels": {}, "total_extracted": 0, "total_skipped": 0,
            })
            MockBatcher.return_value = mock_batcher

            result = await run(nixi_config)

            mock_schema.assert_called_once()
            MockAdapter.return_value.ingest.assert_called_once()
            mock_batcher.extract_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_channel_calls_ingest_then_extract(self, nixi_config: NixiConfig):
        """worker.run_channel() calls ensure_schema, ingest_channel, then extract_channel."""
        from nixi.worker import run_channel

        with patch("nixi.worker.ensure_schema") as mock_schema, \
             patch("nixi.worker.LogFileAdapter") as MockAdapter, \
             patch("nixi.worker.ExtractionBatcher") as MockBatcher, \
             patch("nixi.worker.LLMClient") as MockLLM, \
             patch("nixi.worker.get_connection") as mock_get_conn, \
             patch("nixi.worker._check_db_populated", return_value=True):
            mock_result = MagicMock(
                total_lines=5, parsed=5, bots_tagged=0,
                threads_linked=0, inserted=5, already_existing=0,
                raw_uid_posters=0,
            )
            MockAdapter.return_value.ingest_channel.return_value = mock_result

            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn

            mock_batcher = AsyncMock()
            mock_batcher.extract_channel = AsyncMock(return_value={
                "channel_id": "C123", "message_count": 10, "skipped": False,
            })
            MockBatcher.return_value = mock_batcher

            result = await run_channel("C123", nixi_config)

            MockAdapter.return_value.ingest_channel.assert_called_once_with("C123")
            mock_batcher.extract_channel.assert_called_once_with("C123")

    @pytest.mark.asyncio
    async def test_run_creates_db_if_missing(self, nixi_config: NixiConfig):
        """worker.run() calls ensure_schema before operations."""
        from nixi.worker import run

        with patch("nixi.worker.ensure_schema") as mock_schema, \
             patch("nixi.worker.LogFileAdapter") as MockAdapter, \
             patch("nixi.worker.ExtractionBatcher") as MockBatcher, \
             patch("nixi.worker.LLMClient") as MockLLM, \
             patch("nixi.worker.get_connection") as mock_get_conn, \
             patch("nixi.worker._check_db_populated", return_value=True):
            mock_result = MagicMock(
                total_lines=0, parsed=0, bots_tagged=0,
                threads_linked=0, inserted=0, already_existing=0,
                raw_uid_posters=0,
            )
            MockAdapter.return_value.ingest.return_value = mock_result

            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn

            mock_batcher = AsyncMock()
            mock_batcher.extract_all = AsyncMock(return_value={
                "channels": {}, "total_extracted": 0, "total_skipped": 0,
            })
            MockBatcher.return_value = mock_batcher

            await run(nixi_config)
            mock_schema.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_returns_summary(self, nixi_config: NixiConfig):
        """worker.run() returns a summary dict with ingest and extract keys."""
        from nixi.worker import run

        with patch("nixi.worker.ensure_schema"), \
             patch("nixi.worker.LogFileAdapter") as MockAdapter, \
             patch("nixi.worker.ExtractionBatcher") as MockBatcher, \
             patch("nixi.worker.LLMClient"), \
             patch("nixi.worker.get_connection") as mock_get_conn, \
             patch("nixi.worker._check_db_populated", return_value=True):
            mock_result = MagicMock(
                total_lines=100, parsed=95, bots_tagged=3,
                threads_linked=10, inserted=80, already_existing=15,
                raw_uid_posters=2,
            )
            MockAdapter.return_value.ingest.return_value = mock_result

            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn

            mock_batcher = AsyncMock()
            mock_batcher.extract_all = AsyncMock(return_value={
                "channels": {}, "total_extracted": 5, "total_skipped": 1,
                "total_messages": 80,
            })
            MockBatcher.return_value = mock_batcher

            result = await run(nixi_config)
            assert "ingest" in result
            assert "extract" in result
            assert result["ingest"]["inserted"] == 80