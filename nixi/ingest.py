"""CLI ingest entrypoint for the nixi extraction pipeline.

Provides async entrypoints for:
- run_ingestion: Process all log files and write to nixi_state.db
- run_ingestion_channel: Process a single channel's logs

Both call ensure_schema() at startup to create DB tables if needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nixi.adapter import LogFileAdapter
from nixi.config import NixiConfig
from nixi.db import ensure_schema

logger = logging.getLogger(__name__)


def _load_config() -> NixiConfig:
    """Load NixiConfig — try hermes config first, fall back to env vars."""
    try:
        return NixiConfig.from_config()
    except Exception:
        logger.info("Hermes config not available, using environment variables")
        return NixiConfig.from_env()


async def run_ingestion(config: NixiConfig | None = None, force: bool = False) -> dict[str, Any]:
    """Run full ingestion: walk log_dir, parse, insert into nixi_state.db.

    Creates DB schema if needed, then delegates to LogFileAdapter.ingest().

    Args:
        config: NixiConfig with extraction settings. If None, loaded from
            hermes config or env vars.
        force: If True, re-parse already-ingested data.

    Returns:
        Dict with ingestion summary (total_lines, parsed, inserted, etc.).
    """
    if config is None:
        config = _load_config()

    # Ensure DB schema exists before ingestion
    ensure_schema(config.db_path)

    adapter = LogFileAdapter(config=config)
    result = adapter.ingest(force=force)

    # Convert IngestionResult to dict for CLI output
    summary = {
        "total_lines": result.total_lines,
        "parsed": result.parsed,
        "bots_tagged": result.bots_tagged,
        "threads_linked": result.threads_linked,
        "inserted": result.inserted,
        "already_existing": result.already_existing,
        "raw_uid_posters": result.raw_uid_posters,
    }

    logger.info(
        "Ingestion complete: %d lines, %d parsed, %d inserted, %d already existing",
        result.total_lines,
        result.parsed,
        result.inserted,
        result.already_existing,
    )

    print(f"Ingestion complete: {result.inserted} new messages inserted "
          f"({result.already_existing} already existed)")
    print(f"  Total lines: {result.total_lines}")
    print(f"  Parsed: {result.parsed}")
    print(f"  Bots tagged: {result.bots_tagged}")
    print(f"  Threads linked: {result.threads_linked}")
    print(f"  Raw UID posters: {result.raw_uid_posters}")

    return summary


async def run_ingestion_channel(
    channel_id: str,
    config: NixiConfig | None = None,
) -> dict[str, Any]:
    """Run ingestion for a single channel.

    Creates DB schema if needed, then delegates to
    LogFileAdapter.ingest_channel().

    Args:
        channel_id: Slack channel ID to ingest.
        config: NixiConfig with extraction settings. If None, loaded from
            hermes config or env vars.

    Returns:
        Dict with ingestion summary for the channel.
    """
    if config is None:
        config = _load_config()

    # Ensure DB schema exists before ingestion
    ensure_schema(config.db_path)

    adapter = LogFileAdapter(config=config)
    result = adapter.ingest_channel(channel_id)

    summary = {
        "total_lines": result.total_lines,
        "parsed": result.parsed,
        "bots_tagged": result.bots_tagged,
        "threads_linked": result.threads_linked,
        "inserted": result.inserted,
        "already_existing": result.already_existing,
        "raw_uid_posters": result.raw_uid_posters,
    }

    logger.info(
        "Channel %s ingestion: %d parsed, %d inserted",
        channel_id,
        result.parsed,
        result.inserted,
    )

    print(f"Channel {channel_id}: {result.inserted} new messages inserted "
          f"({result.already_existing} already existed)")

    return summary