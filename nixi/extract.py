"""CLI extract entrypoint for the nixi extraction pipeline.

Provides async entrypoints for:
- run_extraction: Process all channels with unprocessed messages
- run_extraction_channel: Process a single channel

Both call ensure_schema() at startup so the DB is created if missing.
If the DB exists but has 0 rows in scraped_messages, prints a message
telling the user to run `nixi ingest` first and exits gracefully.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nixi.config import NixiConfig
from nixi.db import ensure_schema, get_connection
from nixi.extraction.batch import ExtractionBatcher, LLMClient

logger = logging.getLogger(__name__)


def _check_db_populated(conn, config: NixiConfig) -> bool:
    """Check whether nixi_state.db has any scraped messages.

    Returns True if rows exist, False otherwise. Prints guidance
    message to stdout when DB is empty.
    """
    cursor = conn.execute("SELECT COUNT(*) FROM scraped_messages")
    count = cursor.fetchone()["COUNT(*)"]
    if count == 0:
        print("No scraped messages found in database. Run `nixi ingest` first to load Slack logs.")
        return False
    return True


async def run_extraction(config: NixiConfig | None = None) -> dict[str, Any]:
    """Run extraction across all channels with unprocessed messages.

    Creates DB schema if needed. If DB has no scraped_messages rows,
    prints a message and returns guidance dict.

    Args:
        config: NixiConfig with extraction settings. If None, loaded from
            hermes config or env vars.

    Returns:
        Dict with extraction summary or guidance if DB is empty.
    """
    if config is None:
        try:
            config = NixiConfig.from_config()
        except Exception:
            config = NixiConfig.from_env()

    # Ensure DB schema exists (creates tables if missing)
    ensure_schema(config.db_path)
    conn = get_connection(config.db_path)

    try:
        if not _check_db_populated(conn, config):
            return {"status": "empty_db", "message": "Run nixi ingest first"}

        llm = LLMClient(config)
        batcher = ExtractionBatcher(config, conn, llm)
        result = await batcher.extract_all()
        return result
    finally:
        conn.close()


async def run_extraction_channel(
    channel_id: str,
    config: NixiConfig | None = None,
) -> dict[str, Any]:
    """Run extraction for a single channel.

    Creates DB schema if needed. If DB has no scraped_messages rows,
    prints a message and returns guidance dict.

    Args:
        channel_id: Slack channel ID to extract.
        config: NixiConfig with extraction settings. If None, loaded from
            hermes config or env vars.

    Returns:
        Dict with extraction results or guidance if DB is empty.
    """
    if config is None:
        try:
            config = NixiConfig.from_config()
        except Exception:
            config = NixiConfig.from_env()

    # Ensure DB schema exists (creates tables if missing)
    ensure_schema(config.db_path)
    conn = get_connection(config.db_path)

    try:
        if not _check_db_populated(conn, config):
            return {"status": "empty_db", "message": "Run nixi ingest first"}

        llm = LLMClient(config)
        batcher = ExtractionBatcher(config, conn, llm)
        result = await batcher.extract_channel(channel_id)
        return result
    finally:
        conn.close()