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
from nixi.db import (
    ensure_realtime_schema,
    ensure_schema,
    get_connection,
    get_realtime_unprocessed,
    get_realtime_unprocessed_channels,
    get_unprocessed,
    get_unprocessed_channels,
)
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


async def run_extraction(config: NixiConfig | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Run extraction across all channels with unprocessed messages.

    Creates DB schema if needed. If DB has no scraped_messages rows,
    prints a message and returns guidance dict.

    Args:
        config: NixiConfig with extraction settings. If None, loaded from
            hermes config or env vars.
        dry_run: If True, show what would be extracted without making LLM calls.

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

        if dry_run:
            # In dry-run mode, report what would be extracted without calling LLM
            channels = get_unprocessed_channels(conn)
            total_messages = 0
            channel_info: dict[str, Any] = {}
            for ch_id in channels:
                msgs = get_unprocessed(conn, ch_id, limit=1000)
                channel_info[ch_id] = {
                    "message_count": len(msgs),
                    "status": "would_extract" if len(msgs) >= 20 else "would_skip",
                }
                total_messages += len(msgs)

            print(f"\n[dry-run] {len(channels)} channels with unprocessed messages")
            print(f"  Total messages: {total_messages}")
            for ch_id, info in channel_info.items():
                status = info["status"]
                print(f"  {ch_id}: {info['message_count']} messages ({status})")

            return {
                "status": "dry_run",
                "channels": channel_info,
                "total_messages": total_messages,
            }

        llm = LLMClient(config)
        batcher = ExtractionBatcher(config, conn, llm)
        result = await batcher.extract_all()
        return result
    finally:
        conn.close()


async def run_extraction_channel(
    channel_id: str,
    config: NixiConfig | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run extraction for a single channel.

    Creates DB schema if needed. If DB has no scraped_messages rows,
    prints a message and returns guidance dict.

    Args:
        channel_id: Slack channel ID to extract.
        config: NixiConfig with extraction settings. If None, loaded from
            hermes config or env vars.
        dry_run: If True, show what would be extracted without making LLM calls.

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

        if dry_run:
            msgs = get_unprocessed(conn, channel_id, limit=1000)
            print(f"\n[dry-run] Channel {channel_id}: {len(msgs)} messages would be extracted")
            return {
                "status": "dry_run",
                "channel_id": channel_id,
                "message_count": len(msgs),
            }

        llm = LLMClient(config)
        batcher = ExtractionBatcher(config, conn, llm)
        result = await batcher.extract_channel(channel_id)
        return result
    finally:
        conn.close()


def _check_realtime_db_populated(conn, config: NixiConfig) -> bool:
    """Check whether nixi_state.db has any realtime messages.

    Returns True if rows exist, False otherwise. Prints guidance
    message to stdout when DB is empty.
    """
    cursor = conn.execute("SELECT COUNT(*) FROM realtime_messages")
    count = cursor.fetchone()["COUNT(*)"]
    if count == 0:
        print("No realtime messages found in database. Ensure the ingester is running and receiving events.")
        return False
    return True


async def run_extraction_realtime(config: NixiConfig | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Run extraction across all channels with unprocessed realtime messages.

    Creates realtime schema if needed. If DB has no realtime_messages rows,
    prints a message and returns guidance dict.

    Args:
        config: NixiConfig with extraction settings. If None, loaded from
            hermes config or env vars.
        dry_run: If True, show what would be extracted without making LLM calls.

    Returns:
        Dict with extraction summary or guidance if DB is empty.
    """
    if config is None:
        try:
            config = NixiConfig.from_config()
        except Exception:
            config = NixiConfig.from_env()

    # Ensure both schemas exist (realtime + extraction_log)
    ensure_realtime_schema(config.db_path)
    ensure_schema(config.db_path)
    conn = get_connection(config.db_path)

    try:
        if not _check_realtime_db_populated(conn, config):
            return {"status": "empty_db", "message": "No realtime messages available"}

        if dry_run:
            channels = get_realtime_unprocessed_channels(conn)
            total_messages = 0
            channel_info: dict[str, Any] = {}
            for ch_id in channels:
                msgs = get_realtime_unprocessed(conn, ch_id, limit=1000)
                channel_info[ch_id] = {
                    "message_count": len(msgs),
                    "status": "would_extract" if len(msgs) >= 20 else "would_skip",
                }
                total_messages += len(msgs)

            print(f"\n[dry-run] {len(channels)} channels with unprocessed realtime messages")
            print(f"  Total messages: {total_messages}")
            for ch_id, info in channel_info.items():
                status = info["status"]
                print(f"  {ch_id}: {info['message_count']} messages ({status})")

            return {
                "status": "dry_run",
                "channels": channel_info,
                "total_messages": total_messages,
            }

        llm = LLMClient(config)
        batcher = ExtractionBatcher(config, conn, llm, source="realtime")
        result = await batcher.extract_all()
        return result
    finally:
        conn.close()


async def run_extraction_realtime_channel(
    channel_id: str,
    config: NixiConfig | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run extraction for a single channel from realtime messages.

    Args:
        channel_id: Slack channel ID to extract.
        config: NixiConfig with extraction settings. If None, loaded from
            hermes config or env vars.
        dry_run: If True, show what would be extracted without making LLM calls.

    Returns:
        Dict with extraction results or guidance if DB is empty.
    """
    if config is None:
        try:
            config = NixiConfig.from_config()
        except Exception:
            config = NixiConfig.from_env()

    # Ensure both schemas exist
    ensure_realtime_schema(config.db_path)
    ensure_schema(config.db_path)
    conn = get_connection(config.db_path)

    try:
        if not _check_realtime_db_populated(conn, config):
            return {"status": "empty_db", "message": "No realtime messages available"}

        if dry_run:
            msgs = get_realtime_unprocessed(conn, channel_id, limit=1000)
            print(f"\n[dry-run] Channel {channel_id}: {len(msgs)} realtime messages would be extracted")
            return {
                "status": "dry_run",
                "channel_id": channel_id,
                "message_count": len(msgs),
            }

        llm = LLMClient(config)
        batcher = ExtractionBatcher(config, conn, llm, source="realtime")
        result = await batcher.extract_channel(channel_id)
        return result
    finally:
        conn.close()