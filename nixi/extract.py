"""CLI extract entrypoint for the nixi extraction pipeline.

Provides async entrypoints for:
- run_extraction: Process all channels with unprocessed messages
- run_extraction_channel: Process a single channel
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nixi.config import NixiConfig
from nixi.db import ensure_schema, get_connection
from nixi.extraction.batch import ExtractionBatcher, LLMClient

logger = logging.getLogger(__name__)


async def run_extraction(config: NixiConfig) -> dict[str, Any]:
    """Run extraction across all channels with unprocessed messages.

    Loads config, connects to DB, creates LLM client, instantiates
    ExtractionBatcher, and calls extract_all().

    Args:
        config: NixiConfig with extraction settings.

    Returns:
        Dict with extraction summary from ExtractionBatcher.extract_all().
    """
    ensure_schema(config.db_path)
    conn = get_connection(config.db_path)

    try:
        llm = LLMClient(config)
        batcher = ExtractionBatcher(config, conn, llm)
        result = await batcher.extract_all()
        return result
    finally:
        conn.close()


async def run_extraction_channel(
    channel_id: str,
    config: NixiConfig,
) -> dict[str, Any]:
    """Run extraction for a single channel.

    Args:
        channel_id: Slack channel ID to extract.
        config: NixiConfig with extraction settings.

    Returns:
        Dict with extraction results for the specified channel.
    """
    ensure_schema(config.db_path)
    conn = get_connection(config.db_path)

    try:
        llm = LLMClient(config)
        batcher = ExtractionBatcher(config, conn, llm)
        result = await batcher.extract_channel(channel_id)
        return result
    finally:
        conn.close()