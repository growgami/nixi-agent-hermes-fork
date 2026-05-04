"""Worker orchestration for the nixi extraction pipeline.

Provides async entrypoints that combine ingest + extract:
- run(config): Full pipeline — ingest then extract
- run_channel(channel_id, config): Single channel pipeline
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import get_hermes_home

from nixi.adapter import LogFileAdapter
from nixi.config import NixiConfig
from nixi.db import ensure_schema, get_connection
from nixi.discovery import discover_hermes_home
from nixi.extraction.batch import ExtractionBatcher, LLMClient

logger = logging.getLogger(__name__)


def _ensure_hermes_env() -> Path:
    """Resolve HERMES_HOME and load dotenv for programmatic entry points.

    If HERMES_HOME is already set in the environment, keeps it.
    Otherwise resolves via discover_hermes_home() or get_hermes_home() fallback.
    Always calls load_hermes_dotenv() with both hermes_home and project_env.

    Returns:
        The resolved HERMES_HOME path.
    """
    if os.environ.get("HERMES_HOME"):
        resolved = Path(os.environ["HERMES_HOME"])
    else:
        resolved = discover_hermes_home() or get_hermes_home()
        os.environ["HERMES_HOME"] = str(resolved)

    project_env = Path(__file__).resolve().parents[1] / ".env"
    load_hermes_dotenv(hermes_home=resolved, project_env=project_env)
    return resolved


def _check_db_populated(conn) -> bool:
    """Check whether nixi_state.db has any scraped messages.

    Returns True if rows exist, False otherwise.
    """
    cursor = conn.execute("SELECT COUNT(*) FROM scraped_messages")
    count = cursor.fetchone()["COUNT(*)"]
    return count > 0


async def run(
    config: NixiConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run full pipeline: ingest then extract.

    1. Calls ensure_schema() if DB doesn't exist
    2. Runs LogFileAdapter.ingest()
    3. Runs ExtractionBatcher.extract_all()
    4. Prints summary

    Args:
        config: NixiConfig with extraction settings. If None, loaded from
            hermes config or env vars.
        force: If True, re-parse already-ingested data.

    Returns:
        Dict with ingest and extract summaries.
    """
    _ensure_hermes_env()

    if config is None:
        try:
            config = NixiConfig.from_config()
        except Exception:
            config = NixiConfig.from_env()

    # Ensure DB schema exists
    ensure_schema(config.db_path)

    # Phase 1: Ingest
    adapter = LogFileAdapter(config=config)
    ingest_result = adapter.ingest(force=force)

    ingest_summary = {
        "total_lines": ingest_result.total_lines,
        "parsed": ingest_result.parsed,
        "bots_tagged": ingest_result.bots_tagged,
        "threads_linked": ingest_result.threads_linked,
        "inserted": ingest_result.inserted,
        "already_existing": ingest_result.already_existing,
        "raw_uid_posters": ingest_result.raw_uid_posters,
    }

    logger.info(
        "Ingest complete: %d parsed, %d inserted, %d already_existing",
        ingest_result.parsed,
        ingest_result.inserted,
        ingest_result.already_existing,
    )

    print(f"\n[ingest] {ingest_result.inserted} new messages inserted "
          f"({ingest_result.already_existing} already existed)")
    print(f"  Total lines: {ingest_result.total_lines}")
    print(f"  Parsed: {ingest_result.parsed}")
    print(f"  Bots tagged: {ingest_result.bots_tagged}")
    print(f"  Threads linked: {ingest_result.threads_linked}")
    print(f"  Raw UID posters: {ingest_result.raw_uid_posters}")

    # Phase 2: Extract
    conn = get_connection(config.db_path)
    try:
        if not _check_db_populated(conn):
            print("\nNo scraped messages found. Run `nixi ingest` first to load Slack logs.")
            return {
                "ingest": ingest_summary,
                "extract": {"status": "empty_db", "message": "Run nixi ingest first"},
            }

        llm = LLMClient(config)
        batcher = ExtractionBatcher(config, conn, llm)
        extract_result = await batcher.extract_all()
        extract_result["ingest_summary"] = ingest_summary

        print(f"\n[extract] {extract_result.get('total_extracted', 0)} channels extracted, "
              f"{extract_result.get('total_skipped', 0)} skipped")

        return {"ingest": ingest_summary, "extract": extract_result}
    finally:
        conn.close()


async def run_channel(
    channel_id: str,
    config: NixiConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run pipeline for a single channel: ingest then extract.

    Args:
        channel_id: Slack channel ID to process.
        config: NixiConfig with extraction settings. If None, loaded from
            hermes config or env vars.
        force: If True, re-ingest already-ingested data.

    Returns:
        Dict with ingest and extract summaries for the channel.
    """
    _ensure_hermes_env()

    if config is None:
        try:
            config = NixiConfig.from_config()
        except Exception:
            config = NixiConfig.from_env()

    # Ensure DB schema exists
    ensure_schema(config.db_path)

    # Phase 1: Ingest channel
    adapter = LogFileAdapter(config=config)
    ingest_result = adapter.ingest_channel(channel_id)

    ingest_summary = {
        "total_lines": ingest_result.total_lines,
        "parsed": ingest_result.parsed,
        "bots_tagged": ingest_result.bots_tagged,
        "threads_linked": ingest_result.threads_linked,
        "inserted": ingest_result.inserted,
        "already_existing": ingest_result.already_existing,
        "raw_uid_posters": ingest_result.raw_uid_posters,
    }

    logger.info(
        "Channel %s ingest: %d parsed, %d inserted",
        channel_id,
        ingest_result.parsed,
        ingest_result.inserted,
    )

    print(f"\n[ingest] Channel {channel_id}: {ingest_result.inserted} new messages inserted "
          f"({ingest_result.already_existing} already existed)")

    # Phase 2: Extract channel
    conn = get_connection(config.db_path)
    try:
        if not _check_db_populated(conn):
            print("\nNo scraped messages found. Run `nixi ingest` first to load Slack logs.")
            return {
                "ingest": ingest_summary,
                "extract": {"status": "empty_db", "message": "Run nixi ingest first"},
            }

        llm = LLMClient(config)
        batcher = ExtractionBatcher(config, conn, llm)
        extract_result = await batcher.extract_channel(channel_id)

        print(f"\n[extract] Channel {channel_id}: "
              f"{'skipped' if extract_result.get('skipped') else 'extracted'}")

        return {"ingest": ingest_summary, "extract": extract_result}
    finally:
        conn.close()