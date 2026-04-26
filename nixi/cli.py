"""Nixi CLI entry point.

Subcommands:
    ingest   — Walk log_dir, parse Slack logs, insert into nixi_state.db
    extract  — Run LLM extraction on unprocessed messages
"""

import asyncio
from pathlib import Path

import click

from nixi.config import NixiConfig


@click.group()
def main() -> None:
    """Nixi — Slack log extraction pipeline."""


@main.command()
@click.option("--log-dir", type=click.Path(), default=None, help="Path to slack_logs directory")
@click.option("--output-dir", type=click.Path(), default=None, help="Path to output directory")
def ingest(log_dir: str | None, output_dir: str | None) -> None:
    """Ingest Slack logs into nixi_state.db.

    Walks the log_dir, parses all Slack log files, and inserts messages
    into the database. Creates DB schema on first run.
    """
    from nixi.ingest import run_ingestion

    if log_dir or output_dir:
        config = NixiConfig(
            log_dir=Path(log_dir) if log_dir else Path(),
            output_dir=Path(output_dir) if output_dir else Path.home() / ".nixi" / "output",
        )
    else:
        config = None

    asyncio.run(run_ingestion(config))


@main.command()
@click.option("--channel", type=str, default=None, help="Single channel ID to extract")
@click.option("--output-dir", type=click.Path(), default=None, help="Path to output directory")
def extract(channel: str | None, output_dir: str | None) -> None:
    """Run LLM extraction on unprocessed messages.

    Extracts organizational memory, employee info, and channel skills
    from messages in nixi_state.db that haven't been extracted yet.
    """
    from nixi.extract import run_extraction, run_extraction_channel

    if output_dir:
        config = NixiConfig(
            log_dir=Path(),
            output_dir=Path(output_dir),
        )
    else:
        config = None

    if channel:
        asyncio.run(run_extraction_channel(channel, config))
    else:
        asyncio.run(run_extraction(config))


if __name__ == "__main__":
    main()