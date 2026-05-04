"""Nixi CLI entry point.

Subcommands:
    ingest   — Walk log_dir, parse Slack logs, insert into nixi_state.db
    extract  — Run LLM extraction on unprocessed messages
    run      — Run both ingest + extract sequentially
"""

import asyncio
import os
from pathlib import Path

import click
from rich.console import Console

from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import get_hermes_home
from nixi.config import NixiConfig
from nixi.discovery import discover_hermes_home

console = Console()


@click.group()
@click.option(
    "--hermes-home",
    type=click.Path(),
    default=None,
    help="Path to HERMES_HOME directory (overrides env var and auto-discovery)",
)
@click.pass_context
def main(ctx: click.Context, hermes_home: str | None) -> None:
    """Nixi — Slack log extraction pipeline."""
    # Resolve HERMES_HOME using priority chain:
    # 1. --hermes-home flag (if provided)
    # 2. os.environ.get("HERMES_HOME") (existing env var)
    # 3. discover_hermes_home() (CWD-walk discovery)
    # 4. get_hermes_home() (defaults to ~/.hermes)
    if hermes_home:
        resolved = Path(hermes_home)
    elif os.environ.get("HERMES_HOME"):
        resolved = Path(os.environ["HERMES_HOME"])
    else:
        resolved = discover_hermes_home() or get_hermes_home()

    os.environ["HERMES_HOME"] = str(resolved)

    # Load API keys from both HERMES_HOME/.env and project root .env
    project_env = Path(__file__).resolve().parents[1] / ".env"
    load_hermes_dotenv(hermes_home=resolved, project_env=project_env)


@main.command()
@click.option("--log-dir", type=click.Path(), default=None, help="Path to slack_logs directory")
@click.option("--output-dir", type=click.Path(), default=None, help="Path to output directory")
@click.option("--force", is_flag=True, default=False, help="Re-parse already-ingested data")
@click.option("--channel", type=str, default=None, help="Single channel ID to ingest")
def ingest(
    log_dir: str | None,
    output_dir: str | None,
    force: bool,
    channel: str | None,
) -> None:
    """Ingest Slack logs into nixi_state.db.

    Walks the log_dir, parses all Slack log files, and inserts messages
    into the database. Creates DB schema on first run.

    Use --force to re-parse data that has already been ingested.
    Use --channel to limit ingestion to a single channel.
    """
    from nixi.ingest import run_ingestion, run_ingestion_channel

    if channel:
        console.print(f"[bold blue]nixi ingest[/] — channel {channel}")
    else:
        console.print("[bold blue]nixi ingest[/] — walking log directory")

    if log_dir or output_dir:
        config = NixiConfig(
            log_dir=Path(log_dir) if log_dir else Path(),
            output_dir=Path(output_dir) if output_dir else Path.home() / ".nixi" / "output",
        )
    else:
        config = None

    if channel:
        result = asyncio.run(run_ingestion_channel(channel, config))
    else:
        result = asyncio.run(run_ingestion(config, force=force))

    if result:
        console.print(f"  [green]✓[/] Inserted: [bold]{result.get('inserted', 0)}[/] new messages")
        console.print(f"  Already existing: {result.get('already_existing', 0)}")
        console.print(f"  Parsed: {result.get('parsed', 0)} | Bots tagged: {result.get('bots_tagged', 0)} | "
                      f"Threads linked: {result.get('threads_linked', 0)} | Raw UID posters: {result.get('raw_uid_posters', 0)}")


@main.command()
@click.option("--channel", type=str, default=None, help="Single channel ID to extract")
@click.option("--output-dir", type=click.Path(), default=None, help="Path to output directory")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be extracted without LLM calls")
def extract(
    channel: str | None,
    output_dir: str | None,
    dry_run: bool,
) -> None:
    """Run LLM extraction on unprocessed messages.

    Extracts organizational memory, employee info, and channel skills
    from messages in nixi_state.db that haven't been extracted yet.

    Use --channel to limit extraction to a single channel.
    Use --dry-run to show what would be extracted without making LLM calls.
    """
    from nixi.extract import run_extraction, run_extraction_channel

    label = f"channel {channel}" if channel else "all channels"
    if dry_run:
        console.print(f"[bold blue]nixi extract[/] — {label} [dim](dry-run)[/]")
    else:
        console.print(f"[bold blue]nixi extract[/] — {label}")

    if output_dir:
        config = NixiConfig(
            log_dir=Path(),
            output_dir=Path(output_dir),
        )
    else:
        config = None

    if channel:
        result = asyncio.run(run_extraction_channel(channel, config, dry_run=dry_run))
    else:
        result = asyncio.run(run_extraction(config, dry_run=dry_run))

    if result:
        if result.get("status") == "empty_db":
            console.print("[yellow]⚠ No scraped messages found. Run [bold]nixi ingest[/] first.[/]")
        elif result.get("status") == "dry_run":
            console.print("[dim]Dry run complete — no LLM calls made.[/]")
        else:
            extracted = result.get("total_extracted", 0)
            skipped = result.get("total_skipped", 0)
            console.print(f"  [green]✓[/] Extracted: [bold]{extracted}[/] channels | Skipped: {skipped}")


@main.command()
@click.option("--log-dir", type=click.Path(), default=None, help="Path to slack_logs directory")
@click.option("--output-dir", type=click.Path(), default=None, help="Path to output directory")
@click.option("--force", is_flag=True, default=False, help="Re-ingest already-parsed data")
@click.option("--channel", type=str, default=None, help="Single channel ID to process")
def run(
    log_dir: str | None,
    output_dir: str | None,
    force: bool,
    channel: str | None,
) -> None:
    """Run both ingest + extract sequentially.

    Creates DB schema if needed, then runs ingestion followed by
    extraction. Use --force to re-ingest existing data.
    Use --channel to process a single channel.
    """
    from nixi.worker import run as worker_run
    from nixi.worker import run_channel as worker_run_channel

    label = f"channel {channel}" if channel else "all channels"
    console.print(f"[bold blue]nixi run[/] — {label}")

    if log_dir or output_dir:
        config = NixiConfig(
            log_dir=Path(log_dir) if log_dir else Path(),
            output_dir=Path(output_dir) if output_dir else Path.home() / ".nixi" / "output",
        )
    else:
        config = None

    if channel:
        result = asyncio.run(worker_run_channel(channel, config, force=force))
    else:
        result = asyncio.run(worker_run(config, force=force))

    if result:
        ingest = result.get("ingest", {})
        extract = result.get("extract", {})
        console.print(f"  [green]✓[/] Ingest: [bold]{ingest.get('inserted', 0)}[/] new | "
                      f"{ingest.get('already_existing', 0)} existing")
        if extract.get("status") == "empty_db":
            console.print("[yellow]⚠ No data to extract.[/]")
        else:
            console.print(f"  [green]✓[/] Extract: [bold]{extract.get('total_extracted', 0)}[/] channels")


if __name__ == "__main__":
    main()