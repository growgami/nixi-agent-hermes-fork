"""Nixi CLI entry point.

Placeholder CLI — will be extended in later tasks with extraction subcommands.
"""

import click


@click.group()
def main() -> None:
    """Nixi — Slack log extraction pipeline."""


if __name__ == "__main__":
    main()