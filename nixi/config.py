"""Runtime configuration for the nixi extraction pipeline.

NixiConfig reads config at runtime from HERMES_HOME/config.yaml (nixi: section)
or from environment variables in standalone mode. This is separate from
seed_config.py which handles write-time config generation for hermes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


_DEFAULT_BOT_NAMES = ["Fixi", ".OP", "Toothless", "Cerberus"]


@dataclass
class NixiConfig:
    """Configuration for the nixi extraction pipeline.

    Fields:
        log_dir: Path to the slack_logs directory.
        output_dir: Path to the output directory (contains nixi_state.db).
        extraction_batch_size: Number of messages per extraction batch.
        bot_names: List of known bot display names.
        cooccurrence_threshold: Minimum co-occurrence count for relationship mapping.
        memory_limit: Maximum messages to keep in working memory.
        employee_limit: Maximum employee records to process.
        extraction_model: LLM model for extraction tasks.
    """

    log_dir: Path
    output_dir: Path
    extraction_batch_size: int = 50
    bot_names: list[str] = field(default_factory=lambda: list(_DEFAULT_BOT_NAMES))
    cooccurrence_threshold: int = 3
    memory_limit: int = 10_000
    employee_limit: int = 1375
    extraction_model: str = ""

    @property
    def db_path(self) -> Path:
        """Path to nixi_state.db within output_dir."""
        return self.output_dir / "nixi_state.db"

    @classmethod
    def from_config(cls, config_path: Path | None = None) -> NixiConfig:
        """Load NixiConfig from HERMES_HOME/config.yaml nixi: section.

        Args:
            config_path: Explicit path to config.yaml. If None, reads from
                HERMES_HOME/config.yaml.

        Returns:
            NixiConfig populated from config file values with defaults.
        """
        if config_path is None:
            hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
            config_path = hermes_home / "config.yaml"

        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        nixi = config.get("nixi", {})
        hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

        # Resolve log_dir — default to growgami-slack-logs/slack_logs relative to nixi-agent root
        log_dir = Path(nixi.get("log_dir", "")) if nixi.get("log_dir") else (
            Path(__file__).resolve().parent.parent.parent / "growgami-slack-logs" / "slack_logs"
        )

        # Resolve output_dir — from config, or HERMES_HOME/nixi/output, or ~/.nixi/output
        output_dir_str = nixi.get("output_dir", "")
        if output_dir_str:
            output_dir = Path(output_dir_str)
        else:
            output_dir = hermes_home / "nixi" / "output"

        extraction_model = nixi.get("extraction_model", config.get("model", ""))

        return cls(
            log_dir=log_dir,
            output_dir=output_dir,
            extraction_batch_size=nixi.get("extraction_batch_size", 50),
            bot_names=nixi.get("bot_names", list(_DEFAULT_BOT_NAMES)),
            cooccurrence_threshold=nixi.get("cooccurrence_threshold", 3),
            memory_limit=nixi.get("memory_limit", 10_000),
            employee_limit=nixi.get("employee_limit", 1375),
            extraction_model=extraction_model,
        )

    @classmethod
    def from_env(cls) -> NixiConfig:
        """Load NixiConfig from environment variables with sensible defaults.

        Standalone mode — no hermes config.yaml required.

        Env vars:
            NIXI_LOG_DIR: Path to slack_logs directory.
            NIXI_OUTPUT_DIR: Path to output directory.
            NIXI_EXTRACTION_MODEL: LLM model for extraction.
            HERMES_HOME: Base directory (used for default output_dir).
        """
        hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

        log_dir_str = os.environ.get("NIXI_LOG_DIR", "")
        if log_dir_str:
            log_dir = Path(log_dir_str)
        else:
            log_dir = Path(__file__).resolve().parent.parent.parent / "growgami-slack-logs" / "slack_logs"

        output_dir_str = os.environ.get("NIXI_OUTPUT_DIR", "")
        if output_dir_str:
            output_dir = Path(output_dir_str)
        else:
            output_dir = Path.home() / ".nixi" / "output"

        extraction_model = os.environ.get("NIXI_EXTRACTION_MODEL", "")

        return cls(
            log_dir=log_dir,
            output_dir=output_dir,
            extraction_model=extraction_model,
        )