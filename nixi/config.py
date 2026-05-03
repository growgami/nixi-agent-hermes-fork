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

from hermes_constants import get_hermes_home


_DEFAULT_BOT_NAMES = ["nixi"]


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
        rules_limit: Maximum character size of RULES.md before consolidation.
        extraction_model: LLM model for extraction tasks.
    """

    log_dir: Path
    output_dir: Path
    extraction_batch_size: int = 50
    bot_names: list[str] = field(default_factory=lambda: list(_DEFAULT_BOT_NAMES))
    cooccurrence_threshold: int = 3
    memory_limit: int = 10_000
    employee_limit: int = 1375
    rules_limit: int = 10_000
    extraction_model: str = ""

    @property
    def hermes_home(self) -> Path:
        """Profile-aware HERMES_HOME path, delegated to hermes_constants."""
        return get_hermes_home()

    @property
    def nixi_dir(self) -> Path:
        """Convenience path to hermes_home / nixi for writers and prompt_builder."""
        return self.hermes_home / "nixi"

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
            rules_limit=nixi.get("rules_limit", 10_000),
            extraction_model=extraction_model,
        )

    @classmethod
    def from_env(cls) -> NixiConfig:
        """Load NixiConfig from environment variables with sensible defaults.

        Standalone mode — no hermes config.yaml required.

        Env vars:
            NIXI_LOG_DIR: Path to slack_logs directory.
            NIXI_OUTPUT_DIR: Path to output directory.
            NIXI_EXTRACTION_BATCH_SIZE: Messages per extraction batch (default: 50).
            NIXI_BOT_NAMES: JSON list of bot display names.
            NIXI_COOCCURRENCE_THRESHOLD: Min co-occurrence count (default: 3).
            NIXI_MEMORY_LIMIT: Max messages in working memory (default: 10000).
            NIXI_EMPLOYEE_LIMIT: Max employee records (default: 1375).
            NIXI_RULES_LIMIT: Max character size for RULES.md (default: 10000).
            NIXI_MODEL: LLM model for extraction (fallback: NIXI_EXTRACTION_MODEL).
        """
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

        # Extraction batch size
        extraction_batch_size = int(os.environ.get("NIXI_EXTRACTION_BATCH_SIZE", "50"))

        # Bot names — JSON list or fallback to defaults
        bot_names_env = os.environ.get("NIXI_BOT_NAMES", "")
        if bot_names_env:
            import json
            try:
                bot_names = json.loads(bot_names_env)
            except json.JSONDecodeError:
                bot_names = list(_DEFAULT_BOT_NAMES)
        else:
            bot_names = list(_DEFAULT_BOT_NAMES)

        cooccurrence_threshold = int(os.environ.get("NIXI_COOCCURRENCE_THRESHOLD", "3"))
        memory_limit = int(os.environ.get("NIXI_MEMORY_LIMIT", "10000"))
        employee_limit = int(os.environ.get("NIXI_EMPLOYEE_LIMIT", "1375"))
        rules_limit = int(os.environ.get("NIXI_RULES_LIMIT", "10000"))

        # NIXI_MODEL takes priority; NIXI_EXTRACTION_MODEL as fallback
        extraction_model = os.environ.get("NIXI_MODEL", "") or os.environ.get("NIXI_EXTRACTION_MODEL", "")

        return cls(
            log_dir=log_dir,
            output_dir=output_dir,
            extraction_batch_size=extraction_batch_size,
            bot_names=bot_names,
            cooccurrence_threshold=cooccurrence_threshold,
            memory_limit=memory_limit,
            employee_limit=employee_limit,
            rules_limit=rules_limit,
            extraction_model=extraction_model,
        )