"""Config file writer for nixi tenant seeding.

Writes config.yaml, SOUL.md, AGENTS.md, and the directory structure
into a tenant-scoped HERMES_HOME.
"""

from pathlib import Path

from hermes_cli.config import DEFAULT_CONFIG
from utils import atomic_yaml_write

from nixi.seed_config import generate_seed_config


# Default content templates for a new nixi tenant
_DEFAULT_SOUL_MD = """\
# Company AI Assistant

You are an organizational AI assistant. You help employees with their daily work,
answer questions, and carry out tasks using your available tools. You represent
the company — be professional, helpful, and consistent across all interactions.

## Guidelines

- Be direct and efficient in your responses
- Prioritize accuracy and completeness
- When uncertain, say so clearly rather than guessing
- Respect context from employee overlays — personalize where appropriate
"""

_DEFAULT_AGENTS_MD = """\
# Agents Configuration

## Overview

This file configures the organizational AI assistant's behavior and capabilities.

## Channels

- **Slack**: Primary communication channel for employee interactions
- **Nixi**: Internal gateway for multi-tenant message routing

## Memory

- **Scope**: Organization-level. Memories are shared across all employees.
- **Employee Context**: Per-employee overlays injected via channel_prompt (ephemeral, not persisted)
"""

# Directories to create under HERMES_HOME
_DIRS = [
    "employees",
    "skills",
    "skills/seeded",
    "skills/channel",
    "skills/event",
    "skills/learned",
    "skills/archive",
]


def seed_hermes_home(
    *,
    home: Path,
    company_name: str,
    slack_workspace_id: str,
    model_provider: str,
    model: str,
    home_channel: str = "",
    soul_content: str | None = None,
    agents_content: str | None = None,
) -> None:
    """Seed a tenant-scoped HERMES_HOME with config and directory structure.

    Creates config.yaml (via atomic_yaml_write), SOUL.md, AGENTS.md,
    and the required subdirectory structure.

    Args:
        home: Path to the tenant's HERMES_HOME directory.
        company_name: Organization display name.
        slack_workspace_id: Slack workspace/team ID.
        model_provider: LLM provider name.
        model: LLM model slug.
        home_channel: Slack channel ID for the nixi home channel. When non-empty,
            included as gateway.nixi.home_channel in config.yaml. When empty/omitted,
            the key is left out entirely.
        soul_content: Custom SOUL.md content (falls back to default template).
        agents_content: Custom AGENTS.md content (falls back to default template).
    """
    config = generate_seed_config(
        company_name=company_name,
        slack_workspace_id=slack_workspace_id,
        model_provider=model_provider,
        model=model,
        home_channel=home_channel,
    )

    # Create directory structure
    home.mkdir(parents=True, exist_ok=True)
    for subdir in _DIRS:
        (home / subdir).mkdir(parents=True, exist_ok=True)

    # Write config.yaml atomically
    atomic_yaml_write(home / "config.yaml", config)

    # Write SOUL.md
    soul_path = home / "SOUL.md"
    soul_path.write_text(soul_content or _DEFAULT_SOUL_MD, encoding="utf-8")

    # Write AGENTS.md
    agents_path = home / "AGENTS.md"
    agents_path.write_text(agents_content or _DEFAULT_AGENTS_MD, encoding="utf-8")