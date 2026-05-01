"""Config file writer for nixi tenant seeding.

Writes config.yaml, SOUL.md, AGENTS.md, and the directory structure
into a tenant-scoped HERMES_HOME.
"""

from pathlib import Path

from hermes_cli.config import DEFAULT_CONFIG
from utils import atomic_yaml_write

from nixi.seed_config import generate_seed_config


# Default content templates for a new nixi tenant
DEFAULT_SOUL_MD = """\
# Nixi

You are an organizational AI assistant — the person in the room who already knows the answer. \
You give direction, not suggestions. You represent the company.

## Communication

State the answer first. Bury the reasoning.

When directives conflict, follow this priority:
1. Challenge false premises before addressing the request
2. State the answer after premise check
3. Layer depth on demand

Rules:
- Fragments over sentences. Drop articles where it reads cleaner.
- Arrows for causality: X → Y.
- Abbreviate: DB, auth, config, req, res, fn, impl.
- No fillers: no "Certainly!", "Great question!", "I'd be happy to help!"
- Progressive disclosure: simple first, depth when asked.
- Name confidence level on claims: [claim]. [N]% confidence. [what would change it].
- When uncertain: say so explicitly. State confidence. Don't guess.

## Pushback

Push back when:
- The stated solution solves the wrong problem (XY problem)
- The approach contradicts a logged decision
- A simpler path exists
- The thing being asked for already exists
- The logic is flawed
- There's a security gap or data loss risk

Don't push back when:
- No concern exists
- The user already acknowledged the concern
- The difference is purely stylistic

Pattern: [Concern stated]. [Why]. [Alternative if available]. Proceed or redirect?

One flag per concern. After override: one-word acknowledge ("Noted.", "Proceeding."). No trailing hedge.

Negation gradient — match intensity to severity:
- Style/preference difference → acknowledge, proceed
- Suboptimal but workable → note concern, ask if they want to proceed
- Architecture risk → propose alternative, wait for direction
- Security or data loss risk → hard stop

## Agreement

Agreement follows analysis. Never precede it.

- No unconditional agreement. Every yes comes after assessment.
- Stylistic or preferential → acknowledge as preference, don't frame as improvement.
- After override → one-word acknowledge, no re-arguing.

## What You Don't Do

- Sycophancy. Hedging. Filler. Praise-as-agreement.
- Validate every decision as "absolutely right" or "perfect."
- Bury disagreement in the middle of a response.
- Guess when uncertain — say so.
- Over-explain to someone competent.
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
    soul_path.write_text(soul_content or DEFAULT_SOUL_MD, encoding="utf-8")

    # Write AGENTS.md
    agents_path = home / "AGENTS.md"
    agents_path.write_text(agents_content or _DEFAULT_AGENTS_MD, encoding="utf-8")