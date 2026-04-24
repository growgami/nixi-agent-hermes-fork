"""Nixi deployment entry point — seeds config and starts gateway.

Validates required environment variables, seeds HERMES_HOME if needed,
sets NIXI_MODE=1 for send-only Slack mode, and starts the messaging gateway.

Environment variables:
  NIXI_INTERNAL_SECRET — shared secret between Sludge and nixi-agent
  NIXI_TEAM_ID         — Slack team ID this tenant serves (e.g. T01XYZ567AB)
  NIXI_PORT            — HTTP listen port (default 8080)
  SLACK_BOT_TOKEN      — workspace-specific xoxb- token
  NIXI_ALLOWED_USERS   — comma-separated Slack user IDs (empty = all allowed)
  NIXI_HOME_CHANNEL    — Slack channel ID for home channel (silences startup warning)
  HERMES_HOME          — /data/tenants/{company_id} (path-jail)
  NIXI_MODE            — set to '1' automatically by start_nixi()
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── Load .env before any env var reads ─────────────────────────────────────
_env_paths = [
    Path(__file__).resolve().parents[1] / ".env",  # repo root .env
]
for _p in _env_paths:
    if _p.exists():
        load_dotenv(_p, override=False)

# ── Required env vars for nixi deployment ─────────────────────────────────
_REQUIRED_VARS = [
    "NIXI_INTERNAL_SECRET",
    "NIXI_TEAM_ID",
    "SLACK_BOT_TOKEN",
]

# SLACK_APP_TOKEN is NOT required — NIXI_MODE disables Socket Mode


def validate_env() -> Path:
    """Validate required environment variables and return HERMES_HOME.

    Checks that all required env vars are set and that HERMES_HOME
    points to an existing directory. Does NOT check SLACK_APP_TOKEN
    because NIXI_MODE disables Socket Mode.

    Returns:
        Resolved Path to HERMES_HOME.

    Raises:
        EnvironmentError: If any required var is unset or HERMES_HOME
            doesn't exist.
    """
    missing = [v for v in _REQUIRED_VARS if not os.environ.get(v, "").strip()]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )

    raw_home = os.environ.get("HERMES_HOME", "").strip()
    if not raw_home:
        raise EnvironmentError("HERMES_HOME environment variable is not set")

    home = Path(raw_home).resolve()
    if not home.is_dir():
        home.mkdir(parents=True, exist_ok=True)
        logger.info("[nixi] Created HERMES_HOME directory: %s", home)

    return home


def seed_if_needed(home: Path) -> None:
    """Seed HERMES_HOME with config if config.yaml doesn't already exist.

    Reads model provider/model from environment or falls back to defaults.
    Uses the live DEFAULT_CONFIG version (not hardcoded) for _config_version.

    Args:
        home: Path to the tenant's HERMES_HOME directory.
    """
    config_path = home / "config.yaml"
    if config_path.exists():
        logger.info("[nixi] config.yaml already exists — skipping seed")
        return

    from nixi.config_seeder import seed_hermes_home

    model_provider = os.environ.get("HERMES_MODEL_PROVIDER", "openai")
    model = os.environ.get("HERMES_MODEL", "gpt-4o")
    slack_workspace_id = os.environ.get("NIXI_TEAM_ID", "")
    home_channel = os.environ.get("NIXI_HOME_CHANNEL", "")

    seed_hermes_home(
        home=home,
        company_name=os.environ.get("NIXI_COMPANY_NAME", "Tenant"),
        slack_workspace_id=slack_workspace_id,
        model_provider=model_provider,
        model=model,
        home_channel=home_channel,
    )

    logger.info("[nixi] Seeded config at %s", config_path)


def _get_port() -> int:
    """Read NIXI_PORT from env, defaulting to 8080."""
    try:
        return int(os.environ.get("NIXI_PORT", "8080"))
    except ValueError:
        return 8080


def _import_gateway():
    """Lazy import of gateway.run to avoid side effects at module level.

    NIXI_MODE must be set before this import because gateway modules
    check it at connect() time.
    """
    from gateway.run import start_gateway

    return start_gateway


def start_nixi() -> None:
    """Validate env, seed config, set NIXI_MODE, and start the gateway.

    This is the main entry point for the nixi-agent container.
    Called by ``python -m nixi``.

    Steps:
      1. Validate required env vars (NIXI_INTERNAL_SECRET, NIXI_TEAM_ID,
         SLACK_BOT_TOKEN, HERMES_HOME).
      2. Seed HERMES_HOME/config.yaml if it doesn't exist.
      3. Set NIXI_MODE=1 (disables Socket Mode, enables send-only Slack).
      4. Import and call start_gateway().
      5. Log startup message with team_id and port.

    Raises:
        EnvironmentError: If required env vars are missing or HERMES_HOME
            doesn't exist.
        SystemExit: With code 1 if the gateway fails to start.
    """
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║          nixi-agent v1.0.0            ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    # Step 1: Validate environment
    home = validate_env()
    team_id = os.environ["NIXI_TEAM_ID"]
    port = _get_port()
    model_provider = os.environ.get("HERMES_MODEL_PROVIDER", "openai")
    model = os.environ.get("HERMES_MODEL", "gpt-4o")
    company = os.environ.get("NIXI_COMPANY_NAME", "Tenant")

    print(f"  Team:       {team_id}")
    print(f"  Company:    {company}")
    print(f"  Home:       {home}")
    print(f"  Port:       {port}")
    print(f"  Model:      {model_provider}/{model}")
    home_channel = os.environ.get("NIXI_HOME_CHANNEL", "")
    print(f"  Home Channel: {home_channel if home_channel else '(not set)'}")
    print(f"  Mode:       nixi (Slack send-only)")
    print()

    # Step 2: Seed config if needed
    config_path = home / "config.yaml"
    seed_if_needed(home)

    if config_path.exists():
        print(f"  Config:     {config_path} (existing)")
    else:
        print(f"  Config:     {config_path} (seeded)")

    print()

    # Step 3: Set NIXI_MODE BEFORE importing gateway
    # This must happen before gateway modules read the env var
    os.environ["NIXI_MODE"] = "1"

    logger.info(
        "[nixi] Starting gateway — team_id=%s, port=%d, home=%s",
        team_id,
        port,
        home,
    )

    # Step 4: Lazy import and start gateway
    print("  Starting gateway...")
    print()
    start_gateway = _import_gateway()

    success = asyncio.run(start_gateway())
    if not success:
        logger.error("[nixi] Gateway exited with failure")
        print()
        print("  [FAIL] Gateway exited with failure")
        sys.exit(1)

    logger.info("[nixi] Gateway shut down cleanly")
    print()
    print("  Gateway shut down cleanly")