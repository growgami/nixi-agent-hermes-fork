"""Seed config YAML generator.

Produces the config.yaml dict for a new nixi tenant. Reads _config_version
dynamically from hermes_cli.config.DEFAULT_CONFIG instead of hardcoding it —
this avoids triggering the setup wizard / migration flow on every update.
"""

from hermes_cli.config import DEFAULT_CONFIG


def generate_seed_config(
    *,
    company_name: str,
    slack_workspace_id: str,
    model_provider: str,
    model: str,
) -> dict:
    """Build the config.yaml dict for a nixi tenant.

    Args:
        company_name: Organization display name.
        slack_workspace_id: Slack workspace/team ID (e.g. T01XYZ567AB).
        model_provider: LLM provider name (e.g. "openai").
        model: LLM model slug (e.g. "gpt-4o").

    Returns:
        A dict ready to be written as config.yaml via atomic_yaml_write.
    """
    # Dynamic _config_version — must match what check_config_version() compares
    # against, otherwise the migration wizard fires on every startup.
    config_version = DEFAULT_CONFIG.get("_config_version", 1)

    return {
        "_config_version": config_version,
        "model": model,
        "gateway": {
            "slack": {
                "enabled": True,
                "workspace_id": slack_workspace_id,
            },
            "nixi": {
                "enabled": True,
            },
        },
        "memory": {
            "scope": "organization",
        },
        "terminal": {
            "backend": "local",
            "timeout": 180,
        },
    }