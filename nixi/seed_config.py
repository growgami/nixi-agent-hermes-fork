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
    home_channel: str = "",
) -> dict:
    """Build the config.yaml dict for a nixi tenant.

    Args:
        company_name: Organization display name.
        slack_workspace_id: Slack workspace/team ID (e.g. T01XYZ567AB).
        model_provider: LLM provider name (e.g. "openai").
        model: LLM model slug (e.g. "gpt-4o").
        home_channel: Slack channel ID for the nixi home channel (e.g. C0AE0QVNT1P).
            When non-empty, included as gateway.nixi.home_channel. When empty/omitted,
            the key is left out entirely (backward compatible).

    Returns:
        A dict ready to be written as config.yaml via atomic_yaml_write.
    """
    # Dynamic _config_version — must match what check_config_version() compares
    # against, otherwise the migration wizard fires on every startup.
    config_version = DEFAULT_CONFIG.get("_config_version", 1)

    nixi_section: dict = {"enabled": True}
    if home_channel:
        nixi_section["home_channel"] = home_channel

    # Extraction pipeline defaults under nixi: section
    nixi_extraction: dict = {
        "log_dir": "",  # resolved at runtime from HERMES_HOME or ~/.nixi
        "output_dir": "",  # resolved at runtime from HERMES_HOME or ~/.nixi/output
        "extraction_batch_size": 50,
        "bot_names": ["Fixi", "nixi"],
        "cooccurrence_threshold": 3,
        "memory_limit": 10000,
        "employee_limit": 1375,
    }

    return {
        "_config_version": config_version,
        "model": model,
        "gateway": {
            "slack": {
                "enabled": True,
                "workspace_id": slack_workspace_id,
            },
            "nixi": nixi_section,
        },
        "memory": {
            "scope": "organization",
        },
        "terminal": {
            "backend": "local",
            "timeout": 180,
        },
        "nixi": nixi_extraction,
    }