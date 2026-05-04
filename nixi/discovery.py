"""HERMES_HOME auto-discovery via CWD-walk.

Walks upward from start_dir (default: CWD) looking for a valid nixi project
root by searching for data/tenant/*/config.yaml. Returns the directory
containing data/ (the hermes_home), or None if no valid project found.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _is_valid_config(config_path: Path) -> bool:
    """Check if config.yaml has a nixi: section or model: key at top level.

    Returns False if the file is unparseable or lacks both keys.
    """
    try:
        content = yaml.safe_load(config_path.read_text())
    except (yaml.YAMLError, OSError):
        return False

    if not isinstance(content, dict):
        return False

    return "nixi" in content or "model" in content


def discover_hermes_home(
    start_dir: Path | None = None,
    max_depth: int = 20,
) -> Path | None:
    """Walk upward from start_dir to find a valid HERMES_HOME.

    HERMES_HOME is the directory containing data/tenant/*/config.yaml
    where config.yaml has a nixi: section or model: key.

    Args:
        start_dir: Directory to begin search from. Defaults to CWD.
        max_depth: Maximum number of parent directories to traverse.

    Returns:
        Path to hermes_home (parent of data/), or None if not found.

    Edge cases:
        - Multiple valid tenants at same level → None (ambiguous).
        - Invalid config (no nixi or model keys) → skipped with warning.
        - max_depth exceeded → None.
    """
    current = Path(start_dir) if start_dir is not None else Path.cwd()
    depth = 0

    while depth < max_depth:
        tenant_configs = list(current.glob("data/tenant/*/config.yaml"))

        if tenant_configs:
            valid_configs: list[Path] = []
            hermes_home: Path | None = None
            for config_path in tenant_configs:
                if _is_valid_config(config_path):
                    valid_configs.append(config_path)
                    # All valid configs at this level share the same hermes_home
                    if hermes_home is None:
                        # config.yaml → tenant id dir → tenant/ → data/ → hermes_home
                        hermes_home = config_path.parent.parent.parent.parent
                else:
                    logger.warning(
                        "Skipping invalid config at %s — no nixi: or model: key",
                        config_path,
                    )

            if len(valid_configs) > 1:
                logger.warning(
                    "Ambiguous hermes_home: %d valid tenants found at %s",
                    len(valid_configs),
                    current,
                )
                return None

            if len(valid_configs) == 1:
                return hermes_home

            # All configs were invalid — keep walking up

        parent = current.parent
        if parent == current:
            # Reached filesystem root
            return None
        current = parent
        depth += 1

    logger.debug("Stopped search after max_depth=%d levels", max_depth)
    return None