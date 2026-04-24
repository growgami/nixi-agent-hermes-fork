"""Employee overlay loader.

Reads per-employee context from HERMES_HOME/employees/{user_id}/USER.md.
On first interaction, the file won't exist yet — load_overlay() returns
an empty string and logs a debug message. Auto-creation of the overlay
is handled later by the agent, not this module.
"""

import logging
from pathlib import Path

from hermes_constants import get_hermes_home

from nixi.path_validator import safe_path

logger = logging.getLogger(__name__)


def load_overlay(user_id: str) -> str:
    """Load the employee overlay (USER.md) for the given user.

    Args:
        user_id: The employee's unique identifier (from X-Nixi-User-Id header).

    Returns:
        The contents of the USER.md file, or an empty string if the file
        doesn't exist yet (first interaction).
    """
    home = get_hermes_home()
    try:
        overlay_path = safe_path(home, f"employees/{user_id}/USER.md")
    except Exception:
        logger.warning("Invalid user_id path: %s", user_id)
        return ""

    if overlay_path.is_file():
        return overlay_path.read_text(encoding="utf-8")

    logger.debug("No overlay for user %s (first interaction)", user_id)
    return ""


def get_or_create_employee_dir(user_id: str) -> Path:
    """Get or create the employee directory for the given user.

    Creates HERMES_HOME/employees/{user_id}/ if it doesn't exist.

    Args:
        user_id: The employee's unique identifier.

    Returns:
        The path to the employee directory.
    """
    home = get_hermes_home()
    # Validate user_id through path validator
    emp_dir = safe_path(home, f"employees/{user_id}")
    emp_dir.mkdir(parents=True, exist_ok=True)
    return emp_dir