"""HERMES_HOME path jail validator.

Every file operation in the nixi package goes through safe_path(),
which resolves symlinks and checks that the resolved path stays within
the HERMES_HOME directory. This prevents path traversal attacks in the
multi-tenant deployment model where HERMES_HOME is tenant-scoped.
"""

import os
from pathlib import Path


class PathTraversalError(PermissionError):
    """Raised when a path resolves outside the allowed home directory."""


def safe_path(home: Path, requested: str) -> Path:
    """Join *home* with *requested* and verify the result stays within *home*.

    Resolves symlinks and normalizes the path before checking.
    Raises PathTraversalError if the resolved path escapes *home*.

    Args:
        home: The jail root directory (typically HERMES_HOME).
        requested: A relative (or absolute) path to validate.

    Returns:
        The resolved, validated Path.
    """
    home_resolved = home.resolve()
    # Join first, then resolve — this handles both relative and absolute inputs
    candidate = (home / requested).resolve()

    if not str(candidate).startswith(str(home_resolved)):
        raise PathTraversalError(
            f"Path escapes home directory: {requested!r} resolves to {candidate}, "
            f"which is outside {home_resolved}"
        )

    return candidate


def validate_hermes_home() -> Path:
    """Check that HERMES_HOME env var exists and points to a real directory.

    Returns:
        The resolved HERMES_HOME path.

    Raises:
        EnvironmentError: If HERMES_HOME is unset or doesn't exist.
    """
    raw = os.environ.get("HERMES_HOME", "").strip()
    if not raw:
        raise EnvironmentError("HERMES_HOME environment variable is not set")

    home = Path(raw).resolve()
    if not home.is_dir():
        raise EnvironmentError(f"HERMES_HOME does not exist or is not a directory: {home}")

    return home