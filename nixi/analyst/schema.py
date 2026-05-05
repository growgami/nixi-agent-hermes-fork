"""Schema management for analyst_state.db.

Reads and executes schema_analyst.sql against the separate
analyst_state.db, following the same pattern as nixi.db.ensure_schema().
"""

from __future__ import annotations

from pathlib import Path

from nixi.db import get_connection


def ensure_analyst_schema(db_path: Path | None = None) -> None:
    """Execute schema_analyst.sql DDL to create analyst tables and indexes.

    Creates all analyst tables in a SEPARATE analyst_state.db —
    does NOT modify nixi_state.db.

    Args:
        db_path: Path to analyst_state.db.
            Defaults to get_hermes_home() / "analyst_state.db".
    """
    if db_path is None:
        from hermes_constants import get_hermes_home

        db_path = get_hermes_home() / "analyst_state.db"

    db_path.parent.mkdir(parents=True, exist_ok=True)

    schema_path = Path(__file__).parent.parent / "schemas" / "schema_analyst.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")

    conn = get_connection(db_path)
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()