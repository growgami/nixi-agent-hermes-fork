"""Database helpers for nixi_state.db.

Manages the SQLite database schema, connections, and query helpers
for the Slack log extraction pipeline.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from nixi.models import ScrapedMessage, UserMap

_RAW_UID_RE = re.compile(r"^U[A-Z0-9]{8,}$")
_MENTION_RE = re.compile(r"<@(U[A-Z0-9]+)>")


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode enabled.

    Args:
        db_path: Path to the nixi_state.db file.

    Returns:
        sqlite3.Connection with WAL mode and foreign keys enabled.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(db_path: Path | None = None) -> None:
    """Execute schema.sql DDL to create tables and indexes if they don't exist.

    Args:
        db_path: Path to nixi_state.db. Defaults to NixiConfig.db_path.
    """
    if db_path is None:
        from nixi.config import NixiConfig

        db_path = NixiConfig.from_config().db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)

    schema_path = Path(__file__).parent / "schemas" / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")

    conn = get_connection(db_path)
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


def insert_messages(
    conn: sqlite3.Connection,
    messages: list[ScrapedMessage],
) -> int:
    """Batch INSERT OR IGNORE messages into scraped_messages.

    Args:
        conn: Active SQLite connection.
        messages: List of ScrapedMessage records to insert.

    Returns:
        Number of new rows actually inserted (excludes duplicates).
    """
    if not messages:
        return 0

    rows = [
        (
            m.slack_ts,
            m.channel_id,
            m.channel_name,
            m.user_id,
            m.user_name,
            m.text,
            m.thread_ts,
            m.parent_ts,
            int(m.is_bot),
            m.source_file,
            m.timestamp,
        )
        for m in messages
    ]

    cursor = conn.executemany(
        """INSERT OR IGNORE INTO scraped_messages
           (slack_ts, channel_id, channel_name, user_id, user_name, text,
            thread_ts, parent_ts, is_bot, source_file, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return cursor.rowcount


def get_unprocessed(
    conn: sqlite3.Connection,
    channel_id: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Select messages not yet extracted for a given channel.

    Returns messages where channel_id matches AND NOT EXISTS in
    nixi_extraction_log, ordered by timestamp ASC.
    """
    cursor = conn.execute(
        """SELECT * FROM scraped_messages
           WHERE channel_id = ?
             AND NOT EXISTS (
               SELECT 1 FROM nixi_extraction_log
               WHERE nixi_extraction_log.channel_id = scraped_messages.channel_id
                 AND nixi_extraction_log.slack_ts = scraped_messages.slack_ts
             )
           ORDER BY timestamp ASC
           LIMIT ?""",
        (channel_id, limit),
    )
    return [dict(row) for row in cursor.fetchall()]


def get_unprocessed_channels(conn: sqlite3.Connection) -> list[str]:
    """SELECT DISTINCT channel_ids that have no extraction log entries."""
    cursor = conn.execute(
        """SELECT DISTINCT channel_id FROM scraped_messages
           WHERE NOT EXISTS (
               SELECT 1 FROM nixi_extraction_log
               WHERE nixi_extraction_log.channel_id = scraped_messages.channel_id
                 AND nixi_extraction_log.slack_ts = scraped_messages.slack_ts
           )"""
    )
    return [row["channel_id"] for row in cursor.fetchall()]


def mark_extracted(
    conn: sqlite3.Connection,
    channel_id: str,
    slack_ts_list: list[str],
    batch_id: str,
) -> None:
    """INSERT into nixi_extraction_log for processed messages."""
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (channel_id, slack_ts, batch_id, now)
        for slack_ts in slack_ts_list
    ]
    conn.executemany(
        """INSERT INTO nixi_extraction_log
           (channel_id, slack_ts, extraction_batch, extracted_at)
           VALUES (?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def count_by_channel(conn: sqlite3.Connection) -> dict[str, int]:
    """COUNT messages grouped by channel_id."""
    cursor = conn.execute(
        "SELECT channel_id, COUNT(*) as cnt FROM scraped_messages GROUP BY channel_id"
    )
    return {row["channel_id"]: row["cnt"] for row in cursor.fetchall()}


def build_user_map(conn: sqlite3.Connection, cooccurrence_threshold: int = 3) -> UserMap:
    """Build a UserMap correlating display names with Slack user IDs.

    Uses a co-occurrence heuristic: if a display_name mentions a user_id in
    their messages (via <@U...> patterns), and the pair co-occurs at least
    `cooccurrence_threshold` times across the dataset, the correlation is
    accepted.

    Args:
        conn: Active SQLite connection.
        cooccurrence_threshold: Minimum co-occurrence count to accept a
            display_name → user_id mapping.

    Returns:
        UserMap with name_to_id and id_to_name populated.
    """
    # Collect (display_name, user_mentions_json) for all messages.
    # user_mentions aren't stored as a separate column — we re-extract from text.
    # But we DO have user_id (may be NULL) and user_name for every row.
    name_to_id: dict[str, str | None] = {}
    id_to_name: dict[str, str] = {}

    # Phase 1: Direct user_id correlations from rows where user_id IS NOT NULL
    cursor = conn.execute(
        "SELECT DISTINCT user_name, user_id FROM scraped_messages WHERE user_id IS NOT NULL"
    )
    for row in cursor.fetchall():
        name = row["user_name"]
        uid = row["user_id"]
        if uid:
            name_to_id[name] = uid
            id_to_name[uid] = name

    # Phase 2: Self-mention heuristic — parse <@U...> from text
    # Track co-occurrence counts: (display_name, mentioned_uid) → count
    cooccurrence: dict[tuple[str, str], int] = {}

    cursor = conn.execute(
        "SELECT user_name, text FROM scraped_messages"
    )
    for row in cursor.fetchall():
        display_name = row["user_name"]
        text = row["text"] or ""
        for m in _MENTION_RE.finditer(text):
            mentioned_uid = m.group(1)
            key = (display_name, mentioned_uid)
            cooccurrence[key] = cooccurrence.get(key, 0) + 1

    # Accept correlations that meet threshold
    for (display_name, uid), count in cooccurrence.items():
        if count >= cooccurrence_threshold and display_name not in name_to_id:
            name_to_id[display_name] = uid
            id_to_name.setdefault(uid, display_name)

    return UserMap(name_to_id=name_to_id, id_to_name=id_to_name)