"""LogFileAdapter — walks log directory, parses files, writes to nixi_state.db.

Orchestrates the full ingestion pipeline:
1. Walk slack_logs directory (C* channel dirs)
2. Parse all monthly log files per channel (cross-month)
3. Resolve thread links via per-channel timestamp index
4. Handle orphan thread files (parent deleted from channel logs)
5. Build ScrapedMessage records per channel, insert immediately
6. Two-phase user_id resolution: NULL at insert, UPDATE after user_map build
7. Write user_map.yaml with both name_to_id and id_to_name sections
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from nixi.config import NixiConfig
from nixi.db import (
    build_user_map,
    ensure_schema,
    get_connection,
    insert_messages,
)
from nixi.models import IngestionResult, ScrapedMessage
from nixi.parser import LogParser

logger = logging.getLogger(__name__)


class LogFileAdapter:
    """Walks the log directory, parses Slack log files, and writes to nixi_state.db.

    Constructor accepts a NixiConfig (reads from hermes config or env vars).
    """

    def __init__(self, config: NixiConfig | None = None) -> None:
        if config is None:
            config = NixiConfig.from_config()
        self.config = config
        self.parser = LogParser()

    def ingest(self, force: bool = False) -> IngestionResult:
        """Main entry point: walk log_dir, parse, insert into nixi_state.db.

        Args:
            force: If True, re-parse even if database already has messages.
                Currently unused — ingestion is idempotent via INSERT OR IGNORE.

        Returns:
            IngestionResult with aggregate counts.
        """
        log_dir = self.config.log_dir
        if not log_dir.is_dir():
            raise FileNotFoundError(f"Log directory not found: {log_dir}")

        ensure_schema(self.config.db_path)
        conn = get_connection(self.config.db_path)

        try:
            total_lines = 0
            total_parsed = 0
            total_bots_tagged = 0
            total_threads_linked = 0
            total_inserted = 0
            total_already_existing = 0
            raw_uid_posters: set[str] = set()

            # Walk C* directories (channel IDs)
            channel_dirs = sorted(
                d for d in log_dir.iterdir()
                if d.is_dir() and d.name.startswith("C")
            )

            for channel_dir in channel_dirs:
                channel_id = channel_dir.name
                result = self._ingest_channel(conn, channel_id, channel_dir)
                total_lines += result.total_lines
                total_parsed += result.parsed
                total_bots_tagged += result.bots_tagged
                total_threads_linked += result.threads_linked
                total_inserted += result.inserted
                total_already_existing += result.already_existing
                # Merge raw_uid_posters
                # We need to re-count from the current channel
                raw_uid_posters |= self._count_raw_uid_posters(conn, channel_id)

            # Two-phase user_id resolution
            user_map = build_user_map(conn, self.config.cooccurrence_threshold)

            # UPDATE pass: resolve user_ids where correlation succeeded
            self._resolve_user_ids(conn, user_map)

            # Write user_map.yaml
            self._write_user_map(user_map)

            return IngestionResult(
                total_lines=total_lines,
                parsed=total_parsed,
                bots_tagged=total_bots_tagged,
                threads_linked=total_threads_linked,
                inserted=total_inserted,
                already_existing=total_already_existing,
                raw_uid_posters=len(raw_uid_posters),
            )
        finally:
            conn.close()

    def ingest_channel(self, channel_id: str) -> IngestionResult:
        """Re-ingest a single channel.

        Same cross-month logic, per-channel insert.
        """
        log_dir = self.config.log_dir
        channel_dir = log_dir / channel_id
        if not channel_dir.is_dir():
            raise FileNotFoundError(f"Channel directory not found: {channel_dir}")

        ensure_schema(self.config.db_path)
        conn = get_connection(self.config.db_path)

        try:
            result = self._ingest_channel(conn, channel_id, channel_dir)
            # Rebuild user map and resolve for this channel
            user_map = build_user_map(conn, self.config.cooccurrence_threshold)
            self._resolve_user_ids(conn, user_map)
            self._write_user_map(user_map)

            raw_uid_posters = self._count_raw_uid_posters(conn, channel_id)
            return IngestionResult(
                total_lines=result.total_lines,
                parsed=result.parsed,
                bots_tagged=result.bots_tagged,
                threads_linked=result.threads_linked,
                inserted=result.inserted,
                already_existing=result.already_existing,
                raw_uid_posters=len(raw_uid_posters),
            )
        finally:
            conn.close()

    def _ingest_channel(
        self,
        conn,
        channel_id: str,
        channel_dir: Path,
    ) -> IngestionResult:
        """Parse and insert messages for a single channel directory.

        Cross-month scan: reads ALL YYYY-MM.log files in the channel dir,
        accumulates top-level messages, then resolves thread links.
        Per-channel insert: writes to DB immediately after parsing.
        """
        # Step 1: Parse all monthly logs → top-level messages only
        monthly_logs = sorted(channel_dir.glob("*.log"))
        top_level_messages: list[ScrapedMessage] = []
        total_lines = 0
        parsed_count = 0
        bots_tagged = 0

        # Build timestamp index: slack_ts → ParsedLine for ALL top-level messages
        ts_index: dict[str, "ParsedLine"] = {}

        for log_file in monthly_logs:
            lines = self.parser.parse_channel_file(log_file)
            total_lines += self._count_lines(log_file)
            parsed_count += len(lines)

            for parsed in lines:
                ts_index[parsed.slack_ts] = parsed

        # Step 2: Resolve thread links
        threads_dir = channel_dir / "threads"
        thread_messages: list[ScrapedMessage] = []
        threads_linked = 0
        orphan_thread_files: list[str] = []

        if threads_dir.is_dir():
            for thread_file in sorted(threads_dir.glob("*.log")):
                # Derive parent_ts from thread filename
                # Thread filename format: {slack_ts}.log
                parent_ts = thread_file.stem

                # Check if parent_ts appears in the timestamp index
                if parent_ts in ts_index:
                    parent_parsed = ts_index[parent_ts]
                    # Override parent's thread_ts if it doesn't already have one
                    if parent_parsed.thread_parent_ts is None:
                        # Parent is a top-level message, set its thread_ts
                        pass  # Thread file exists means parent starts a thread
                else:
                    # Orphan thread: parent message deleted from channel logs
                    orphan_thread_files.append(thread_file.name)
                    logger.info(
                        "Orphan thread file: %s (parent_ts=%s not in channel logs)",
                        thread_file.name,
                        parent_ts,
                    )

                # Parse thread file regardless
                thread_lines = self.parser.parse_thread_file(thread_file)
                for tl in thread_lines:
                    # Set parent_ts on all thread replies
                    # The first line in a thread file IS the parent message
                    # All lines (including parent) get parent_ts from filename
                    threads_linked += 1

                    # Build ScrapedMessage for this thread line
                    display_name = tl.display_name
                    is_raw_uid = self.parser.is_raw_uid(display_name)
                    is_bot = self.parser.is_bot_message(
                        display_name, self.config.bot_names
                    )
                    bots_tagged += 1 if is_bot else 0

                    # For thread replies: parent_ts comes from filename
                    # thread_ts = parent_ts for all lines in the thread
                    msg = self._build_scraped_message(
                        parsed=tl,
                        channel_id=channel_id,
                        parent_ts=parent_ts,
                        is_bot=is_bot,
                        is_raw_uid=is_raw_uid,
                    )
                    thread_messages.append(msg)

        # Step 3: Build ScrapedMessage for top-level messages
        # Skip top-level messages that already have thread files —
        # they'll be picked up from the thread file parse above
        thread_parent_ts_set = set()
        if threads_dir.is_dir():
            for tf in threads_dir.glob("*.log"):
                thread_parent_ts_set.add(tf.stem)

        for parsed in ts_index.values():
            is_bot = self.parser.is_bot_message(
                parsed.display_name, self.config.bot_names
            )
            bots_tagged += 1 if is_bot else 0
            is_raw_uid = self.parser.is_raw_uid(parsed.display_name)

            # If this top-level message has a thread file, skip adding it
            # from the channel log — it's already in thread_messages
            # from the thread file parse (first line = parent message)
            if parsed.slack_ts in thread_parent_ts_set:
                # The parent message is already included from the thread file
                # Don't add a duplicate from the channel log
                continue

            msg = self._build_scraped_message(
                parsed=parsed,
                channel_id=channel_id,
                parent_ts=None,
                is_bot=is_bot,
                is_raw_uid=is_raw_uid,
            )
            top_level_messages.append(msg)

        # Step 4: Per-channel insert — write all messages for this channel
        all_messages = top_level_messages + thread_messages
        inserted, already_existing = self._insert_channel_messages(
            conn, all_messages
        )

        return IngestionResult(
            total_lines=total_lines,
            parsed=parsed_count + threads_linked,
            bots_tagged=bots_tagged,
            threads_linked=threads_linked,
            inserted=inserted,
            already_existing=already_existing,
            raw_uid_posters=0,  # Computed at aggregate level
        )

    def _build_scraped_message(
        self,
        parsed,
        channel_id: str,
        parent_ts: str | None,
        is_bot: bool,
        is_raw_uid: bool,
    ) -> ScrapedMessage:
        """Build a ScrapedMessage from a ParsedLine.

        Raw UID posters: user_id AND user_name both set to the UID value.
        Normal posters: user_id=None (resolved later via user_map).
        """
        display_name = parsed.display_name
        user_id: str | None
        user_name: str

        if is_raw_uid:
            user_id = display_name  # e.g. "U09NDP0R44Q"
            user_name = display_name
        else:
            user_id = None  # Resolved in two-phase pass
            user_name = display_name

        # Timestamp conversion: slack_ts → ISO datetime
        timestamp = datetime.fromtimestamp(
            float(parsed.slack_ts), tz=timezone.utc
        ).isoformat()

        # Determine thread_ts
        thread_ts: str | None = parent_ts

        # Source file tracking — use channel_id as source identifier
        source_file = channel_id

        return ScrapedMessage(
            slack_ts=parsed.slack_ts,
            channel_id=channel_id,
            channel_name=channel_id,  # No name resolution at bootstrap
            user_id=user_id,
            user_name=user_name,
            text=parsed.raw_text,
            thread_ts=thread_ts,
            parent_ts=parent_ts,
            is_bot=is_bot,
            source_file=source_file,
            timestamp=timestamp,
        )

    def _insert_channel_messages(
        self,
        conn,
        messages: list[ScrapedMessage],
    ) -> tuple[int, int]:
        """Insert messages for a single channel. Returns (inserted, already_existing)."""
        total = len(messages)
        inserted = insert_messages(conn, messages)
        already_existing = total - inserted
        return inserted, already_existing

    def _resolve_user_ids(self, conn, user_map) -> None:
        """UPDATE scraped_messages where user_id IS NULL and display_name has a mapping."""
        for display_name, uid in user_map.name_to_id.items():
            if uid is not None:
                conn.execute(
                    """UPDATE scraped_messages
                       SET user_id = ?
                       WHERE user_name = ? AND user_id IS NULL""",
                    (uid, display_name),
                )
        conn.commit()

    def _count_raw_uid_posters(self, conn, channel_id: str) -> set[str]:
        """Get distinct raw UID poster names for a channel."""
        cursor = conn.execute(
            "SELECT DISTINCT user_name FROM scraped_messages WHERE channel_id = ? AND user_id = user_name",
            (channel_id,),
        )
        # A raw UID poster has user_id == user_name (both set to the UID)
        raw_uid_pattern = re.compile(r"^U[A-Z0-9]{8,}$")
        result: set[str] = set()
        for row in cursor.fetchall():
            name = row["user_name"]
            if raw_uid_pattern.match(name):
                result.add(name)
        return result

    def _write_user_map(self, user_map) -> None:
        """Write user_map.yaml with both name_to_id and id_to_name sections.

        Writes to the output_dir/schemas/ directory so each tenant gets
        its own user map alongside nixi_state.db.
        """
        schemas_dir = self.config.output_dir / "schemas"
        schemas_dir.mkdir(parents=True, exist_ok=True)
        user_map_path = schemas_dir / "user_map.yaml"

        data = {
            "name_to_id": {
                k: v for k, v in user_map.name_to_id.items()
            },
            "id_to_name": dict(user_map.id_to_name),
        }
        user_map_path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _count_lines(filepath: Path) -> int:
        """Count total lines in a file."""
        return len(filepath.read_text(encoding="utf-8").splitlines())