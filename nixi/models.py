"""Data models for the nixi extraction pipeline.

Defines the core data structures used throughout the Slack log extraction
pipeline: ParsedLine, ScrapedMessage, ExtractionBatch, IngestionResult,
UserMap, and Link.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Link:
    """A hyperlink extracted from Slack message text.

    Named replacement for tuple[str, str] — url is the raw URL,
    display is the link text shown to users.
    """

    url: str
    display: str


@dataclass
class ParsedLine:
    """A single parsed line from a Slack log file.

    Attributes:
        slack_ts: Slack timestamp (e.g. "1766766571.412779").
        thread_parent_ts: Parent message timestamp if this is a thread reply,
            None for top-level messages.
        display_name: The @-prefixed name without the @ (e.g. "Kuro").
        raw_text: Full message text, may contain newlines for multi-line messages.
        user_mentions: Extracted Slack user IDs from <@U...> patterns.
        channel_refs: Extracted channel IDs from <#C...|...> patterns (ID only).
        special_mentions: Extracted special mentions (<!here>, <!channel>,
            <!everyone>, <!subteam^...>).
        links: Extracted links from <url|display> patterns, with HTML entities
            decoded (&amp; → &, &lt; → <, &gt; → >).
    """

    slack_ts: str
    thread_parent_ts: str | None
    display_name: str
    raw_text: str
    user_mentions: list[str] = field(default_factory=list)
    channel_refs: list[str] = field(default_factory=list)
    special_mentions: list[str] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)


@dataclass
class ScrapedMessage:
    """A message ready for insertion into scraped_messages table.

    Attributes:
        slack_ts: Slack timestamp.
        channel_id: Channel ID (e.g. "C06M81FSKFF").
        channel_name: Human-readable channel name.
        user_id: Slack user ID (may be None for raw-UID posters).
        user_name: Display name at time of scraping.
        text: Message text.
        thread_ts: Thread timestamp (for thread replies).
        parent_ts: Parent message timestamp (for thread replies).
        is_bot: Whether the poster is a known bot.
        source_file: Source log file path (or channel ID for synthetic sources).
        timestamp: ISO datetime string.
    """

    slack_ts: str
    channel_id: str
    channel_name: str
    user_id: str | None
    user_name: str
    text: str
    thread_ts: str | None
    parent_ts: str | None
    is_bot: bool
    source_file: str
    timestamp: str


@dataclass
class ExtractionBatch:
    """Metadata for a batch of extracted messages.

    Attributes:
        batch_id: Unique identifier for this extraction batch.
        channel_id: Channel this batch was extracted from.
        message_count: Number of messages in the batch.
        extracted_at: ISO datetime string when extraction occurred.
    """

    batch_id: str
    channel_id: str
    message_count: int
    extracted_at: str


@dataclass
class IngestionResult:
    """Result summary from ingesting a channel's log file.

    Attributes:
        total_lines: Total lines processed from the log file.
        parsed: Lines that matched the parser regex.
        bots_tagged: Lines identified as bot messages.
        threads_linked: Lines with thread parent timestamps.
        inserted: New rows inserted into scraped_messages.
        already_existing: Rows that already existed (deduplicated).
        raw_uid_posters: Distinct display names that were raw Slack UIDs.
    """

    total_lines: int
    parsed: int
    bots_tagged: int
    threads_linked: int
    inserted: int
    already_existing: int
    raw_uid_posters: int


@dataclass
class RealtimeMessage:
    """A message from the realtime Socket Mode storage pipeline.

    Stored in the realtime_messages table by the Go ingester. The Python
    extraction pipeline reads from this table (never writes to it).

    Attributes:
        slack_ts: Slack timestamp (e.g. "1766766571.412779").
        channel_id: Channel ID (e.g. "C06M81FSKFF").
        user_id: Slack user ID. Nullable — bot/system messages may lack this.
        text: Message text.
        thread_ts: Thread timestamp (for thread replies).
        parent_ts: Parent message timestamp (for thread replies).
        is_bot: Whether the poster is a bot.
        channel_type: Slack channel type (e.g. "channel", "group", "im", "mpim").
            Nullable — not available in scraped_messages.
        event_id: Socket Mode event ID — unique per delivery.
        client_msg_id: Client-generated message ID. Nullable.
        team_id: Slack workspace/team ID. Nullable.
        timestamp: ISO datetime string.
    """

    slack_ts: str
    channel_id: str
    user_id: str | None
    text: str
    thread_ts: str | None
    parent_ts: str | None
    is_bot: bool
    channel_type: str | None
    event_id: str
    client_msg_id: str | None
    team_id: str | None
    timestamp: str


@dataclass
class UserMap:
    """Bidirectional user name ↔ ID mapping.

    Attributes:
        name_to_id: Maps display names to Slack user IDs.
            Values may be None for unmapped users.
        id_to_name: Maps Slack user IDs to display names.
            Used for resolving user_id from display_name at insertion time.
    """

    name_to_id: dict[str, str | None] = field(default_factory=dict)
    id_to_name: dict[str, str] = field(default_factory=dict)