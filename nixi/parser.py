"""Log file parser for the nixi extraction pipeline.

Parses Slack log files with the format:
    [slack_ts] (thread:parent_ts) @display_name: message text

Supports:
- Single-line and multi-line message accumulation
- Thread lines (optional skip mode for channel files)
- Slack mention extraction: <@U...>, <#C...|...>, <!here/channel/everyone/subteam>
- Link extraction: <url|display> with HTML entity decoding
- Raw UID detection: display names matching U[A-Z0-9]{8,}
- Bot name detection: configurable bot name list
"""

from __future__ import annotations

import re
from pathlib import Path

from nixi.models import Link, ParsedLine


# ── Regex patterns ────────────────────────────────────────────────────────────

# Main line pattern: [slack_ts] (optional thread marker) @name: text
LINE_RE = re.compile(
    r"^\[(\d+\.\d+)\]\s*(?:\(thread:([\d.]+)\)\s*)?@([^:]+):\s*(.*)$"
)

# Raw UID pattern: U followed by 8+ uppercase alphanumeric chars
RAW_UID_RE = re.compile(r"^U[A-Z0-9]{8,}$")

# Slack mention patterns
USER_MENTION_RE = re.compile(r"<@U[A-Z0-9]+>")
CHANNEL_REF_RE = re.compile(r"<#(C[A-Z0-9]+)\|[^>]*>")
SPECIAL_MENTION_RE = re.compile(r"<!(?:here|channel|everyone|subteam\^[A-Z0-9]+)>")
LINK_RE = re.compile(r"<([^|>]+)\|([^>]+)>")

# HTML entity decoding
_HTML_ENTITY_MAP = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
}


def _decode_html_entities(text: str) -> str:
    """Decode HTML entities in a string (&amp; → &, &lt; → <, &gt; → >)."""
    for entity, char in _HTML_ENTITY_MAP.items():
        text = text.replace(entity, char)
    return text


class LogParser:
    """Stateful parser for Slack log files.

    Supports two modes:
    - Channel file mode (skip_thread_lines=True): Lines with (thread:...)
      prefix are skipped entirely.
    - Thread file mode (skip_thread_lines=False): All lines parsed, including
      thread markers.
    """

    # Default bot names — configurable via is_bot_message()
    DEFAULT_BOT_NAMES: list[str] = ["Fixi", "nixi"]

    def parse_line(self, line: str) -> ParsedLine | None:
        """Parse a single line into a ParsedLine.

        Returns None for blank/whitespace-only lines or lines that don't
        match the expected format. Continuation lines (no timestamp) return
        None — they're handled by parse_file().
        """
        stripped = line.strip()
        if not stripped:
            return None

        match = LINE_RE.match(stripped)
        if not match:
            return None

        slack_ts = match.group(1)
        thread_parent_ts = match.group(2)  # None if no (thread:...) prefix
        display_name = match.group(3)
        raw_text = match.group(4)

        # Extract mentions from raw text
        # <@U04K8NLDCG0> → extract U04K8NLDCG0 (strip <@ and >)
        user_mentions = []
        for m in USER_MENTION_RE.finditer(raw_text):
            full_match = m.group(0)
            # Extract just the UID portion: <@U04K8NLDCG0> → U04K8NLDCG0
            user_mentions.append(full_match[2:-1])

        channel_refs = [m.group(1) for m in CHANNEL_REF_RE.finditer(raw_text)]

        special_mentions = [m.group(0) for m in SPECIAL_MENTION_RE.finditer(raw_text)]

        links = []
        for m in LINK_RE.finditer(raw_text):
            url = _decode_html_entities(m.group(1))
            display = _decode_html_entities(m.group(2))
            links.append(Link(url=url, display=display))

        return ParsedLine(
            slack_ts=slack_ts,
            thread_parent_ts=thread_parent_ts,
            display_name=display_name,
            raw_text=raw_text,
            user_mentions=user_mentions,
            channel_refs=channel_refs,
            special_mentions=special_mentions,
            links=links,
        )

    @staticmethod
    def is_raw_uid(display_name: str) -> bool:
        """Check if a display name is a raw Slack user ID format.

        Returns True for names like 'U09NDP0R44Q', False for normal names.
        """
        return bool(RAW_UID_RE.match(display_name))

    @staticmethod
    def is_bot_message(
        display_name: str,
        bot_names: list[str] | None = None,
    ) -> bool:
        """Check if a display name belongs to a known bot.

        Args:
            display_name: The poster's display name.
            bot_names: List of bot names. Defaults to DEFAULT_BOT_NAMES.

        Returns:
            True if display_name matches a known bot name.
            Raw UIDs are NOT bots unless they happen to match a bot name.
        """
        names = bot_names or LogParser.DEFAULT_BOT_NAMES
        return display_name in names

    def parse_file(
        self,
        filepath: Path,
        *,
        skip_thread_lines: bool = False,
    ) -> list[ParsedLine]:
        """Parse a log file into a list of ParsedLines.

        Handles multi-line messages: lines that don't match the regex are
        appended to the previous message's raw_text with newline separator.

        Args:
            filepath: Path to the log file.
            skip_thread_lines: If True, skip lines with (thread:...) prefix
                entirely. Used for channel file parsing.

        Returns:
            List of ParsedLine objects.
        """
        results: list[ParsedLine] = []
        current: ParsedLine | None = None

        for line in filepath.read_text(encoding="utf-8").splitlines():
            parsed = self.parse_line(line)

            if parsed is not None:
                # Skip thread lines in channel mode
                if skip_thread_lines and parsed.thread_parent_ts is not None:
                    continue

                # Emit previous accumulated message
                if current is not None:
                    results.append(current)
                current = parsed
            elif current is not None:
                # Continuation line: append to current message
                stripped = line.strip()
                if stripped:  # skip empty continuation lines
                    current.raw_text += "\n" + stripped
            # else: blank line before any message — skip

        # Emit final accumulated ParsedLine
        if current is not None:
            results.append(current)

        return results

    def parse_channel_file(self, filepath: Path) -> list[ParsedLine]:
        """Parse a channel log file, skipping thread reply lines.

        Delegates to parse_file with skip_thread_lines=True.
        """
        return self.parse_file(filepath, skip_thread_lines=True)

    def parse_thread_file(self, filepath: Path) -> list[ParsedLine]:
        """Parse a thread log file, keeping all lines including thread markers.

        Delegates to parse_file with skip_thread_lines=False.
        """
        return self.parse_file(filepath, skip_thread_lines=False)