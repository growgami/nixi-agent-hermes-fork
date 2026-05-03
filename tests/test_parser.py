"""Tests for nixi.parser — log file parser for Slack extraction pipeline.

Covers: ParsedLine extraction, multi-line accumulation, thread/channel modes,
mention/link/special-mention extraction, bot detection, raw UID detection,
and real log file parsing.
"""

import textwrap
from pathlib import Path

import pytest

from nixi.parser import LogParser
from nixi.models import Link, ParsedLine


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def parser() -> LogParser:
    """Default LogParser instance."""
    return LogParser()


@pytest.fixture
def sample_channel_log(tmp_path: Path) -> Path:
    """A channel-style log file with thread markers (skipped in channel mode)."""
    content = textwrap.dedent("""\
        [1766766571.412779] @Kuro: hello world
        [1766775007.615089] @Jin: check this <@U04K8NLDCG0>
        [1766775007.999999] (thread:1766766571.412779) @OG: thread reply here
        [1766780001.123456] @Riya: multi-line
        message continues here
        and here too
        [1766789999.000001] @nixi: I'm a bot
    """)
    path = tmp_path / "2025-12.log"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def sample_thread_log(tmp_path: Path) -> Path:
    """A thread-style log file (all lines kept, including thread markers)."""
    content = textwrap.dedent("""\
        [1766766571.412779] @Kuro: original message
        [1766775007.615089] (thread:1766766571.412779) @Jin: first reply
        [1766775008.999999] (thread:1766766571.412779) @OG: second reply
    """)
    path = tmp_path / "thread.log"
    path.write_text(content, encoding="utf-8")
    return path


# ── parse_line: Standard single-line message ──────────────────────────────────

class TestParseLine:
    def test_standard_message(self, parser: LogParser):
        line = "[1766766571.412779] @Kuro: hello world"
        result = parser.parse_line(line)
        assert result is not None
        assert result.slack_ts == "1766766571.412779"
        assert result.thread_parent_ts is None
        assert result.display_name == "Kuro"
        assert result.raw_text == "hello world"

    def test_message_with_thread_marker(self, parser: LogParser):
        line = "[1766775007.615089] (thread:1766766571.412779) @Jin: reply"
        result = parser.parse_line(line)
        assert result is not None
        assert result.slack_ts == "1766775007.615089"
        assert result.thread_parent_ts == "1766766571.412779"
        assert result.display_name == "Jin"
        assert result.raw_text == "reply"

    def test_blank_line_returns_none(self, parser: LogParser):
        assert parser.parse_line("") is None
        assert parser.parse_line("   ") is None
        assert parser.parse_line("\t") is None

    def test_no_match_line_returns_none(self, parser: LogParser):
        """A continuation line (no timestamp) returns None — parse_file handles these."""
        assert parser.parse_line("just some text") is None


# ── parse_line: User mention extraction ──────────────────────────────────────

class TestUserMentions:
    def test_single_mention(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: check <@U04K8NLDCG0>"
        result = parser.parse_line(line)
        assert result is not None
        assert result.user_mentions == ["U04K8NLDCG0"]

    def test_multiple_mentions(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: cc <@U04K8NLDCG0> and <@U073D278H62>"
        result = parser.parse_line(line)
        assert result is not None
        assert result.user_mentions == ["U04K8NLDCG0", "U073D278H62"]


# ── parse_line: Channel ref extraction ────────────────────────────────────────

class TestChannelRefs:
    def test_channel_ref_with_display(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: posted in <#C06M81FSKFF|general>"
        result = parser.parse_line(line)
        assert result is not None
        assert result.channel_refs == ["C06M81FSKFF"]

    def test_channel_ref_empty_display(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: posted in <#C06M81FSKFF|>"
        result = parser.parse_line(line)
        assert result is not None
        assert result.channel_refs == ["C06M81FSKFF"]

    def test_multiple_channel_refs(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: <#C06M81FSKFF|general> and <#C06M6MATUDB|random>"
        result = parser.parse_line(line)
        assert result is not None
        assert result.channel_refs == ["C06M81FSKFF", "C06M6MATUDB"]


# ── parse_line: Special mention extraction ────────────────────────────────────

class TestSpecialMentions:
    def test_here_mention(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: <!here> important"
        result = parser.parse_line(line)
        assert result is not None
        assert result.special_mentions == ["<!here>"]

    def test_channel_mention(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: <!channel> FYI"
        result = parser.parse_line(line)
        assert result is not None
        assert result.special_mentions == ["<!channel>"]

    def test_everyone_mention(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: <!everyone> all hands"
        result = parser.parse_line(line)
        assert result is not None
        assert result.special_mentions == ["<!everyone>"]

    def test_usergroup_mention(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: <!subteam^S0805PGRHV2> devs"
        result = parser.parse_line(line)
        assert result is not None
        assert result.special_mentions == ["<!subteam^S0805PGRHV2>"]

    def test_multiple_special_mentions(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: <!here> and <!channel> and <!subteam^S0805PGRHV2>"
        result = parser.parse_line(line)
        assert result is not None
        assert "<!here>" in result.special_mentions
        assert "<!channel>" in result.special_mentions
        assert "<!subteam^S0805PGRHV2>" in result.special_mentions

    def test_special_mentions_not_in_user_mentions(self, parser: LogParser):
        """Special mentions must NOT go into user_mentions."""
        line = "[1766775007.615089] @Jin: <!here> <@U04K8NLDCG0>"
        result = parser.parse_line(line)
        assert result is not None
        assert result.user_mentions == ["U04K8NLDCG0"]
        assert result.special_mentions == ["<!here>"]


# ── parse_line: Link extraction ───────────────────────────────────────────────

class TestLinkExtraction:
    def test_simple_link(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: check <https://example.com|example>"
        result = parser.parse_line(line)
        assert result is not None
        assert result.links == [Link(url="https://example.com", display="example")]

    def test_html_entities_decoded(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: <http://example.com?a=1&amp;b=2|link>"
        result = parser.parse_line(line)
        assert result is not None
        assert result.links == [Link(url="http://example.com?a=1&b=2", display="link")]

    def test_html_lt_gt_decoded(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: <http://example.com?a=&lt;x&gt;|link>"
        result = parser.parse_line(line)
        assert result is not None
        assert result.links == [Link(url="http://example.com?a=<x>", display="link")]

    def test_multiple_links(self, parser: LogParser):
        line = "[1766775007.615089] @Jin: <https://a.com|a> and <https://b.com|b>"
        result = parser.parse_line(line)
        assert result is not None
        assert len(result.links) == 2
        assert result.links[0].display == "a"
        assert result.links[1].display == "b"


# ── is_raw_uid / is_bot_message ──────────────────────────────────────────────

class TestRawUidAndBot:
    def test_is_raw_uid_true(self, parser: LogParser):
        assert parser.is_raw_uid("U09NDP0R44Q") is True

    def test_is_raw_uid_false_normal_name(self, parser: LogParser):
        assert parser.is_raw_uid("Jin") is False

    def test_is_raw_uid_false_short(self, parser: LogParser):
        assert parser.is_raw_uid("UABC") is False

    def test_is_bot_message_default(self, parser: LogParser):
        assert parser.is_bot_message("nixi") is True
        assert parser.is_bot_message("Fixi") is True

    def test_is_bot_message_normal_user(self, parser: LogParser):
        assert parser.is_bot_message("Jin") is False
        assert parser.is_bot_message("Kuro") is False

    def test_is_bot_message_raw_uid_not_bot(self, parser: LogParser):
        """Raw UIDs are NOT bots unless they match a bot name."""
        assert parser.is_bot_message("U09NDP0R44Q") is False

    def test_is_bot_message_custom_list(self, parser: LogParser):
        custom = ["CustomBot", "AnotherBot"]
        assert parser.is_bot_message("CustomBot", custom) is True
        assert parser.is_bot_message("Fixi", custom) is False
        assert parser.is_bot_message("Jin", custom) is False


# ── parse_file: Multi-line accumulation ────────────────────────────────────────

class TestParseFile:
    def test_multiline_message(self, parser: LogParser, tmp_path: Path):
        content = "[1766780001.123456] @Riya: line1\nline2\nline3\n"
        path = tmp_path / "test.log"
        path.write_text(content, encoding="utf-8")
        results = parser.parse_file(path)
        assert len(results) == 1
        assert results[0].raw_text == "line1\nline2\nline3"

    def test_multiline_with_new_message(self, parser: LogParser, tmp_path: Path):
        content = "[1766780001.123456] @Riya: first\ncontinued\n[1766780002.789012] @Jin: second\n"
        path = tmp_path / "test.log"
        path.write_text(content, encoding="utf-8")
        results = parser.parse_file(path)
        assert len(results) == 2
        assert results[0].raw_text == "first\ncontinued"
        assert results[1].raw_text == "second"

    def test_file_ending_without_trailing_newline(self, parser: LogParser, tmp_path: Path):
        """Final accumulated ParsedLine must be emitted even without trailing newline."""
        content = "[1766780001.123456] @Riya: last message line"
        path = tmp_path / "test.log"
        path.write_text(content, encoding="utf-8")
        results = parser.parse_file(path)
        assert len(results) == 1
        assert results[0].raw_text == "last message line"

    def test_file_ending_with_multiline_no_trailing_newline(self, parser: LogParser, tmp_path: Path):
        content = "[1766780001.123456] @Riya: line1\nline2"
        path = tmp_path / "test.log"
        path.write_text(content, encoding="utf-8")
        results = parser.parse_file(path)
        assert len(results) == 1
        assert results[0].raw_text == "line1\nline2"


# ── parse_file: skip_thread_lines mode ────────────────────────────────────────

class TestParseFileSkipThreads:
    def test_channel_mode_skips_thread_lines(self, parser: LogParser, tmp_path: Path):
        content = (
            "[1766766571.412779] @Kuro: original\n"
            "[1766775007.615089] (thread:1766766571.412779) @OG: thread reply\n"
            "[1766780001.123456] @Riya: new topic\n"
        )
        path = tmp_path / "test.log"
        path.write_text(content, encoding="utf-8")
        results = parser.parse_file(path, skip_thread_lines=True)
        assert len(results) == 2
        assert results[0].display_name == "Kuro"
        assert results[1].display_name == "Riya"

    def test_thread_mode_keeps_all_lines(self, parser: LogParser, tmp_path: Path):
        content = (
            "[1766766571.412779] @Kuro: original\n"
            "[1766775007.615089] (thread:1766766571.412779) @OG: thread reply\n"
        )
        path = tmp_path / "test.log"
        path.write_text(content, encoding="utf-8")
        results = parser.parse_file(path, skip_thread_lines=False)
        assert len(results) == 2
        assert results[0].display_name == "Kuro"
        assert results[1].display_name == "OG"


class TestParseChannelAndThreadFile:
    def test_parse_channel_file(self, parser: LogParser, tmp_path: Path):
        content = (
            "[1766766571.412779] @Kuro: msg1\n"
            "[1766775007.615089] (thread:1766766571.412779) @OG: skip me\n"
            "[1766780001.123456] @Riya: msg2\n"
        )
        path = tmp_path / "channel.log"
        path.write_text(content, encoding="utf-8")
        results = parser.parse_channel_file(path)
        assert len(results) == 2

    def test_parse_thread_file(self, parser: LogParser, tmp_path: Path):
        content = (
            "[1766766571.412779] @Kuro: original\n"
            "[1766775007.615089] (thread:1766766571.412779) @OG: reply\n"
        )
        path = tmp_path / "thread.log"
        path.write_text(content, encoding="utf-8")
        results = parser.parse_thread_file(path)
        assert len(results) == 2


# ── Real log file parsing ─────────────────────────────────────────────────────

class TestRealLogParsing:
    """Integration test: parse real log files from growgami-slack-logs."""

    @pytest.fixture
    def real_log_dir(self) -> Path:
        """Path to the real slack logs directory."""
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        log_dir = repo_root / "growgami-slack-logs" / "slack_logs"
        return log_dir

    def test_parse_real_channel_file(self, parser: LogParser, real_log_dir: Path):
        """Parse a real channel log without errors."""
        channels = sorted(real_log_dir.iterdir())
        # Find a channel with a .log file
        for chan_dir in channels:
            if not chan_dir.is_dir():
                continue
            log_files = list(chan_dir.glob("*.log"))
            if log_files:
                results = parser.parse_channel_file(log_files[0])
                assert len(results) > 0, f"No lines parsed from {log_files[0]}"
                # All lines should have valid slack_ts
                for r in results:
                    assert r.slack_ts, f"Missing slack_ts in parsed line from {log_files[0]}"
                    assert r.display_name, f"Missing display_name in parsed line from {log_files[0]}"
                return
        pytest.skip("No real log files available for testing")

    def test_parse_real_thread_file(self, parser: LogParser, real_log_dir: Path):
        """Parse a real thread log without errors."""
        channels = sorted(real_log_dir.iterdir())
        for chan_dir in channels:
            if not chan_dir.is_dir():
                continue
            threads_dir = chan_dir / "threads"
            if threads_dir.is_dir():
                thread_files = list(threads_dir.glob("*.log"))
                if thread_files:
                    results = parser.parse_thread_file(thread_files[0])
                    assert len(results) > 0, f"No lines parsed from {thread_files[0]}"
                    for r in results:
                        assert r.slack_ts, f"Missing slack_ts in parsed line from {thread_files[0]}"
                    return
        pytest.skip("No thread log files available for testing")