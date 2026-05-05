"""
Tests for Slack chunk builder helpers (slack_chunks module).

Covers: make_markdown_chunk, make_task_update, make_plan_update,
        make_url_source, TASK_STATUS constants, truncation, and
        invalid status coercion.
"""

import logging

import pytest

from gateway.platforms.slack_chunks import (
    TASK_STATUS_COMPLETE,
    TASK_STATUS_ERROR,
    TASK_STATUS_IN_PROGRESS,
    TASK_STATUS_PENDING,
    _truncate_with_ellipsis,
    make_markdown_chunk,
    make_plan_update,
    make_task_update,
    make_url_source,
)


# ── Constants ────────────────────────────────────────────────────────────

class TestConstants:
    """Status constant values match Slack API spec."""

    def test_pending_value(self):
        assert TASK_STATUS_PENDING == "pending"

    def test_in_progress_value(self):
        assert TASK_STATUS_IN_PROGRESS == "in_progress"

    def test_complete_value(self):
        assert TASK_STATUS_COMPLETE == "complete"

    def test_error_value(self):
        assert TASK_STATUS_ERROR == "error"


# ── make_markdown_chunk ─────────────────────────────────────────────────

class TestMakeMarkdownChunk:
    """make_markdown_chunk returns correct schema."""

    def test_basic(self):
        result = make_markdown_chunk("hello")
        assert result == {"type": "markdown_text", "text": "hello"}

    def test_empty_string(self):
        result = make_markdown_chunk("")
        assert result == {"type": "markdown_text", "text": ""}

    def test_multiline(self):
        text = "line1\nline2\nline3"
        result = make_markdown_chunk(text)
        assert result == {"type": "markdown_text", "text": text}

    def test_long_text_not_truncated(self):
        # markdown_text has no 256-char truncation
        text = "a" * 500
        result = make_markdown_chunk(text)
        assert result["text"] == text
        assert len(result["text"]) == 500


# ── make_task_update ─────────────────────────────────────────────────────

class TestMakeTaskUpdate:
    """make_task_update builds valid task_update chunks."""

    def test_minimal_valid(self):
        result = make_task_update("t1", "Search", "in_progress")
        assert result == {
            "type": "task_update",
            "id": "t1",
            "title": "Search",
            "status": "in_progress",
            "details": "",
            "output": "",
            "sources": [],
        }

    def test_all_fields(self):
        sources = [make_url_source("Google", "https://google.com")]
        result = make_task_update(
            "t2",
            "Search",
            "complete",
            details="Found results",
            output="3 results found",
            sources=sources,
        )
        assert result["type"] == "task_update"
        assert result["id"] == "t2"
        assert result["title"] == "Search"
        assert result["status"] == "complete"
        assert result["details"] == "Found results"
        assert result["output"] == "3 results found"
        assert len(result["sources"]) == 1

    def test_pending_status(self):
        result = make_task_update("t1", "Task", "pending")
        assert result["status"] == "pending"

    def test_error_status(self):
        result = make_task_update("t1", "Task", "error")
        assert result["status"] == "error"

    def test_invalid_status_coerced_to_in_progress(self):
        result = make_task_update("t1", "Task", "unknown_status")
        assert result["status"] == "in_progress"

    def test_invalid_status_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            make_task_update("t1", "Task", "bad_status")
        assert "Invalid task status" in caplog.text
        assert "bad_status" in caplog.text

    def test_invalid_status_never_raises(self):
        # Must not raise ValueError — hot path, crashing would be worse
        for bad in (None, 123, "", "INVALID", "completed"):
            result = make_task_update("t1", "Task", bad)
            assert result["status"] == "in_progress"

    def test_sources_default_empty_list(self):
        result = make_task_update("t1", "Task", "in_progress")
        assert result["sources"] == []

    def test_sources_none_becomes_empty_list(self):
        result = make_task_update("t1", "Task", "in_progress", sources=None)
        assert result["sources"] == []


# ── Truncation ───────────────────────────────────────────────────────────

class TestTruncation:
    """256-char truncation with ellipsis suffix."""

    def _make_long_string(self, length: int) -> str:
        return "x" * length

    def test_under_limit_not_truncated(self):
        text = "short title"
        result = make_task_update("t1", text, "in_progress")
        assert result["title"] == text

    def test_exactly_256_not_truncated(self):
        text = "a" * 256
        result = make_task_update("t1", text, "in_progress")
        assert result["title"] == text
        assert len(result["title"]) == 256

    def test_over_256_truncated_with_ellipsis(self):
        text = "a" * 300
        result = make_task_update("t1", text, "in_progress")
        assert len(result["title"]) == 256
        assert result["title"].endswith("…")
        # 255 chars + 1 ellipsis char = 256
        assert result["title"] == "a" * 255 + "…"

    def test_details_truncated(self):
        text = "b" * 300
        result = make_task_update("t1", "Title", "in_progress", details=text)
        assert len(result["details"]) == 256
        assert result["details"].endswith("…")

    def test_output_truncated(self):
        text = "c" * 300
        result = make_task_update("t1", "Title", "in_progress", output=text)
        assert len(result["output"]) == 256
        assert result["output"].endswith("…")

    def test_truncate_helper_directly(self):
        assert _truncate_with_ellipsis("short", 256) == "short"
        assert _truncate_with_ellipsis("a" * 256, 256) == "a" * 256
        result = _truncate_with_ellipsis("a" * 300, 256)
        assert len(result) == 256
        assert result == "a" * 255 + "…"

    def test_truncate_custom_limit(self):
        result = _truncate_with_ellipsis("a" * 20, 10)
        assert len(result) == 10
        assert result == "a" * 9 + "…"

    def test_empty_string_not_truncated(self):
        result = make_task_update("t1", "Title", "in_progress", details="")
        assert result["details"] == ""


# ── make_url_source ──────────────────────────────────────────────────────

class TestMakeUrlSource:
    """make_url_source builds url source dicts. URLs must NOT be truncated."""

    def test_basic(self):
        result = make_url_source("Click here", "https://example.com")
        assert result == {"type": "url", "text": "Click here", "url": "https://example.com"}

    def test_url_not_truncated(self):
        long_url = "https://example.com/" + "a" * 500
        result = make_url_source("Click", long_url)
        assert result["url"] == long_url
        assert len(result["url"]) > 256  # URL never truncated

    def test_text_not_truncated(self):
        # make_url_source does NOT truncate its text field either
        # (it's a label, not a task_update field)
        long_text = "x" * 300
        result = make_url_source(long_text, "https://example.com")
        assert result["text"] == long_text


# ── make_plan_update ─────────────────────────────────────────────────────

class TestMakePlanUpdate:
    """make_plan_update builds plan_update chunks with title truncation."""

    def test_basic(self):
        result = make_plan_update("My Plan")
        assert result == {"type": "plan_update", "title": "My Plan"}

    def test_title_truncated_at_256(self):
        long_title = "z" * 300
        result = make_plan_update(long_title)
        assert len(result["title"]) == 256
        assert result["title"].endswith("…")
        assert result["title"] == "z" * 255 + "…"

    def test_title_exactly_256_preserved(self):
        title = "a" * 256
        result = make_plan_update(title)
        assert result["title"] == title
        assert len(result["title"]) == 256