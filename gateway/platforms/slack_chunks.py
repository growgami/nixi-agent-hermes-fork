"""
Slack chunk builder helpers for Thinking Steps.

Constructs chunk dicts for the Slack chat.appendStream `chunks` parameter:
- markdown_text chunks
- task_update chunks (with status validation and field truncation)
- plan_update chunks (with title truncation)
- url_source dicts (URLs never truncated)

This module is imported by run.py, NOT by stream_consumer.py.
The consumer stores plain list[dict] — it has no Slack-specific knowledge.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Status constants ──────────────────────────────────────────────────────

TASK_STATUS_PENDING = "pending"
TASK_STATUS_IN_PROGRESS = "in_progress"
TASK_STATUS_COMPLETE = "complete"
TASK_STATUS_ERROR = "error"

_VALID_STATUSES = frozenset({
    TASK_STATUS_PENDING,
    TASK_STATUS_IN_PROGRESS,
    TASK_STATUS_COMPLETE,
    TASK_STATUS_ERROR,
})

# Maximum length for string fields in task_update and plan_update.
# Slack API enforces a 256-char limit per field; we truncate with a
# visible ellipsis indicator so users know content was trimmed.
_MAX_FIELD_LENGTH = 256


# ── Truncation helper ────────────────────────────────────────────────────

def _truncate_with_ellipsis(text: str, max_len: int = _MAX_FIELD_LENGTH) -> str:
    """Truncate text from the end, appending an ellipsis if truncated.

    Preserves strings at or below max_len unchanged.  Strings exceeding
    max_len are trimmed to (max_len - 1) characters + "…" so the total
    length stays within the limit and the ellipsis is a visible indicator
    that content was removed.

    URLs must NEVER be truncated — truncation breaks them.
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


# ── Chunk builders ───────────────────────────────────────────────────────

def make_markdown_chunk(text: str) -> dict:
    """Build a markdown_text chunk dict.

    Args:
        text: Markdown content to display. Not truncated — the
              Slack API handles markdown_text length independently.

    Returns:
        ``{"type": "markdown_text", "text": text}``
    """
    return {"type": "markdown_text", "text": text}


def make_task_update(
    id: str,
    title: str,
    status: str,
    *,
    details: str = "",
    output: str = "",
    sources: Optional[list[dict]] = None,
) -> dict:
    """Build a task_update chunk dict.

    Status must be one of ``pending``, ``in_progress``, ``complete``, or
    ``error``.  Invalid statuses are coerced to ``in_progress`` with a
    WARNING log — this function never raises because it runs on the
    streaming hot path and a ``ValueError`` would crash the consumer.

    String fields ``title``, ``details``, and ``output`` are truncated
    to 256 characters with a trailing ``"…"`` when they exceed the limit.
    ``sources`` URL dicts are NOT truncated (truncating URLs breaks them).

    Args:
        id: Unique identifier for this task (usually the tool call ID).
        title: Human-readable task name. Truncated at 256 chars.
        status: One of ``pending``, ``in_progress``, ``complete``,
                ``error``. Invalid values coerced to ``in_progress``.
        details: Optional detail text. Truncated at 256 chars.
        output: Optional output text. Truncated at 256 chars.
        sources: Optional list of url-source dicts (from
                 :func:`make_url_source`). Not truncated.

    Returns:
        task_update chunk dict ready for the Slack ``chunks`` array.
    """
    if status not in _VALID_STATUSES:
        logger.warning(
            "Invalid task status %r, coercing to 'in_progress'",
            status,
        )
        status = TASK_STATUS_IN_PROGRESS

    return {
        "type": "task_update",
        "id": id,
        "title": _truncate_with_ellipsis(title),
        "status": status,
        "details": _truncate_with_ellipsis(details),
        "output": _truncate_with_ellipsis(output),
        "sources": sources if sources is not None else [],
    }


def make_plan_update(title: str) -> dict:
    """Build a plan_update chunk dict.

    Args:
        title: Plan title. Truncated at 256 chars with ``"…"`` suffix.

    Returns:
        ``{"type": "plan_update", "title": …}``
    """
    return {
        "type": "plan_update",
        "title": _truncate_with_ellipsis(title),
    }


def make_url_source(text: str, url: str) -> dict:
    """Build a url source dict for use in task_update.sources arrays.

    Neither ``text`` nor ``url`` is truncated.  URLs must never be
    truncated — truncation breaks them.

    Args:
        text: Display text for the URL link.
        url: Target URL.

    Returns:
        ``{"type": "url", "text": text, "url": url}``
    """
    return {"type": "url", "text": text, "url": url}