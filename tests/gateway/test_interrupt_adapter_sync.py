"""Tests verifying that the interrupt handler synchronizes both pending stores.

When the interrupt handler writes text to runner._pending_messages, it must
also populate adapter._pending_messages with the full MessageEvent so that
_dequeue_pending_event (which reads from adapter) finds the event with all
metadata intact.

Regression test for a bug where only the runner's text-only store was updated,
so _dequeue_pending_event returned None and the follow-up was silently dropped.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Minimal module stubs so we can import gateway code without heavy deps
# ---------------------------------------------------------------------------
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    merge_pending_message_event,
)
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubAdapter(BasePlatformAdapter):
    """Minimal adapter for testing pending-store sync."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _make_event(text="hello", chat_id="123456", message_id="msg1"):
    """Build a minimal MessageEvent."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInterruptAdapterSync:
    """When interrupt handler writes to runner._pending_messages, adapter
    store must also be populated with the full MessageEvent."""

    def test_merge_pending_message_event_stores_full_event(self):
        """merge_pending_message_event stores a MessageEvent with all metadata."""
        adapter = _StubAdapter()
        event = _make_event(text="follow-up question", message_id="msg42")
        sk = build_session_key(event.source)

        merge_pending_message_event(adapter._pending_messages, sk, event)

        assert sk in adapter._pending_messages
        stored = adapter._pending_messages[sk]
        assert isinstance(stored, MessageEvent)
        assert stored.text == "follow-up question"
        assert stored.message_id == "msg42"
        assert stored.source.platform == Platform.TELEGRAM

    def test_merge_pending_message_event_merges_text(self):
        """merge_pending_message_event with merge_text=True appends text."""
        adapter = _StubAdapter()
        event1 = _make_event(text="first", message_id="m1")
        sk = build_session_key(event1.source)

        merge_pending_message_event(adapter._pending_messages, sk, event1)

        event2 = _make_event(text="second", message_id="m2")
        merge_pending_message_event(
            adapter._pending_messages, sk, event2, merge_text=True
        )

        stored = adapter._pending_messages[sk]
        assert "first" in stored.text
        assert "second" in stored.text
        assert stored.message_id == "m1"  # original preserved

    def test_dequeue_finds_event_after_merge(self):
        """After merge_pending_message_event, adapter.get_pending_message
        returns the full MessageEvent."""
        adapter = _StubAdapter()
        event = _make_event(text="dequeue me", message_id="dm1")
        sk = build_session_key(event.source)

        merge_pending_message_event(adapter._pending_messages, sk, event)

        dequeued = adapter.get_pending_message(sk)
        assert dequeued is not None
        assert isinstance(dequeued, MessageEvent)
        assert dequeued.text == "dequeue me"
        assert dequeued.message_id == "dm1"
        # get_pending_message pops, so second call returns None
        assert adapter.get_pending_message(sk) is None

    def test_interrupt_sync_logic_no_adapter(self):
        """Simulating the interrupt handler: when adapter is None,
        runner._pending_messages still gets text without crash."""
        runner_pending = {}
        event = _make_event(text="orphan message")
        sk = build_session_key(event.source)

        # Simulate what the interrupt handler does for runner store
        if sk in runner_pending:
            runner_pending[sk] += "\n" + event.text
        else:
            runner_pending[sk] = event.text

        # Simulate what the adapter sync does when adapter is None
        adapter = None
        if adapter:
            merge_pending_message_event(adapter._pending_messages, sk, event)

        # runner store should still have the text
        assert sk in runner_pending
        assert runner_pending[sk] == "orphan message"

    def test_interrupt_sync_logic_with_adapter(self):
        """Simulating the interrupt handler: when adapter exists,
        both stores are populated."""
        runner_pending = {}
        adapter = _StubAdapter()
        event = _make_event(text="follow-up question", message_id="msg1")
        sk = build_session_key(event.source)

        # 1. Write text to runner store (existing behavior)
        if sk in runner_pending:
            runner_pending[sk] += "\n" + event.text
        else:
            runner_pending[sk] = event.text

        # 2. Synchronize adapter store with full event (NEW behavior)
        merge_pending_message_event(adapter._pending_messages, sk, event)

        # Verify runner store has text
        assert sk in runner_pending
        assert runner_pending[sk] == "follow-up question"

        # Verify adapter store has full MessageEvent
        assert sk in adapter._pending_messages
        stored = adapter._pending_messages[sk]
        assert isinstance(stored, MessageEvent)
        assert stored.text == "follow-up question"
        assert stored.message_id == "msg1"
        assert stored.source.platform == Platform.TELEGRAM

    def test_interrupt_sync_logic_merge_text(self):
        """Second interrupt merges text in runner store and replaces event
        in adapter store."""
        runner_pending = {}
        adapter = _StubAdapter()
        sk = build_session_key(_make_event().source)

        event1 = _make_event(text="first message", message_id="m1")
        event2 = _make_event(text="second message", message_id="m2")

        # First interrupt
        if sk in runner_pending:
            runner_pending[sk] += "\n" + event1.text
        else:
            runner_pending[sk] = event1.text
        merge_pending_message_event(adapter._pending_messages, sk, event1)

        # Second interrupt
        if sk in runner_pending:
            runner_pending[sk] += "\n" + event2.text
        else:
            runner_pending[sk] = event2.text
        merge_pending_message_event(adapter._pending_messages, sk, event2)

        # Runner store has both messages merged
        assert "first message" in runner_pending[sk]
        assert "second message" in runner_pending[sk]

        # Adapter store has the latest event (replaced, not merged text)
        stored = adapter._pending_messages[sk]
        assert isinstance(stored, MessageEvent)
        assert stored.text == "second message"

    def test_dequeue_round_trip(self):
        """Full round-trip: interrupt writes → dequeue reads MessageEvent
        with all metadata intact."""
        from gateway.run import _dequeue_pending_event

        adapter = _StubAdapter()
        event = _make_event(text="round trip", message_id="rt1")
        sk = build_session_key(event.source)

        merge_pending_message_event(adapter._pending_messages, sk, event)

        dequeued = _dequeue_pending_event(adapter, sk)
        assert dequeued is not None
        assert isinstance(dequeued, MessageEvent)
        assert dequeued.text == "round trip"
        assert dequeued.message_id == "rt1"
        assert dequeued.source.platform == Platform.TELEGRAM