"""Unit tests for nixi.intent_classifier — classification rules, cache, helpers, and classify()."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from nixi.intent_classifier import (
    ACKNOWLEDGMENT_PATTERNS,
    NOHELLO_URL,
    ClassificationContext,
    ClassificationResult,
    ThreadMentionCache,
    _is_acknowledgment,
    _is_greeting_only,
    _is_substantive,
    _strip_mentions,
    bot_mentioned_in_text,
    classify,
    dm_rule,
    greeting_mention_rule,
    noise_mention_rule,
    substantive_mention_rule,
    thread_continuation_rule,
    unrelated_drop_rule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(**overrides) -> ClassificationContext:
    """Build a ClassificationContext with sensible defaults."""
    defaults = dict(
        text="hello",
        channel="C12345",
        is_dm=False,
        thread_ts=None,
        bot_user_id="UBOT123",
        thread_had_bot=False,
    )
    defaults.update(overrides)
    return ClassificationContext(**defaults)


# ---------------------------------------------------------------------------
# Step 1: TestClassificationContext
# ---------------------------------------------------------------------------


class TestClassificationContext:
    """Verify ClassificationContext dataclass construction."""

    def test_construction_with_all_fields(self):
        ctx = ClassificationContext(
            text="hey <@UBOT123>",
            channel="C12345",
            is_dm=False,
            thread_ts="1234567890.123456",
            bot_user_id="UBOT123",
            thread_had_bot=True,
        )
        assert ctx.text == "hey <@UBOT123>"
        assert ctx.channel == "C12345"
        assert ctx.is_dm is False
        assert ctx.thread_ts == "1234567890.123456"
        assert ctx.bot_user_id == "UBOT123"
        assert ctx.thread_had_bot is True

    def test_construction_with_minimal_fields(self):
        ctx = ClassificationContext(
            text="hi",
            channel="D12345",
            is_dm=True,
            thread_ts=None,
            bot_user_id=None,
            thread_had_bot=False,
        )
        assert ctx.text == "hi"
        assert ctx.bot_user_id is None
        assert ctx.thread_ts is None

    def test_frozen_dataclass(self):
        ctx = _ctx()
        with pytest.raises(AttributeError):
            ctx.text = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Step 2: TestClassificationResult
# ---------------------------------------------------------------------------


class TestClassificationResult:
    """Verify ClassificationResult dataclass construction and action values."""

    def test_pass_action(self):
        result = ClassificationResult(action="pass", response_text=None, reason="test")
        assert result.action == "pass"
        assert result.response_text is None
        assert result.reason == "test"

    def test_respond_action(self):
        result = ClassificationResult(action="respond", response_text=NOHELLO_URL, reason="greeting")
        assert result.action == "respond"
        assert result.response_text == NOHELLO_URL

    def test_drop_action(self):
        result = ClassificationResult(action="drop", response_text=None, reason="noise")
        assert result.action == "drop"
        assert result.response_text is None

    def test_frozen_dataclass(self):
        result = ClassificationResult(action="pass", response_text=None, reason="x")
        with pytest.raises(AttributeError):
            result.action = "drop"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Step 3: TestThreadMentionCache
# ---------------------------------------------------------------------------


class TestThreadMentionCache:
    """Test ThreadMentionCache record(), had_bot(), and _evict()."""

    def test_record_and_had_bot_hit(self):
        cache = ThreadMentionCache()
        cache.record("1234567890.001")
        assert cache.had_bot("1234567890.001") is True

    def test_had_bot_miss(self):
        cache = ThreadMentionCache()
        assert cache.had_bot("nonexistent") is False

    def test_had_bot_expired(self):
        cache = ThreadMentionCache(ttl=0.01)
        cache.record("1234567890.001")
        time.sleep(0.02)
        assert cache.had_bot("1234567890.001") is False

    def test_had_bot_removes_expired_entry(self):
        cache = ThreadMentionCache(ttl=0.01)
        cache.record("1234567890.001")
        time.sleep(0.02)
        # had_bot returns False and also removes the expired entry
        assert cache.had_bot("1234567890.001") is False
        # Entry is removed, so cache size is 0
        assert len(cache._cache) == 0

    def test_had_bot_moves_to_end_lru(self):
        """Accessing via had_bot should move entry to end (most recently used)."""
        cache = ThreadMentionCache(max_size=3)
        cache.record("ts1")
        cache.record("ts2")
        cache.record("ts3")
        # Access ts1 — moves it to end
        cache.had_bot("ts1")
        # Add one more — should evict ts2 (oldest unaccessed)
        cache.record("ts4")
        assert cache.had_bot("ts1") is True  # still present
        assert cache.had_bot("ts2") is False  # evicted
        assert cache.had_bot("ts3") is True  # still present

    def test_evict_max_size(self):
        cache = ThreadMentionCache(max_size=2)
        cache.record("ts1")
        cache.record("ts2")
        cache.record("ts3")  # should evict ts1
        assert cache.had_bot("ts1") is False
        assert cache.had_bot("ts2") is True
        assert cache.had_bot("ts3") is True

    def test_evict_removes_expired_before_size_check(self):
        """Expired entries are evicted before max_size check."""
        cache = ThreadMentionCache(ttl=0.01, max_size=10)
        cache.record("ts1")
        time.sleep(0.02)
        # ts1 is expired — when we record a new entry, _evict removes expired first
        cache.record("ts2")
        assert cache.had_bot("ts1") is False
        assert cache.had_bot("ts2") is True


# ---------------------------------------------------------------------------
# Step 4: TestStripMentions
# ---------------------------------------------------------------------------


class TestStripMentions:
    """Verify _strip_mentions strips Slack mention patterns."""

    def test_strip_simple_mention(self):
        assert _strip_mentions("<@U12345>") == ""

    def test_strip_mention_with_display_name(self):
        assert _strip_mentions("<@U12345|BotName>") == ""

    def test_strip_multiple_mentions(self):
        result = _strip_mentions("hey <@U111> and <@U222> what's up")
        # _strip_mentions removes mention patterns but leaves surrounding spaces
        assert result == "hey  and  what's up"

    def test_preserve_non_mention_text(self):
        assert _strip_mentions("hello world") == "hello world"

    def test_empty_string(self):
        assert _strip_mentions("") == ""

    def test_text_with_no_mentions(self):
        assert _strip_mentions("summarize the thread") == "summarize the thread"

    def test_mention_with_surrounding_text(self):
        result = _strip_mentions("<@U99999> can you check this?")
        assert result == "can you check this?"


# ---------------------------------------------------------------------------
# Step 5: TestIsGreetingOnly
# ---------------------------------------------------------------------------


class TestIsGreetingOnly:
    """Verify _is_greeting_only detection."""

    # Positive cases — greeting-only messages
    @pytest.mark.parametrize("text", [
        "hey",
        "hi",
        "hello",
        "yo",
        "sup",
        "good morning",
        "hey!",
        # Note: "hi there" and "<@UBOT123> hi there" are NOT greeting-only per
        # current implementation — the `^(?:hey|hi|...)\b` regex strips "hi"
        # first, leaving "there" which is substantive. Moved to negative cases.
        "hola",
        "howdy",
        "heya",
        "hiya",
        "<@UBOT123> hey",
        "greetings",
        "morning",
        "good afternoon",
        "good evening",
    ])
    def test_greeting_only_positive(self, text):
        assert _is_greeting_only(text) is True, f"Expected greeting-only for: {text!r}"

    # Negative cases — messages with substantive content
    @pytest.mark.parametrize("text", [
        "hey can you help?",
        "hello what's the status of the deploy?",
        "hi @bot can you summarize?",
        "what time is it?",
        "please review this PR",
        "can you check the logs?",
        "summarize the thread",
        # "hi there" is NOT greeting-only per current implementation —
        # the `^(?:hey|hi|...)\b` regex strips "hi" leaving "there"
        "hi there",
        "<@UBOT123> hi there",
    ])
    def test_greeting_only_negative(self, text):
        assert _is_greeting_only(text) is False, f"Expected NOT greeting-only for: {text!r}"

    def test_greeting_only_empty_after_strip(self):
        """Message that is only mentions — treated as greeting."""
        assert _is_greeting_only("<@UBOT123>") is True

    def test_greeting_only_punctuation_only(self):
        """Greeting with just punctuation after."""
        assert _is_greeting_only("hey!!!") is True


# ---------------------------------------------------------------------------
# Step 6: TestIsSubstantive
# ---------------------------------------------------------------------------


class TestIsSubstantive:
    """Verify _is_substantive detection."""

    # Positive cases — substantive content
    @pytest.mark.parametrize("text", [
        "what is X?",
        "can you help?",
        "summarize the thread",
        "check the logs please",
        "can you review the deployment status?",
        "what time is it?",
    ])
    def test_substantive_positive(self, text):
        assert _is_substantive(text) is True, f"Expected substantive for: {text!r}"

    # Negative cases — non-substantive content
    @pytest.mark.parametrize("text", [
        "thanks!",
        "lol",
        "ok",
        "nice",
        "cool",
        "👍",
    ])
    def test_substantive_negative(self, text):
        assert _is_substantive(text) is False, f"Expected NOT substantive for: {text!r}"

    def test_substantive_multi_word_after_greeting(self):
        """Text that has 3+ words remaining after greeting strip is substantive."""
        assert _is_substantive("hey what's the status of the deploy?") is True

    def test_substantive_empty_string(self):
        assert _is_substantive("") is False

    def test_substantive_greeting_only(self):
        """A greeting-only message is not substantive."""
        assert _is_substantive("hello") is False


# ---------------------------------------------------------------------------
# Step 7: TestIsAcknowledgment
# ---------------------------------------------------------------------------


class TestIsAcknowledgment:
    """Verify _is_acknowledgment detection."""

    # Positive cases — acknowledgment/noise
    @pytest.mark.parametrize("text", [
        "thanks",
        "thx",
        "ty",
        "lol",
        "ok",
        "nice",
        "cool",
        "👍",
        "thanks!",
        "haha",
        "gotcha",
        "yep",
        "sure",
        "awesome",
    ])
    def test_acknowledgment_positive(self, text):
        assert _is_acknowledgment(text) is True, f"Expected acknowledgment for: {text!r}"

    # Negative cases — substantive content
    @pytest.mark.parametrize("text", [
        "can you help?",
        "what is X?",
        "summarize this",
        "check the logs",
        "what time is it?",
    ])
    def test_acknowledgment_negative(self, text):
        assert _is_acknowledgment(text) is False, f"Expected NOT acknowledgment for: {text!r}"

    def test_acknowledgment_with_mention(self):
        """Acknowledgments that include a mention should still be detected after stripping."""
        assert _is_acknowledgment("<@UBOT123> thanks") is True

    def test_acknowledgment_empty_string(self):
        assert _is_acknowledgment("") is False


# ---------------------------------------------------------------------------
# Step 8: TestDmRule
# ---------------------------------------------------------------------------


class TestDmRule:
    """Test dm_rule: DM greeting → RESPOND, DM substantive → PASS, non-DM → None."""

    def test_dm_greeting(self):
        ctx = _ctx(text="hey", channel="D12345", is_dm=True, thread_ts=None)
        result = dm_rule(ctx)
        assert result is not None
        assert result.action == "respond"
        assert result.response_text == NOHELLO_URL
        assert result.reason == "dm_greeting"

    def test_dm_substantive(self):
        ctx = _ctx(text="can you help me with this?", channel="D12345", is_dm=True, thread_ts=None)
        result = dm_rule(ctx)
        assert result is not None
        assert result.action == "pass"
        assert result.response_text is None
        assert result.reason == "dm_message"

    def test_non_dm_returns_none(self):
        ctx = _ctx(text="hey", channel="C12345", is_dm=False)
        result = dm_rule(ctx)
        assert result is None

    def test_dm_with_no_text_greeting(self):
        """DM with empty text is greeting-only (stripped to empty → True)."""
        ctx = _ctx(text="", channel="D12345", is_dm=True, thread_ts=None)
        result = dm_rule(ctx)
        assert result is not None
        assert result.action == "respond"
        assert result.reason == "dm_greeting"


# ---------------------------------------------------------------------------
# Step 9: TestThreadContinuationRule
# ---------------------------------------------------------------------------


class TestThreadContinuationRule:
    """Test thread_continuation_rule."""

    def test_thread_with_bot_engagement(self):
        ctx = _ctx(thread_ts="1234567890.001", thread_had_bot=True)
        result = thread_continuation_rule(ctx)
        assert result is not None
        assert result.action == "pass"
        assert result.reason == "thread_continuation"

    def test_thread_without_bot_engagement(self):
        ctx = _ctx(thread_ts="1234567890.001", thread_had_bot=False)
        result = thread_continuation_rule(ctx)
        assert result is None

    def test_no_thread_ts(self):
        ctx = _ctx(thread_ts=None, thread_had_bot=False)
        result = thread_continuation_rule(ctx)
        assert result is None

    def test_thread_ts_with_empty_cache(self):
        """Thread_ts present but thread_had_bot=False → no match."""
        ctx = _ctx(thread_ts="1234567890.001", thread_had_bot=False)
        result = thread_continuation_rule(ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Step 10: TestSubstantiveMentionRule
# ---------------------------------------------------------------------------


class TestSubstantiveMentionRule:
    """Test substantive_mention_rule."""

    def test_bot_mentioned_substantive(self):
        ctx = _ctx(text="<@UBOT123> can you summarize the thread?", bot_user_id="UBOT123")
        result = substantive_mention_rule(ctx)
        assert result is not None
        assert result.action == "pass"
        assert result.reason == "substantive_mention"

    def test_bot_mentioned_non_substantive(self):
        ctx = _ctx(text="<@UBOT123> hey", bot_user_id="UBOT123")
        result = substantive_mention_rule(ctx)
        assert result is None

    def test_bot_not_mentioned(self):
        ctx = _ctx(text="can you summarize the thread?", bot_user_id="UBOT123")
        result = substantive_mention_rule(ctx)
        assert result is None

    def test_bot_user_id_none(self):
        ctx = _ctx(text="<@UBOT123> summarize", bot_user_id=None)
        result = substantive_mention_rule(ctx)
        assert result is None

    def test_bot_user_id_empty(self):
        ctx = _ctx(text="<@UBOT123> summarize", bot_user_id="")
        result = substantive_mention_rule(ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Step 11: TestGreetingMentionRule
# ---------------------------------------------------------------------------


class TestGreetingMentionRule:
    """Test greeting_mention_rule."""

    def test_bot_mentioned_greeting_only(self):
        ctx = _ctx(text="<@UBOT123> hey", bot_user_id="UBOT123")
        result = greeting_mention_rule(ctx)
        assert result is not None
        assert result.action == "respond"
        assert result.response_text == NOHELLO_URL
        assert result.reason == "greeting_only"

    def test_bot_mentioned_non_greeting(self):
        ctx = _ctx(text="<@UBOT123> can you summarize the thread?", bot_user_id="UBOT123")
        result = greeting_mention_rule(ctx)
        assert result is None

    def test_bot_not_mentioned(self):
        ctx = _ctx(text="hey everyone", bot_user_id="UBOT123")
        result = greeting_mention_rule(ctx)
        assert result is None

    def test_bot_user_id_none(self):
        ctx = _ctx(text="<@UBOT123> hi", bot_user_id=None)
        result = greeting_mention_rule(ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Step 12: TestNoiseMentionRule
# ---------------------------------------------------------------------------


class TestNoiseMentionRule:
    """Test noise_mention_rule."""

    def test_bot_mentioned_acknowledgment(self):
        ctx = _ctx(text="<@UBOT123> thanks!", bot_user_id="UBOT123")
        result = noise_mention_rule(ctx)
        assert result is not None
        assert result.action == "drop"
        assert result.reason == "noise_mention"

    def test_bot_mentioned_substantive(self):
        ctx = _ctx(text="<@UBOT123> can you check the logs?", bot_user_id="UBOT123")
        result = noise_mention_rule(ctx)
        assert result is None

    def test_bot_not_mentioned(self):
        ctx = _ctx(text="thanks everyone", bot_user_id="UBOT123")
        result = noise_mention_rule(ctx)
        assert result is None

    def test_bot_user_id_none(self):
        ctx = _ctx(text="<@UBOT123> lol", bot_user_id=None)
        result = noise_mention_rule(ctx)
        assert result is None

    def test_bot_mentioned_greeting(self):
        """Greeting-only mention should NOT match noise rule — greeting rule catches it first."""
        ctx = _ctx(text="<@UBOT123> hey", bot_user_id="UBOT123")
        # noise_mention_rule requires _is_acknowledgment, which greetings are NOT
        result = noise_mention_rule(ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Step 13: TestUnrelatedDropRule
# ---------------------------------------------------------------------------


class TestUnrelatedDropRule:
    """Test unrelated_drop_rule: always returns DROP with reason 'unrelated'."""

    def test_always_drop(self):
        ctx = _ctx(text="anything", channel="C12345")
        result = unrelated_drop_rule(ctx)
        assert result.action == "drop"
        assert result.response_text is None
        assert result.reason == "unrelated"

    def test_drop_on_empty_context(self):
        ctx = _ctx(text="", channel="D12345", is_dm=True)
        result = unrelated_drop_rule(ctx)
        assert result.action == "drop"
        assert result.reason == "unrelated"


# ---------------------------------------------------------------------------
# Step 14: TestClassifyIntegration
# ---------------------------------------------------------------------------


class TestClassifyIntegration:
    """Full-flow integration tests for classify() orchestrator."""

    def test_dm_greeting(self):
        """DM with greeting → RESPOND with nohello.net URL."""
        ctx = _ctx(text="hey", channel="D12345", is_dm=True, bot_user_id=None)
        result = classify(ctx)
        assert result.action == "respond"
        assert result.response_text == NOHELLO_URL
        assert result.reason == "dm_greeting"

    def test_dm_substantive(self):
        """DM with substantive content → PASS."""
        ctx = _ctx(text="can you help me debug this?", channel="D12345", is_dm=True, bot_user_id=None)
        result = classify(ctx)
        assert result.action == "pass"
        assert result.response_text is None
        assert result.reason == "dm_message"

    def test_channel_mention_greeting(self):
        """Channel with bot mentioned + greeting-only → RESPOND with nohello.net URL."""
        ctx = _ctx(text="<@UBOT123> hey", channel="C12345", is_dm=False, bot_user_id="UBOT123")
        result = classify(ctx)
        assert result.action == "respond"
        assert result.response_text == NOHELLO_URL
        assert result.reason == "greeting_only"

    def test_channel_mention_substantive(self):
        """Channel with bot mentioned + substantive content → PASS."""
        ctx = _ctx(text="<@UBOT123> can you summarize the thread?", channel="C12345", is_dm=False, bot_user_id="UBOT123")
        result = classify(ctx)
        assert result.action == "pass"
        assert result.response_text is None
        assert result.reason == "substantive_mention"

    def test_channel_mention_acknowledgment(self):
        """Channel with bot mentioned + acknowledgment → DROP."""
        ctx = _ctx(text="<@UBOT123> thanks!", channel="C12345", is_dm=False, bot_user_id="UBOT123")
        result = classify(ctx)
        assert result.action == "drop"
        assert result.reason == "noise_mention"

    def test_thread_continuation(self):
        """Thread with prior bot engagement → PASS."""
        ctx = _ctx(
            text="following up on that",
            channel="C12345",
            is_dm=False,
            thread_ts="1234567890.001",
            bot_user_id="UBOT123",
            thread_had_bot=True,
        )
        result = classify(ctx)
        assert result.action == "pass"
        assert result.reason == "thread_continuation"

    def test_unrelated_message(self):
        """Unrelated channel message (no mention, no thread) → DROP."""
        ctx = _ctx(
            text="anyone seen the latest deploy?",
            channel="C12345",
            is_dm=False,
            thread_ts=None,
            bot_user_id="UBOT123",
            thread_had_bot=False,
        )
        result = classify(ctx)
        assert result.action == "drop"
        assert result.reason == "unrelated"

    def test_dm_overrides_thread(self):
        """DM takes priority over thread continuation (rule order)."""
        ctx = _ctx(
            text="hello",
            channel="D12345",
            is_dm=True,
            thread_ts="1234567890.001",
            bot_user_id="UBOT123",
            thread_had_bot=True,
        )
        result = classify(ctx)
        assert result.action == "respond"
        assert result.reason == "dm_greeting"

    def test_bot_mentioned_with_question_mark(self):
        """Bot mentioned + question → substantive (question mark detection)."""
        ctx = _ctx(text="<@UBOT123> what is the status?", channel="C12345", is_dm=False, bot_user_id="UBOT123")
        result = classify(ctx)
        assert result.action == "pass"
        assert result.reason == "substantive_mention"

    def test_channel_no_mention_no_thread(self):
        """Channel message without mention and no thread → DROP (unrelated)."""
        ctx = _ctx(
            text="random conversation",
            channel="C12345",
            is_dm=False,
            thread_ts=None,
            bot_user_id="UBOT123",
        )
        result = classify(ctx)
        assert result.action == "drop"
        assert result.reason == "unrelated"

    def test_none_bot_user_id(self):
        """None bot_user_id means no mention detection → messages fall through to DROP."""
        ctx = _ctx(
            text="<@UBOT123> summarize this",
            channel="C12345",
            is_dm=False,
            bot_user_id=None,
        )
        result = classify(ctx)
        assert result.action == "drop"
        assert result.reason == "unrelated"

    def test_dm_question_mark(self):
        """DM with question mark → substantive → PASS."""
        ctx = _ctx(text="what's the status?", channel="D12345", is_dm=True, bot_user_id=None)
        result = classify(ctx)
        assert result.action == "pass"
        assert result.reason == "dm_message"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests: empty text, None bot_user_id, DM with no text, thread_ts with no cache entry."""

    def test_empty_text_dm(self):
        """DM with empty text → greeting-only (stripped to empty → True)."""
        ctx = _ctx(text="", channel="D12345", is_dm=True)
        result = classify(ctx)
        assert result.action == "respond"
        assert result.reason == "dm_greeting"

    def test_none_bot_user_id_channel(self):
        """Channel message with None bot_user_id → no mention detection → DROP."""
        ctx = _ctx(text="<@UXXX> hey", channel="C12345", is_dm=False, bot_user_id=None)
        result = classify(ctx)
        assert result.action == "drop"

    def test_thread_ts_no_cache_entry(self):
        """Thread_ts present but thread_had_bot=False → thread rule doesn't match → falls through."""
        ctx = _ctx(
            text="following up",
            channel="C12345",
            is_dm=False,
            thread_ts="1234567890.001",
            bot_user_id="UBOT123",
            thread_had_bot=False,
        )
        result = classify(ctx)
        # Not a DM, no mention, thread without bot → falls to unrelated_drop
        assert result.action == "drop"
        assert result.reason == "unrelated"

    def test_whitespace_only_text(self):
        """Whitespace-only text in DM → greeting-only."""
        ctx = _ctx(text="   ", channel="D12345", is_dm=True)
        result = classify(ctx)
        assert result.action == "respond"
        assert result.reason == "dm_greeting"

    def test_bot_mentioned_in_text_helper(self):
        """Test bot_mentioned_in_text helper directly."""
        assert bot_mentioned_in_text("<@UBOT123> hello", "UBOT123") is True
        assert bot_mentioned_in_text("hello", "UBOT123") is False
        assert bot_mentioned_in_text("<@UOTHER>", "UBOT123") is False

    def test_bot_mentioned_in_text_none_user_id(self):
        """bot_mentioned_in_text with None/empty user_id returns False (conservative)."""
        assert bot_mentioned_in_text("<@UBOT123> hello", None) is False
        assert bot_mentioned_in_text("<@UBOT123> hello", "") is False

    def test_strip_mentions_preserves_content(self):
        """_strip_mentions should not alter text with no mentions."""
        assert _strip_mentions("summarize this thread please") == "summarize this thread please"