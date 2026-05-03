"""Pre-LLM intent classifier — gates messages before they reach the main agent.

Classification flow:
    1. Construct ClassificationContext from the Slack event
    2. classify() evaluates ordered rules until one returns non-None
    3. Result action determines dispatch:
       - "pass"    → continue to overlay load + LLM (normal flow)
       - "respond" → send response_text directly, skip LLM
       - "drop"    → discard the message entirely

Rule order matters. First non-None result wins. Fallthrough = DROP.

Rules:
    dm_rule                 — DMs: greeting-only → RESPOND (nohello.net),
                              substantive → PASS
    thread_continuation_rule — Thread with prior bot engagement → PASS
    substantive_mention_rule — Bot mentioned + substantive content → PASS
    greeting_mention_rule    — Bot mentioned + greeting-only → RESPOND (nohello.net)
    noise_mention_rule       — Bot mentioned + acknowledgment/noise → DROP
    unrelated_drop_rule     — Catch-all → DROP

Bot mention detection:
    Mention-dependent rules recognize bot invocations via both:
    - <@USERID> Slack mentions (via bot_mentioned_in_text)
    - Natural-language name mentions (via bot_name_mentioned)
    The ``bot_names`` field on ClassificationContext triggers name-based
    matching. Adapters should always include "nixi" as a default.

ThreadMentionCache tracks which threads the bot has engaged in,
enabling thread continuation detection without Slack API calls.
"""

from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationContext:
    """Input to the classifier — all fields extracted from the Slack event.

    Fields:
        text:         Raw message text (may contain <@U...> mentions).
        channel:      Slack channel ID (e.g. "C01234" or "D01234" for DMs).
        is_dm:        True when the message arrived via DM channel.
        thread_ts:    Thread parent timestamp, or None if not in a thread.
        bot_user_id:  The bot's Slack user ID (e.g. "U0ABCDE12"), or None if
                      NIXI_BOT_USER_ID is unset.
        thread_had_bot: True when ThreadMentionCache records prior bot
                      engagement in this thread.
        bot_names:    Tuple of bot display names for natural-language mention
                      detection (e.g. ("nixi",)). When non-empty,
                      triggers name-based matching in mention-dependent rules
                      alongside <@USERID> matching. Default empty tuple for
                      backward compatibility. Adapters should always include
                      "nixi" as a default.
    """

    text: str
    channel: str
    is_dm: bool
    thread_ts: Optional[str]
    bot_user_id: Optional[str]
    thread_had_bot: bool
    bot_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClassificationResult:
    """Output of the classifier.

    Fields:
        action:        "pass" (forward to LLM), "respond" (send reply directly),
                       or "drop" (discard silently).
        response_text: Text to send when action is "respond", None otherwise.
        reason:        Human-readable label for observability/logging.
    """

    action: Literal["pass", "respond", "drop"]
    response_text: Optional[str]
    reason: str


# ---------------------------------------------------------------------------
# Thread mention cache
# ---------------------------------------------------------------------------


class ThreadMentionCache:
    """Lightweight in-memory cache for tracking bot engagement in threads.

    Records thread_ts values where the bot was mentioned or responded.
    Supports TTL-based expiry and max-size eviction (LRU by insertion order).

    Not thread-safe — callers (gateway_adapter) run in the asyncio event loop
    where operations are sequential between await points.
    """

    def __init__(self, ttl: float = 1800, max_size: int = 1024) -> None:
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._ttl = ttl
        self._max_size = max_size

    def record(self, thread_ts: str) -> None:
        """Record that the bot was engaged in a thread."""
        self._cache[thread_ts] = time.time()
        self._evict()

    def had_bot(self, thread_ts: str) -> bool:
        """Return True if the bot was recently engaged in this thread."""
        if thread_ts not in self._cache:
            return False
        ts = self._cache[thread_ts]
        if time.time() - ts >= self._ttl:
            del self._cache[thread_ts]
            return False
        # Move to end (most recently accessed)
        self._cache.move_to_end(thread_ts)
        return True

    def _evict(self) -> None:
        """Remove expired entries, then evict oldest entries if still over max_size."""
        now = time.time()
        # First pass: remove all expired entries
        expired_keys = [
            k for k, ts in self._cache.items() if now - ts >= self._ttl
        ]
        for k in expired_keys:
            del self._cache[k]

        # Second pass: if still over limit, evict oldest by insertion order
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Pattern constants
# ---------------------------------------------------------------------------

NOHELLO_URL = "https://nohello.net"

# Slack mention patterns: <@U12345> and <@U12345|DisplayName>
_MENTION_RE = re.compile(r"<@U[A-Z0-9]+(?:\|[^>]+)?>")

# Greeting patterns — compiled regex for flexible matching.
# Matches standalone greetings that leave no substantive content.
_GREETING_WORDS: frozenset[str] = frozenset(
    {
        "hi", "hey", "hello", "hola", "yo", "sup", "howdy",
        "greetings", "heya", "hiya", "morning", "afternoon", "evening",
    }
)

# Flexible greeting regex: matches common greeting phrases
# including "good morning", "hey there", "hi bot", "hello!", etc.
GREETING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:good\s+)?(?:morning|afternoon|evening)\b", re.IGNORECASE),
    re.compile(r"^(?:hey|hi|hello|hola|yo|sup|howdy|heya|hiya|greetings)\b", re.IGNORECASE),
    re.compile(r"\b(?:hey|hi)\s+there\b", re.IGNORECASE),
]

# Acknowledgment patterns — responses that acknowledge but don't request action.
ACKNOWLEDGMENT_PATTERNS: frozenset[str] = frozenset(
    {
        # Verbal acknowledgments
        "thanks", "thank", "thx", "ty", "thnks", "tnx", "thanx",
        "ok", "okay", "k", "kk", "kewl",
        "nice", "cool", "awesome", "great", "sweet", "dope", "sick",
        "lol", "haha", "hehe", "lmao", "rofl", "lmfao",
        "gotcha", "got it", "understood", "makes sense",
        "yep", "yup", "yes", "yeah", "ya",
        "nope", "no", "nah",
        "alright", "aight", "fine",
        "right", "sure", "absolutely", "definitely",
        "wow", "omg", "jeez",
        # Emoji acknowledgments
        "\U0001f44d",   # 👍
        "\U0001f64f",   # 🙏
        "\U0001f60a",   # 😊
        "\U0001f602",   # 😂
        "\u2764\ufe0f",  # ❤️
        "\U0001f389",   # 🎉
    }
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _strip_mentions(text: str) -> str:
    """Remove all Slack mention patterns (<@U...> and <@U...|Name>) from text."""
    return _MENTION_RE.sub("", text).strip()


def bot_mentioned_in_text(text: str, bot_user_id: str) -> bool:
    """Return True if the bot is directly mentioned in text.

    Checks for ``<@{bot_user_id}>`` in the message text.
    Returns False if bot_user_id is empty/None — conservative:
    no false positives when the ID is not configured.
    """
    if not bot_user_id:
        return False
    return f"<@{bot_user_id}>" in text


def bot_name_mentioned(text: str, bot_names: tuple[str, ...]) -> bool:
    """Return True if any bot name is mentioned in text, using lenient matching.

    For names with 3+ characters, matches the bot name's **prefix** (first
    N-1 characters) followed by zero or more repetitions of the last character,
    bounded by word boundaries. This recognises casual typing variations like
    "nix" for "nixi" or "nixiiiii" for excited typing, while rejecting
    false-positive substrings like "nixification".

    Algorithm per name:
        - Length < 3: use exact word-boundary match ``\\b{name}\\b``
        - Length ≥ 3: prefix = name[:-1], last_char = name[-1].
          Pattern: ``\\b{prefix}{last_char}*\\b`` with ``re.IGNORECASE``.

    Examples with bot name "nixi":
        - "nix"   → matches (3-char prefix, zero trailing i's)    ✅
        - "Nixi"  → matches (case-insensitive exact)              ✅
        - "nixi"  → matches (exact)                                ✅
        - "nixiiiii" → matches (excited typing)                    ✅
        - "nixification" → no match (word boundary fails after i) ❌

    Args:
        text:      The message text to search.
        bot_names: Tuple of bot display names to look for.

    Returns:
        True if any name matches, False otherwise.
        Returns False when bot_names is empty (conservative — no false positives).

    Note:
        Names containing non-word characters (hyphens, apostrophes) may not
        match correctly with ``\\b`` boundaries since ``\\b`` transitions
        between ``\\w`` and ``\\W`` chars. For such names, a different boundary
        strategy may be needed in the future.
    """
    if not bot_names:
        return False
    parts: list[str] = []
    for name in bot_names:
        if len(name) < 3:
            # Short names: exact word-boundary match only
            parts.append(re.escape(name))
        else:
            # Lenient prefix match: prefix + last_char repeated zero or more times
            prefix = re.escape(name[:-1])
            last_char = re.escape(name[-1])
            parts.append(f"{prefix}{last_char}*")
    pattern = re.compile(
        r"\b(" + "|".join(parts) + r")\b",
        re.IGNORECASE,
    )
    return bool(pattern.search(text))


def _is_greeting_only(text: str) -> bool:
    """Return True if text is a greeting with no substantive content.

    A message is "greeting-only" if, after stripping mentions and matching
    greeting patterns, the remaining content is empty or trivial (just
    punctuation, just the bot name repeated).
    """
    stripped = _strip_mentions(text).strip()
    if not stripped:
        # Message was only mentions — treat as greeting
        return True

    # Remove greeting patterns from the beginning of the text
    remaining = stripped
    for pattern in GREETING_PATTERNS:
        remaining = pattern.sub("", remaining).strip()

    # Remove common interjections and punctuation-only remains
    remaining = remaining.strip().rstrip("!.?,").strip()

    # After removing greeting patterns, check what's left
    if not remaining:
        return True

    # Check if what remains is just the bot name or a greeting word
    remaining_lower = remaining.lower()
    if remaining_lower in _GREETING_WORDS:
        return True

    # Check if the original text matches any greeting pattern and is short
    # enough to be considered greeting-only (<=2 words after mention strip)
    words = stripped.split()
    if len(words) <= 2:
        all_greeting = all(w.lower().rstrip("!.?,").lstrip("@#") in _GREETING_WORDS for w in words)
        if all_greeting:
            return True

    return False


def _is_substantive(text: str) -> bool:
    """Return True if text contains an actionable request or substantive content.

    Substantive means the text has: question marks, imperative verbs, or
    sufficient non-greeting content (3+ words after greeting removal).
    """
    stripped = _strip_mentions(text).strip()
    if not stripped:
        return False

    # Question marks indicate a request
    if "?" in stripped:
        return True

    # Check word count after removing common greeting prefixes
    remaining = stripped
    for pattern in GREETING_PATTERNS:
        remaining = pattern.sub("", remaining).strip()

    remaining = remaining.strip().rstrip("!.?,").strip()
    if not remaining:
        return False

    words = remaining.split()
    if len(words) >= 3:
        return True

    return False


def _is_acknowledgment(text: str) -> bool:
    """Return True if text is a non-actionable acknowledgment (thanks, lol, ok, etc.).

    Strips mentions first, then checks if the remaining text (after stripping
    punctuation) matches known acknowledgment patterns.
    """
    stripped = _strip_mentions(text).strip()
    if not stripped:
        return False

    # Normalize: strip punctuation, lowercase
    normalized = stripped.lower().rstrip("!.?,").strip()

    # Single-word or short-phrase acknowledgments
    if normalized in ACKNOWLEDGMENT_PATTERNS:
        return True

    # Multi-word check: all words are acknowledgments or trivial
    words = normalized.split()
    if len(words) <= 3:
        return all(w.rstrip("!.?,") in ACKNOWLEDGMENT_PATTERNS for w in words)

    return False


# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------


def dm_rule(ctx: ClassificationContext) -> Optional[ClassificationResult]:
    """DM messages: greeting-only → RESPOND with nohello.net, else → PASS."""
    if not ctx.is_dm:
        return None

    if _is_greeting_only(ctx.text):
        return ClassificationResult(
            action="respond",
            response_text=NOHELLO_URL,
            reason="dm_greeting",
        )

    return ClassificationResult(
        action="pass",
        response_text=None,
        reason="dm_message",
    )


def thread_continuation_rule(ctx: ClassificationContext) -> Optional[ClassificationResult]:
    """Thread messages where bot was previously engaged → PASS."""
    if ctx.thread_ts and ctx.thread_had_bot:
        return ClassificationResult(
            action="pass",
            response_text=None,
            reason="thread_continuation",
        )
    return None


def substantive_mention_rule(ctx: ClassificationContext) -> Optional[ClassificationResult]:
    """Bot mentioned with substantive content → PASS.

    Recognizes both <@USERID> mentions and natural-language bot name mentions.
    """
    bot_invoked = bot_mentioned_in_text(ctx.text, ctx.bot_user_id) or bot_name_mentioned(ctx.text, ctx.bot_names)
    if not bot_invoked:
        return None
    if _is_substantive(ctx.text):
        return ClassificationResult(
            action="pass",
            response_text=None,
            reason="substantive_mention",
        )
    return None


def greeting_mention_rule(ctx: ClassificationContext) -> Optional[ClassificationResult]:
    """Bot mentioned with greeting-only content → RESPOND with nohello.net.

    Recognizes both <@USERID> mentions and natural-language bot name mentions.
    """
    bot_invoked = bot_mentioned_in_text(ctx.text, ctx.bot_user_id) or bot_name_mentioned(ctx.text, ctx.bot_names)
    if not bot_invoked:
        return None
    if _is_greeting_only(ctx.text):
        return ClassificationResult(
            action="respond",
            response_text=NOHELLO_URL,
            reason="greeting_only",
        )
    return None


def noise_mention_rule(ctx: ClassificationContext) -> Optional[ClassificationResult]:
    """Bot mentioned with acknowledgment/noise content → DROP.

    Recognizes both <@USERID> mentions and natural-language bot name mentions.
    """
    bot_invoked = bot_mentioned_in_text(ctx.text, ctx.bot_user_id) or bot_name_mentioned(ctx.text, ctx.bot_names)
    if not bot_invoked:
        return None
    if _is_acknowledgment(ctx.text):
        return ClassificationResult(
            action="drop",
            response_text=None,
            reason="noise_mention",
        )
    return None


def unrelated_drop_rule(ctx: ClassificationContext) -> ClassificationResult:
    """Catch-all: no rule matched → DROP."""
    return ClassificationResult(
        action="drop",
        response_text=None,
        reason="unrelated",
    )


# ---------------------------------------------------------------------------
# Classifier orchestrator
# ---------------------------------------------------------------------------

CLASSIFICATION_RULES: list = [
    dm_rule,
    thread_continuation_rule,
    substantive_mention_rule,
    greeting_mention_rule,
    noise_mention_rule,
    # unrelated_drop_rule is called explicitly as fallthrough — not in the
    # list because it always returns a result (never None).
]


def classify(ctx: ClassificationContext) -> ClassificationResult:
    """Evaluate classification rules in order; first non-None result wins.

    Falls through to DROP if no rule matches.
    """
    for rule in CLASSIFICATION_RULES:
        result = rule(ctx)
        if result is not None:
            logger.debug(
                "[nixi] classifier: action=%s reason=%s text=%.60s",
                result.action,
                result.reason,
                ctx.text,
            )
            return result

    # Explicit fallthrough — no rule had an opinion
    return unrelated_drop_rule(ctx)