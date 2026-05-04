"""Nixi platform adapter — receives events from Sludge via HTTP POST.

Sludge (the Slack gateway service) POSTs Slack event payloads to this
adapter on port 8080. The adapter validates auth and tenant headers,
loads per-employee context overlays, and dispatches messages through
the gateway. Outbound delivery is delegated to the Slack adapter via
gateway_runner.cross-platform delivery.

Security:
- Bearer token auth via hmac.compare_digest (constant-time comparison)
- Team ID validation rejects cross-tenant events (403)
- Body size limit checked before payload parsing
- Employee overlay injected via channel_prompt (ephemeral, never persisted)
"""

import asyncio
import hmac
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.platforms.helpers import MessageDeduplicator

from nixi.employee_provider import load_overlay
from nixi.intent_classifier import (
    ClassificationContext,
    ClassificationResult,
    ThreadMentionCache,
    bot_mentioned_in_text,
    bot_name_mentioned,
    classify,
)
from nixi.protocols import NOHELLO_PROTOCOL

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
MAX_BODY_BYTES = 1_048_576  # 1 MB


def check_nixi_requirements() -> bool:
    """Check if Nixi adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


class NixiAdapter(BasePlatformAdapter):
    """Receives HTTP POSTs from Sludge and dispatches messages through the gateway.

    Cross-platform delivery (send/send_image/send_document) delegates to the
    Slack adapter via self.gateway_runner, following the same pattern as
    WebhookAdapter.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NIXI)
        extra = config.extra or {}
        self._internal_secret: str = os.getenv(
            "NIXI_INTERNAL_SECRET", extra.get("internal_secret", "")
        )
        self._team_id: str = os.getenv(
            "NIXI_TEAM_ID", extra.get("team_id", "")
        )
        self._host: str = extra.get("host", DEFAULT_HOST)
        self._port: int = int(extra.get("port", DEFAULT_PORT))

        # Concurrency limit: bounds concurrent handle_message() calls to
        # prevent exhausting LLM API rate limits under concurrent load.
        # Config extra takes precedence; env var is fallback; default is 10.
        _concurrency_raw = extra.get("NIXI_CONCURRENCY_LIMIT")
        if _concurrency_raw is None:
            _concurrency_raw = os.getenv("NIXI_CONCURRENCY_LIMIT", "10")
        try:
            self._concurrency_limit: int = int(_concurrency_raw)
        except (ValueError, TypeError):
            self._concurrency_limit = 10
        if self._concurrency_limit < 1:
            self._concurrency_limit = 1
        self._concurrency_semaphore: asyncio.Semaphore = asyncio.Semaphore(
            self._concurrency_limit
        )

        # Overlay cache: avoids repeated disk reads for USER.md overlays.
        # OrderedDict preserves insertion order for LRU eviction.
        # Maps user_id → (overlay_text, timestamp).
        self._overlay_cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        _ttl_raw = extra.get("NIXI_OVERLAY_CACHE_TTL")
        if _ttl_raw is None:
            _ttl_raw = os.getenv("NIXI_OVERLAY_CACHE_TTL", "300")
        try:
            self._overlay_cache_ttl: int = int(_ttl_raw)
        except (ValueError, TypeError):
            self._overlay_cache_ttl = 300
        _max_raw = extra.get("NIXI_OVERLAY_CACHE_MAX")
        if _max_raw is None:
            _max_raw = os.getenv("NIXI_OVERLAY_CACHE_MAX", "256")
        try:
            self._overlay_cache_max: int = int(_max_raw)
        except (ValueError, TypeError):
            self._overlay_cache_max = 256

        # Deduplication: prevents duplicate processing of events with the same
        # event_ts (e.g. Slack Socket Mode reconnect redeliveries or network
        # retries). Per-adapter-instance, consistent with single-tenant-per-
        # Machine architecture.
        self._dedup = MessageDeduplicator()

        # Intent classifier: pre-LLM classifier that gates messages
        # before they reach the main agent. ThreadMentionCache tracks which
        # threads the bot has engaged in for thread-continuation detection.
        self._mention_cache: ThreadMentionCache = ThreadMentionCache(
            ttl=1800, max_size=1024
        )

        # Bot user ID for detecting direct mentions (<@UBOT123>). Used by
        # the classifier to determine bot_mentioned. Log a warning when
        # unset — the classifier cannot detect channel mentions without it.
        self._bot_user_id: str = os.getenv(
            "NIXI_BOT_USER_ID", extra.get("bot_user_id", "")
        )
        if not self._bot_user_id:
            logger.warning(
                "[nixi] NIXI_BOT_USER_ID not set — classifier will not detect channel mentions"
            )

        # Bot names for natural-language mention detection (e.g. "nixi", "fixi").
        # Env var override takes precedence (JSON list string), then config extra,
        # then default. "nixi" is always included as a baseline.
        _bot_names_env = os.getenv("NIXI_BOT_NAMES", "")
        if _bot_names_env:
            try:
                _parsed = json.loads(_bot_names_env)
                if isinstance(_parsed, list):
                    _bot_names_set = set(_parsed)
                else:
                    logger.warning(
                        "[nixi] NIXI_BOT_NAMES env var is not a JSON list, falling back to config"
                    )
                    _bot_names_set = set(extra.get("bot_names", ["nixi"]))
            except json.JSONDecodeError:
                logger.warning(
                    "[nixi] NIXI_BOT_NAMES env var is not valid JSON, falling back to config"
                )
                _bot_names_set = set(extra.get("bot_names", ["nixi"]))
        else:
            _bot_names_set = set(extra.get("bot_names", ["nixi"]))

        _bot_names_set.add("nixi")  # Always include baseline name
        self._bot_names: tuple[str, ...] = tuple(sorted(_bot_names_set))
        logger.info("[nixi] Bot names for classifier: %s", self._bot_names)

        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        # Set externally by GatewayRunner (same pattern as WebhookAdapter)
        self.gateway_runner = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the aiohttp HTTP server and begin listening for events."""
        if not AIOHTTP_AVAILABLE:
            logger.error("[nixi] aiohttp not available — cannot start adapter")
            return False

        if not self._internal_secret:
            logger.error(
                "[nixi] NIXI_INTERNAL_SECRET not set — cannot start adapter. "
                "Set the NIXI_INTERNAL_SECRET environment variable."
            )
            return False

        app = web.Application()
        app.router.add_post("/nixi/event", self._handle_nixi_event)
        app.router.add_get("/health", self._handle_health)
        self._app = app

        # Port conflict detection
        import socket as _socket

        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                _s.settimeout(1)
                _s.connect(("127.0.0.1", self._port))
            logger.error(
                "[nixi] Port %d already in use. "
                "Set a different port in config.yaml: platforms.nixi.extra.port",
                self._port,
            )
            return False
        except (ConnectionRefusedError, OSError):
            pass  # port is free

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        self._mark_connected()
        logger.info(
            "[nixi] Listening on %s:%d (team_id=%s, concurrency_limit=%d)",
            self._host,
            self._port,
            self._team_id or "(not set)",
            self._concurrency_limit,
        )
        return True

    async def disconnect(self) -> None:
        """Shut down the aiohttp HTTP server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
            self._app = None
        self._mark_disconnected()
        logger.info("[nixi] Disconnected")

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — health check endpoint."""
        return web.json_response(
            {"status": "ok", "team_id": self._team_id}
        )

    async def _handle_nixi_event(
        self, request: "web.Request"
    ) -> "web.Response":
        """POST /nixi/event — receive and process an event from Sludge.

        Validates auth and tenant headers, extracts user info, loads employee
        overlay, and dispatches the message through the gateway.
        """
        # ── Auth ──────────────────────────────────────────────────────
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return web.json_response(
                {"error": "Missing or invalid Authorization header"}, status=401
            )
        token = auth_header[7:].strip()
        if not hmac.compare_digest(token, self._internal_secret):
            return web.json_response(
                {"error": "Invalid authorization token"}, status=401
            )

        # ── Team ID validation ───────────────────────────────────────
        team_id = request.headers.get("X-Nixi-Team-Id", "")
        if self._team_id and not hmac.compare_digest(team_id, self._team_id):
            logger.warning(
                "[nixi] Rejected event with wrong team_id=%s (expected %s)",
                team_id,
                self._team_id,
            )
            return web.json_response(
                {"error": "Team ID mismatch"}, status=403
            )

        # ── Body size check ──────────────────────────────────────────
        content_length = request.content_length or 0
        if content_length > MAX_BODY_BYTES:
            return web.json_response(
                {"error": "Payload too large"}, status=413
            )

        # ── Parse body ───────────────────────────────────────────────
        try:
            raw_body = await request.read()
        except Exception as e:
            logger.error("[nixi] Failed to read request body: %s", e)
            return web.json_response(
                {"error": "Bad request"}, status=400
            )

        try:
            event_data = json.loads(raw_body)
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON payload"}, status=400
            )

        # ── Extract user info ────────────────────────────────────────
        user_id = request.headers.get("X-Nixi-User-Id", "unknown")
        user_name = request.headers.get("X-Nixi-User-Name", "unknown")

        # ── Dispatch ─────────────────────────────────────────────────
        await self._dispatch_event(event_data, user_id, user_name)

        return web.json_response({"status": "ok"}, status=200)

    async def _run_with_semaphore(self, message_event: MessageEvent) -> None:
        """Acquire the concurrency semaphore, call handle_message, and release on completion.

        This bounds the number of simultaneous LLM API calls to
        NIXI_CONCURRENCY_LIMIT (default 10), preventing rate limit exhaustion
        under concurrent load. The semaphore is released in the finally block
        so exceptions don't leak the permit.
        """
        await self._concurrency_semaphore.acquire()
        try:
            await self.handle_message(message_event)
        finally:
            self._concurrency_semaphore.release()

    def _get_overlay_with_cache(self, user_id: str) -> str:
        """Return cached overlay for user_id, loading from disk on miss or expiry.

        Cache semantics:
        - Hit (entry exists, not expired): return cached value. O(1) lookup, no eviction.
        - Miss (no entry): call load_overlay(), store result with current timestamp.
        - Expired (entry exists but TTL elapsed): delete entry, fall through to miss path.
        - Eviction on insert only: after inserting a new entry, if cache exceeds
          _overlay_cache_max, first evict all expired entries, then evict oldest
          (popitem(last=False) — insertion-order LRU) until under limit.
        """
        now = time.time()

        # ── Cache hit path (O(1), no eviction) ───────────────────────────
        if user_id in self._overlay_cache:
            overlay_text, ts = self._overlay_cache[user_id]
            if now - ts < self._overlay_cache_ttl:
                return overlay_text
            # Expired — remove and fall through to reload
            del self._overlay_cache[user_id]

        # ── Cache miss path: load from disk ──────────────────────────────
        overlay_text = load_overlay(user_id)
        self._overlay_cache[user_id] = (overlay_text, now)

        # ── Eviction on insert ────────────────────────────────────────────
        if len(self._overlay_cache) > self._overlay_cache_max:
            # First pass: evict all expired entries
            expired_keys = [
                k for k, (_, ts) in self._overlay_cache.items()
                if now - ts >= self._overlay_cache_ttl
            ]
            for k in expired_keys:
                del self._overlay_cache[k]

            # Second pass: if still over limit, evict oldest by insertion order
            while len(self._overlay_cache) > self._overlay_cache_max:
                self._overlay_cache.popitem(last=False)

        return overlay_text

    async def _dispatch_event(
        self,
        event_data: Dict[str, Any],
        user_id: str,
        user_name: str,
    ) -> None:
        """Extract Slack event fields, classify intent, and dispatch accordingly.

        Classification determines whether to DROP (discard) or PASS (continue
        to overlay load + LLM with NOHELLO_PROTOCOL injected).
        """
        # Extract Slack event fields from the payload
        # Sludge sends the full Slack event envelope; the relevant fields
        # are in event_data itself or event_data.get("event", {})
        event = event_data.get("event", event_data)

        # Subtype filter: only process regular user messages (subtype=None).
        # Subtypes like "message_changed", "bot_add", "file_share" etc. are
        # not user-originated messages and should not reach the classifier.
        subtype = event.get("subtype")
        if subtype:
            logger.debug(
                "[nixi] Skipping event with subtype=%s event_ts=%s",
                subtype,
                event.get("event_ts", "?"),
            )
            return

        # Slack event field mapping: message events have both 'event_ts' and
        # 'ts' (typically equal). Use event_ts first (more specific), fall back
        # to ts. Since app_mention is filtered in Sludge, only message events
        # reach here.
        event_ts = event.get("event_ts") or event.get("ts", "")

        # Synchronous dedup check BEFORE asyncio.create_task to prevent the
        # race condition where two events pass the check before either is
        # recorded.
        if event_ts and self._dedup.is_duplicate(event_ts):
            logger.info("[nixi] Skipping duplicate event: event_ts=%s", event_ts)
            return

        text = event.get("text", "")
        channel = event.get("channel", "")
        thread_ts = event.get("thread_ts")

        # ── Intent classification ──────────────────────────────────────
        # Determine DM status: channel_type=="im" is primary, "D" prefix
        # on channel ID is Slack's DM convention. Empty channel defaults to
        # NOT DM — never assume DM from absence of data.
        is_dm = event.get("channel_type") == "im" or (
            bool(channel) and channel.startswith("D")
        )

        # Determine bot mention status
        bot_mentioned = bot_mentioned_in_text(text, self._bot_user_id)
        bot_name_detected = bot_name_mentioned(text, self._bot_names)
        bot_invoked = bot_mentioned or bot_name_detected

        # Determine thread-continuation status from cache
        thread_had_bot = (
            self._mention_cache.had_bot(thread_ts) if thread_ts else False
        )

        ctx = ClassificationContext(
            text=text,
            channel=channel,
            is_dm=is_dm,
            thread_ts=thread_ts,
            bot_user_id=self._bot_user_id or None,
            thread_had_bot=thread_had_bot,
            bot_names=self._bot_names,
        )

        logger.debug(
            "[nixi] Message receipt: user=%s channel=%s text_len=%d bot_mentioned=%s bot_name_detected=%s",
            user_id,
            channel,
            len(text),
            bot_mentioned,
            bot_name_detected,
        )
        result = classify(ctx)

        logger.info(
            "[nixi] Classified event: user=%s channel=%s action=%s reason=%s text_len=%d",
            user_id,
            channel,
            result.action,
            result.reason,
            len(text),
        )

        # Record thread presence when the bot will engage: (a) directly
        # mentioned, or (b) classified as PASS for a thread message (nixi
        # will respond in this thread, making bot "present" for future
        # continuation detection).
        if thread_ts and (bot_invoked or result.action == "pass"):
            self._mention_cache.record(thread_ts)

        if result.action == "drop":
            logger.info(
                "[nixi] Dropping event: reason=%s user=%s channel=%s",
                result.reason,
                user_id,
                channel,
            )
            return

        # PASS: continue to overlay load + LLM
        # Load employee overlay for ephemeral context injection (cached)
        overlay = self._get_overlay_with_cache(user_id)

        # Build channel_prompt: always include NOHELLO_PROTOCOL,
        # prepend employee overlay if present.
        if overlay:
            channel_prompt = f"{overlay}\n\n{NOHELLO_PROTOCOL}"
        else:
            channel_prompt = NOHELLO_PROTOCOL

        # Build session key and source
        chat_id = channel or f"nixi:{user_id}"
        source = self.build_source(
            chat_id=chat_id,
            chat_name=f"nixi/{channel}" if channel else f"nixi/dm/{user_id}",
            chat_type="group" if channel else "dm",
            user_id=user_id,
            user_name=user_name,
            thread_id=thread_ts,
        )

        message_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event_data,
            message_id=event_ts or None,
            channel_prompt=channel_prompt,
        )

        logger.info(
            "[nixi] Dispatching event (action=pass): user=%s channel=%s text_len=%d overlay=%d",
            user_id,
            channel,
            len(text),
            len(overlay) if overlay else 0,
        )

        # Non-blocking dispatch via handle_message, bounded by concurrency semaphore
        task = asyncio.create_task(self._run_with_semaphore(message_event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # ------------------------------------------------------------------
    # Outbound delivery (delegates to Slack adapter)
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Delegate outbound message delivery to the Slack adapter."""
        if not self.gateway_runner:
            return SendResult(
                success=False, error="Slack adapter not available for delivery"
            )
        slack_adapter = self.gateway_runner.adapters.get(Platform.SLACK)
        if not slack_adapter:
            return SendResult(
                success=False, error="Slack adapter not available for delivery"
            )
        return await slack_adapter.send(
            chat_id, content, reply_to=reply_to, metadata=metadata
        )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Delegate outbound image delivery to the Slack adapter."""
        if not self.gateway_runner:
            return SendResult(
                success=False, error="Slack adapter not available for delivery"
            )
        slack_adapter = self.gateway_runner.adapters.get(Platform.SLACK)
        if not slack_adapter:
            return SendResult(
                success=False, error="Slack adapter not available for delivery"
            )
        return await slack_adapter.send_image(
            chat_id, image_url, caption=caption, reply_to=reply_to, metadata=metadata
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Delegate outbound document delivery to the Slack adapter."""
        if not self.gateway_runner:
            return SendResult(
                success=False, error="Slack adapter not available for delivery"
            )
        slack_adapter = self.gateway_runner.adapters.get(Platform.SLACK)
        if not slack_adapter:
            return SendResult(
                success=False, error="Slack adapter not available for delivery"
            )
        return await slack_adapter.send_document(
            chat_id,
            file_path,
            caption=caption,
            file_name=file_name,
            reply_to=reply_to,
            **kwargs,
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """No-op — Nixi has no typing indicator API."""
        logger.debug("[nixi] send_typing called for %s (no-op)", chat_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """No-op — Nixi doesn't have a chat info API."""
        logger.debug("[nixi] get_chat_info called for %s (no-op)", chat_id)
        return {"name": chat_id, "type": "nixi"}