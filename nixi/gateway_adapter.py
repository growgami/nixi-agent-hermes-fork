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

from nixi.employee_provider import load_overlay

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
        """Extract Slack event fields, load overlay, construct MessageEvent, and dispatch."""
        # Extract Slack event fields from the payload
        # Sludge sends the full Slack event envelope; the relevant fields
        # are in event_data itself or event_data.get("event", {})
        event = event_data.get("event", event_data)
        text = event.get("text", "")
        channel = event.get("channel", "")
        thread_ts = event.get("thread_ts")

        # Load employee overlay for ephemeral context injection (cached)
        overlay = self._get_overlay_with_cache(user_id)

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
            channel_prompt=overlay if overlay else None,
        )

        logger.info(
            "[nixi] Dispatching event: user=%s channel=%s text_len=%d overlay=%d",
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