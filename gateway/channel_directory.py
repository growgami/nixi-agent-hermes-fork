"""
Channel directory -- cached map of reachable channels/contacts per platform.

Built on gateway startup, refreshed periodically (every 5 min), and saved to
~/.hermes/channel_directory.json.  The send_message tool reads this file for
action="list" and for resolving human-friendly channel names to numeric IDs.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

DIRECTORY_PATH = get_hermes_home() / "channel_directory.json"


def _normalize_channel_query(value: str) -> str:
    return value.lstrip("#").strip().lower()


def _channel_target_name(platform_name: str, channel: Dict[str, Any]) -> str:
    """Return the human-facing target label shown to users for a channel entry."""
    name = channel["name"]
    if platform_name == "discord" and channel.get("guild"):
        return f"#{name}"
    if platform_name != "discord" and channel.get("type"):
        return f"{name} ({channel['type']})"
    return name


def _session_entry_id(origin: Dict[str, Any]) -> Optional[str]:
    chat_id = origin.get("chat_id")
    if not chat_id:
        return None
    thread_id = origin.get("thread_id")
    if thread_id:
        return f"{chat_id}:{thread_id}"
    return str(chat_id)


def _session_entry_name(origin: Dict[str, Any]) -> str:
    base_name = origin.get("chat_name") or origin.get("user_name") or str(origin.get("chat_id"))
    thread_id = origin.get("thread_id")
    if not thread_id:
        return base_name

    topic_label = origin.get("chat_topic") or f"topic {thread_id}"
    return f"{base_name} / {topic_label}"


# ---------------------------------------------------------------------------
# Build / refresh
# ---------------------------------------------------------------------------

async def build_channel_directory(adapters: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Build a channel directory from connected platform adapters and session data.

    Returns the directory dict and writes it to DIRECTORY_PATH.
    """
    from gateway.config import Platform

    platforms: Dict[str, List[Dict[str, str]]] = {}

    for platform, adapter in adapters.items():
        try:
            if platform == Platform.DISCORD:
                platforms["discord"] = _build_discord(adapter)
            elif platform == Platform.SLACK:
                platforms["slack"] = await _build_slack(adapter)
        except Exception as e:
            logger.warning("Channel directory: failed to build %s: %s", platform.value, e)

    # Platforms that don't support direct channel enumeration get session-based
    # discovery automatically.  Skip infrastructure entries that aren't messaging
    # platforms — everything else falls through to _build_from_sessions().
    # "nixi" is handled below: in NIXI_MODE it mirrors "slack" plus session
    # entries; without NIXI_MODE it should not appear at all.
    _SKIP_SESSION_DISCOVERY = frozenset({"local", "api_server", "webhook", "nixi"})
    for plat in Platform:
        plat_name = plat.value
        if plat_name in _SKIP_SESSION_DISCOVERY or plat_name in platforms:
            continue
        platforms[plat_name] = _build_from_sessions(plat_name)

    # In NIXI_MODE, Platform.NIXI is in adapters and Platform.SLACK may also be
    # present.  Add a "nixi" section mirroring "slack" so that both nixi: and
    # slack: prefixes resolve in send_message.
    if Platform.NIXI in adapters and "slack" in platforms:
        nixi_channels = []
        slack_channels = platforms["slack"]
        for ch in slack_channels:
            nixi_ch = dict(ch)
            # Session-based nixi entries may have names like "nixi/C0AE0QVNT1P" —
            # strip the "nixi/" prefix so the name is a resolvable channel ID/name.
            if nixi_ch.get("name", "").startswith("nixi/"):
                nixi_ch["name"] = nixi_ch["name"].removeprefix("nixi/")
            nixi_channels.append(nixi_ch)
        # Also merge any session-based nixi entries that aren't in the slack list.
        nixi_session = _build_from_sessions("nixi")
        existing_ids = {ch["id"] for ch in nixi_channels}
        for ch in nixi_session:
            stripped_name = ch.get("name", "")
            if stripped_name.startswith("nixi/"):
                ch["name"] = stripped_name.removeprefix("nixi/")
            if ch["id"] not in existing_ids:
                nixi_channels.append(ch)
                existing_ids.add(ch["id"])
        platforms["nixi"] = nixi_channels

    directory = {
        "updated_at": datetime.now().isoformat(),
        "platforms": platforms,
    }

    try:
        atomic_json_write(DIRECTORY_PATH, directory)
    except Exception as e:
        logger.warning("Channel directory: failed to write: %s", e)

    return directory


def build_channel_directory_sync(adapters: Dict[Any, Any], loop=None) -> Dict[str, Any]:
    """Synchronous wrapper for build_channel_directory.

    Used from threads that can't await (e.g. the cron ticker thread).
    If *loop* is a running asyncio event loop, schedules the coroutine on it.
    Otherwise, creates a fresh event loop via asyncio.run().
    """
    try:
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                build_channel_directory(adapters), loop,
            )
            return future.result(timeout=30)
        return asyncio.run(build_channel_directory(adapters))
    except Exception as e:
        logger.warning("Channel directory: sync build failed: %s", e)
        return {"updated_at": None, "platforms": {}}


def _build_discord(adapter) -> List[Dict[str, str]]:
    """Enumerate all text channels and forum channels the Discord bot can see."""
    channels = []
    client = getattr(adapter, "_client", None)
    if not client:
        return channels

    try:
        import discord as _discord  # noqa: F401 — SDK presence check
    except ImportError:
        return channels

    for guild in client.guilds:
        for ch in guild.text_channels:
            channels.append({
                "id": str(ch.id),
                "name": ch.name,
                "guild": guild.name,
                "type": "channel",
            })
        # Forum channels (type 15) — creating a message auto-spawns a thread post.
        forums = getattr(guild, "forum_channels", None) or []
        for ch in forums:
            channels.append({
                "id": str(ch.id),
                "name": ch.name,
                "guild": guild.name,
                "type": "forum",
            })
        # Also include DM-capable users we've interacted with is not
        # feasible via guild enumeration; those come from sessions.

    # Merge any DMs from session history
    channels.extend(_build_from_sessions("discord"))
    return channels


async def _build_slack(adapter) -> List[Dict[str, str]]:
    """List Slack channels the bot has joined via conversations_list API.

    In NIXI_MODE, uses ``_primary_client`` (AsyncWebClient).  In Socket Mode,
    uses ``_app.client`` (accessed through the AsyncApp).  Falls back to
    session data when no client is available or on ``missing_scope`` errors.
    """
    # Determine which Slack API client to use.
    # NIXI_MODE: _primary_client is an AsyncWebClient set in _connect_nixi_mode().
    # Socket Mode: _app is an AsyncApp — use _app.client for API calls.
    client = getattr(adapter, "_primary_client", None)
    if client is None:
        app = getattr(adapter, "_app", None)
        if app is not None:
            client = getattr(app, "client", None)

    if client is None:
        return _build_from_sessions("slack")

    channels: List[Dict[str, str]] = []
    try:
        from slack_sdk.errors import SlackApiError

        cursor = None
        max_channels = 1000
        while len(channels) < max_channels:
            resp = await client.conversations_list(
                types="public_channel,private_channel",
                limit=200,
                cursor=cursor or "",
            )
            for ch in resp.get("channels", []):
                channels.append({
                    "id": ch["id"],
                    "name": ch["name"],
                    "is_private": ch.get("is_private", False),
                    "type": "private_channel" if ch.get("is_private") else "channel",
                })
            # Pagination: continue if there's a next_cursor
            metadata = resp.get("response_metadata", {})
            next_cursor = metadata.get("next_cursor", "")
            if not next_cursor:
                break
            cursor = next_cursor

    except SlackApiError as e:
        error_str = str(e)
        if "missing_scope" in error_str or (hasattr(e, "response") and isinstance(e.response, dict) and e.response.get("error") == "missing_scope"):
            logger.warning(
                "Channel directory: Slack API missing_scope error (needs channels:read). "
                "Falling back to session data.",
            )
        else:
            logger.warning("Channel directory: Slack API error: %s. Falling back to session data.", e)
        return _build_from_sessions("slack")
    except Exception as e:
        logger.warning("Channel directory: failed to enumerate Slack channels: %s. Falling back to session data.", e)
        return _build_from_sessions("slack")

    # Merge session data — add any channels from sessions that aren't already
    # in the API results (deduplicate by ID).
    session_channels = _build_from_sessions("slack")
    existing_ids = {ch["id"] for ch in channels}
    for ch in session_channels:
        if ch["id"] not in existing_ids:
            channels.append(ch)
            existing_ids.add(ch["id"])

    return channels


def _build_from_sessions(platform_name: str) -> List[Dict[str, str]]:
    """Pull known channels/contacts from sessions.json origin data."""
    sessions_path = get_hermes_home() / "sessions" / "sessions.json"
    if not sessions_path.exists():
        return []

    entries = []
    try:
        with open(sessions_path, encoding="utf-8") as f:
            data = json.load(f)

        seen_ids = set()
        for _key, session in data.items():
            origin = session.get("origin") or {}
            if origin.get("platform") != platform_name:
                continue
            entry_id = _session_entry_id(origin)
            if not entry_id or entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            entries.append({
                "id": entry_id,
                "name": _session_entry_name(origin),
                "type": session.get("chat_type", "dm"),
                "thread_id": origin.get("thread_id"),
            })
    except Exception as e:
        logger.debug("Channel directory: failed to read sessions for %s: %s", platform_name, e)

    return entries


# ---------------------------------------------------------------------------
# Read / resolve
# ---------------------------------------------------------------------------

def load_directory() -> Dict[str, Any]:
    """Load the cached channel directory from disk."""
    if not DIRECTORY_PATH.exists():
        return {"updated_at": None, "platforms": {}}
    try:
        with open(DIRECTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"updated_at": None, "platforms": {}}


def lookup_channel_type(platform_name: str, chat_id: str) -> Optional[str]:
    """Return the channel ``type`` string (e.g. ``"channel"``, ``"forum"``) for *chat_id*, or *None* if unknown."""
    directory = load_directory()
    for ch in directory.get("platforms", {}).get(platform_name, []):
        if ch.get("id") == chat_id:
            return ch.get("type")
    return None


# nixi and slack share the same channel directory data — each can
# fall back to the other when the primary section has no match.
_PLATFORM_ALIAS = {
    "nixi": "slack",
    "slack": "nixi",
}


def _resolve_in_channels(
    platform_name: str,
    channels: List[Dict[str, Any]],
    query: str,
) -> Optional[str]:
    """Try to resolve *query* against *channels* for *platform_name*.

    This is the inner matching loop extracted from ``resolve_channel_name``
    so it can be reused for cross-platform aliasing.
    """
    # 1. Exact name match, including the display labels shown by send_message(action="list")
    for ch in channels:
        if _normalize_channel_query(ch["name"]) == query:
            return ch["id"]
        if _normalize_channel_query(_channel_target_name(platform_name, ch)) == query:
            return ch["id"]

    # 2. Guild-qualified match for Discord ("GuildName/channel")
    if "/" in query:
        guild_part, ch_part = query.rsplit("/", 1)
        for ch in channels:
            guild = ch.get("guild", "").strip().lower()
            if guild == guild_part and _normalize_channel_query(ch["name"]) == ch_part:
                return ch["id"]

    # 3. Partial prefix match (only if unambiguous)
    matches = [ch for ch in channels if _normalize_channel_query(ch["name"]).startswith(query)]
    if len(matches) == 1:
        return matches[0]["id"]

    return None


def resolve_channel_name(platform_name: str, name: str) -> Optional[str]:
    """
    Resolve a human-friendly channel name to a numeric ID.

    Matching strategy (case-insensitive, first match wins):
    - Discord: "bot-home", "#bot-home", "GuildName/bot-home"
    - Telegram: display name or group name
    - Slack: "engineering", "#engineering"

    Cross-platform aliasing: "nixi" and "slack" fall back to each other
    when no match is found in the primary section.
    """
    directory = load_directory()
    all_platforms = directory.get("platforms", {})
    channels = all_platforms.get(platform_name, [])

    query = _normalize_channel_query(name)

    # Try the requested platform section first.
    if channels:
        result = _resolve_in_channels(platform_name, channels, query)
        if result is not None:
            return result

    # Cross-platform fallback: nixi ↔ slack
    alias = _PLATFORM_ALIAS.get(platform_name)
    if alias:
        alias_channels = all_platforms.get(alias, [])
        if alias_channels:
            result = _resolve_in_channels(alias, alias_channels, query)
            if result is not None:
                return result

    return None


def format_directory_for_display() -> str:
    """Format the channel directory as a human-readable list for the model."""
    directory = load_directory()
    platforms = directory.get("platforms", {})

    if not any(platforms.values()):
        return "No messaging platforms connected or no channels discovered yet."

    lines = ["Available messaging targets:\n"]

    for plat_name, channels in sorted(platforms.items()):
        if not channels:
            continue

        # Group Discord channels by guild
        if plat_name == "discord":
            guilds: Dict[str, List] = {}
            dms: List = []
            for ch in channels:
                guild = ch.get("guild")
                if guild:
                    guilds.setdefault(guild, []).append(ch)
                else:
                    dms.append(ch)

            for guild_name, guild_channels in sorted(guilds.items()):
                lines.append(f"Discord ({guild_name}):")
                for ch in sorted(guild_channels, key=lambda c: c["name"]):
                    lines.append(f"  discord:{_channel_target_name(plat_name, ch)}")
            if dms:
                lines.append("Discord (DMs):")
                for ch in dms:
                    lines.append(f"  discord:{_channel_target_name(plat_name, ch)}")
            lines.append("")
        else:
            lines.append(f"{plat_name.title()}:")
            for ch in channels:
                lines.append(f"  {plat_name}:{_channel_target_name(plat_name, ch)}")
            lines.append("")

    lines.append('Use these as the "target" parameter when sending.')
    lines.append('Bare platform name (e.g. "telegram") sends to home channel.')

    return "\n".join(lines)
