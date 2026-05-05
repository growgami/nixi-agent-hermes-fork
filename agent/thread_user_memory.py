"""ThreadUserMemoryManager — wraps SessionDB CRUD for thread/user memory tools.

Provides:
- prefetch(): queries both tables and formats <thread-context>/<user-context>
  blocks for injection into agent prompts.
- thread_set/get/delete: JSON-returning CRUD over thread_memory.
- user_set/get/delete: JSON-returning CRUD over user_memory.
- handle_tool_call(): dispatches thread_mem.* / user_mem.* tool calls.

All methods catch exceptions and return JSON error strings, matching the
tools/registry.py tool_error() pattern.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ThreadUserMemoryManager:
    """Manages thread-scoped and user-scoped memory for gateway sessions.

    Wraps SessionDB CRUD methods, formats context blocks for injection into
    agent prompts, and dispatches thread_mem.* / user_mem.* tool calls.
    """

    def __init__(self, db: "SessionDB"):
        self._db = db

    # ── Context formatting ────────────────────────────────────────────

    def prefetch(self, session_key: str, user_id: str) -> str:
        """Query both tables and format context blocks for agent injection.

        Returns a formatted string with <thread-context> and/or <user-context>
        blocks. Returns empty string if both memories are empty.
        """
        parts: list[str] = []

        try:
            thread_entries = self._db.get_thread_memory(session_key)
            if thread_entries:
                lines = [
                    "<thread-context>",
                    "[System note: The following is thread-scoped working memory, NOT new user input. This context is ephemeral and tied to this conversation thread.]",
                ]
                for key, value in thread_entries:
                    lines.append(f"- {key}: {value}")
                lines.append("</thread-context>")
                parts.append("\n".join(lines))
        except Exception:
            logger.exception("prefetch: failed to read thread_memory for %s", session_key)

        try:
            user_entries = self._db.get_user_memory(user_id)
            if user_entries:
                lines = [
                    "<user-context>",
                    "[System note: The following is user-level memory that persists across channels and sessions. Treat as background knowledge about this user.]",
                ]
                for key, value in user_entries:
                    lines.append(f"- {key}: {value}")
                lines.append("</user-context>")
                parts.append("\n".join(lines))
        except Exception:
            logger.exception("prefetch: failed to read user_memory for %s", user_id)

        return "\n\n".join(parts)

    # ── Thread memory CRUD ────────────────────────────────────────────

    def thread_set(self, session_key: str, key: str, value: str) -> str:
        """Set a thread-memory key. Returns JSON {"success": true, "key": key}."""
        try:
            self._db.set_thread_memory(session_key, key, value)
            return json.dumps({"success": True, "key": key}, ensure_ascii=False)
        except Exception as e:
            logger.exception("thread_set failed")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def thread_get(self, session_key: str, key: Optional[str] = None) -> str:
        """Get thread-memory value(s).

        If key is provided: {"key": key, "value": value_or_null}.
        If key is None: {"keys": {"key1": "value1", ...}}.
        """
        try:
            entries = self._db.get_thread_memory(session_key)
            entries_dict = dict(entries)
            if key is not None:
                return json.dumps(
                    {"key": key, "value": entries_dict.get(key)},
                    ensure_ascii=False,
                )
            return json.dumps({"keys": entries_dict}, ensure_ascii=False)
        except Exception as e:
            logger.exception("thread_get failed")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def thread_delete(self, session_key: str, key: str) -> str:
        """Delete a thread-memory key. Returns JSON {"success": true, "key": key}."""
        try:
            self._db.delete_thread_memory_key(session_key, key)
            return json.dumps({"success": True, "key": key}, ensure_ascii=False)
        except Exception as e:
            logger.exception("thread_delete failed")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    # ── User memory CRUD ──────────────────────────────────────────────

    def user_set(self, user_id: str, key: str, value: str, ttl_hours: float = None) -> str:
        """Set a user-memory key. Returns JSON {"success": true, "key": key}."""
        try:
            self._db.set_user_memory(user_id, key, value, ttl_hours=ttl_hours)
            return json.dumps({"success": True, "key": key}, ensure_ascii=False)
        except Exception as e:
            logger.exception("user_set failed")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def user_get(self, user_id: str, key: str) -> str:
        """Get a user-memory value. Returns JSON {"key": key, "value": value}."""
        try:
            entries = self._db.get_user_memory(user_id)
            entries_dict = dict(entries)
            return json.dumps(
                {"key": key, "value": entries_dict.get(key)},
                ensure_ascii=False,
            )
        except Exception as e:
            logger.exception("user_get failed")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def user_delete(self, user_id: str, key: str) -> str:
        """Delete a user-memory key. Returns JSON {"success": true, "key": key}."""
        try:
            self._db.delete_user_memory(user_id, key)
            return json.dumps({"success": True, "key": key}, ensure_ascii=False)
        except Exception as e:
            logger.exception("user_delete failed")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    # ── Bulk delete ──────────────────────────────────────────────────

    def delete_thread_memory(self, session_key: str) -> None:
        """Delete all thread memory for a session. Delegates to SessionDB."""
        try:
            self._db.delete_thread_memory(session_key)
        except Exception:
            logger.exception("delete_thread_memory failed for %s", session_key)

    # ── Tool call dispatch ────────────────────────────────────────────

    def handle_tool_call(self, function_name: str, function_args: dict) -> str:
        """Dispatch thread_mem.* and user_mem.* tool calls.

        Extracts session_key from function_args["_gateway_session_key"]
        and user_id from function_args["_user_id"].

        Returns JSON error strings for auth failures or unknown functions.
        """
        session_key = function_args.get("_gateway_session_key", "")
        user_id = function_args.get("_user_id", "")

        if function_name == "thread_mem.set":
            if not session_key:
                return json.dumps(
                    {"error": "thread_mem is only available in gateway sessions"},
                    ensure_ascii=False,
                )
            return self.thread_set(
                session_key,
                function_args.get("key", ""),
                function_args.get("value", ""),
            )

        elif function_name == "thread_mem.get":
            if not session_key:
                return json.dumps(
                    {"error": "thread_mem is only available in gateway sessions"},
                    ensure_ascii=False,
                )
            return self.thread_get(
                session_key,
                function_args.get("key"),
            )

        elif function_name == "thread_mem.delete":
            if not session_key:
                return json.dumps(
                    {"error": "thread_mem is only available in gateway sessions"},
                    ensure_ascii=False,
                )
            return self.thread_delete(
                session_key,
                function_args.get("key", ""),
            )

        elif function_name == "user_mem.set":
            if not user_id:
                return json.dumps(
                    {"error": "user_mem is only available for identified users"},
                    ensure_ascii=False,
                )
            return self.user_set(
                user_id,
                function_args.get("key", ""),
                function_args.get("value", ""),
                ttl_hours=function_args.get("ttl_hours"),
            )

        elif function_name == "user_mem.get":
            if not user_id:
                return json.dumps(
                    {"error": "user_mem is only available for identified users"},
                    ensure_ascii=False,
                )
            return self.user_get(
                user_id,
                function_args.get("key", ""),
            )

        elif function_name == "user_mem.delete":
            if not user_id:
                return json.dumps(
                    {"error": "user_mem is only available for identified users"},
                    ensure_ascii=False,
                )
            return self.user_delete(
                user_id,
                function_args.get("key", ""),
            )

        else:
            return json.dumps(
                {"error": f"Unknown function: {function_name}"},
                ensure_ascii=False,
            )