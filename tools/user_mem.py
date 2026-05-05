"""User-scoped memory tool registration.

Registers user_mem.set, user_mem.get, and user_mem.delete as agent
tools. Handler lambdas extract _thread_user_memory_manager,
_gateway_session_key, and _user_id from **kwargs and delegate to
ThreadUserMemoryManager.handle_tool_call().
"""

from tools.registry import registry, tool_error


def check_requirements() -> bool:
    """User memory has no external requirements — always available."""
    return True


# ── JSON Schema constants ────────────────────────────────────────────

USER_MEM_SET_SCHEMA = {
    "name": "user_mem.set",
    "description": (
        "Store a key-value pair in user-level persistent memory. "
        "This memory persists across channels and sessions for the "
        "identified user. Use it to remember preferences, facts about "
        "the user, or other long-lived information."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The key to store the value under.",
            },
            "value": {
                "type": "string",
                "description": "The value to store.",
            },
            "ttl_hours": {
                "type": "number",
                "description": (
                    "Optional time-to-live in hours. After this duration, "
                    "the key will be automatically removed. Omit for "
                    "permanent storage."
                ),
            },
        },
        "required": ["key", "value"],
    },
}

USER_MEM_GET_SCHEMA = {
    "name": "user_mem.get",
    "description": (
        "Retrieve a value from user-level persistent memory. "
        "Returns the value stored under the given key for the "
        "identified user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The key to look up.",
            },
        },
        "required": ["key"],
    },
}

USER_MEM_DELETE_SCHEMA = {
    "name": "user_mem.delete",
    "description": (
        "Delete a key from user-level persistent memory. "
        "Removes the key-value pair for the identified user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The key to delete.",
            },
        },
        "required": ["key"],
    },
}


# ── Handler helpers ──────────────────────────────────────────────────

def _user_mem_set(args: dict, **kwargs) -> str:
    mgr = kwargs.get("_thread_user_memory_manager")
    if mgr is None:
        return tool_error("user_mem is only available in gateway sessions")
    function_args = {
        **args,
        "_gateway_session_key": kwargs.get("_gateway_session_key", ""),
        "_user_id": kwargs.get("_user_id", ""),
    }
    return mgr.handle_tool_call("user_mem.set", function_args)


def _user_mem_get(args: dict, **kwargs) -> str:
    mgr = kwargs.get("_thread_user_memory_manager")
    if mgr is None:
        return tool_error("user_mem is only available in gateway sessions")
    function_args = {
        **args,
        "_gateway_session_key": kwargs.get("_gateway_session_key", ""),
        "_user_id": kwargs.get("_user_id", ""),
    }
    return mgr.handle_tool_call("user_mem.get", function_args)


def _user_mem_delete(args: dict, **kwargs) -> str:
    mgr = kwargs.get("_thread_user_memory_manager")
    if mgr is None:
        return tool_error("user_mem is only available in gateway sessions")
    function_args = {
        **args,
        "_gateway_session_key": kwargs.get("_gateway_session_key", ""),
        "_user_id": kwargs.get("_user_id", ""),
    }
    return mgr.handle_tool_call("user_mem.delete", function_args)


# ── Registration ─────────────────────────────────────────────────────

registry.register(
    name="user_mem.set",
    toolset="user_mem",
    schema=USER_MEM_SET_SCHEMA,
    handler=_user_mem_set,
    check_fn=check_requirements,
    emoji="👤",
)

registry.register(
    name="user_mem.get",
    toolset="user_mem",
    schema=USER_MEM_GET_SCHEMA,
    handler=_user_mem_get,
    check_fn=check_requirements,
    emoji="👤",
)

registry.register(
    name="user_mem.delete",
    toolset="user_mem",
    schema=USER_MEM_DELETE_SCHEMA,
    handler=_user_mem_delete,
    check_fn=check_requirements,
    emoji="👤",
)