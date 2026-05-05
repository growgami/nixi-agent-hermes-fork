"""Thread-scoped memory tool registration.

Registers thread_mem.set, thread_mem.get, and thread_mem.delete as agent
tools. Handler lambdas extract _thread_user_memory_manager and
_gateway_session_key from **kwargs and delegate to
ThreadUserMemoryManager.handle_tool_call().
"""

from tools.registry import registry, tool_error


def check_requirements() -> bool:
    """Thread memory has no external requirements — always available."""
    return True


# ── JSON Schema constants ────────────────────────────────────────────

THREAD_MEM_SET_SCHEMA = {
    "name": "thread_mem.set",
    "description": (
        "Store a key-value pair in thread-scoped working memory. "
        "This memory is ephemeral and tied to the current conversation thread. "
        "Use it to remember facts, intermediate results, or context across "
        "turns within this thread. Values are private to this session."
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
        },
        "required": ["key", "value"],
    },
}

THREAD_MEM_GET_SCHEMA = {
    "name": "thread_mem.get",
    "description": (
        "Retrieve a value from thread-scoped working memory. "
        "If a key is provided, returns that specific value. "
        "If no key is provided, returns all key-value pairs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The key to look up. Omit to retrieve all keys.",
            },
        },
        "required": ["key"],
    },
}

THREAD_MEM_DELETE_SCHEMA = {
    "name": "thread_mem.delete",
    "description": (
        "Delete a key from thread-scoped working memory. "
        "Removes the key-value pair from the current thread."
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

def _thread_mem_set(args: dict, **kwargs) -> str:
    mgr = kwargs.get("_thread_user_memory_manager")
    if mgr is None:
        return tool_error("thread_mem is only available in gateway sessions")
    function_args = {
        **args,
        "_gateway_session_key": kwargs.get("_gateway_session_key", ""),
        "_user_id": kwargs.get("_user_id", ""),
    }
    return mgr.handle_tool_call("thread_mem.set", function_args)


def _thread_mem_get(args: dict, **kwargs) -> str:
    mgr = kwargs.get("_thread_user_memory_manager")
    if mgr is None:
        return tool_error("thread_mem is only available in gateway sessions")
    function_args = {
        **args,
        "_gateway_session_key": kwargs.get("_gateway_session_key", ""),
        "_user_id": kwargs.get("_user_id", ""),
    }
    return mgr.handle_tool_call("thread_mem.get", function_args)


def _thread_mem_delete(args: dict, **kwargs) -> str:
    mgr = kwargs.get("_thread_user_memory_manager")
    if mgr is None:
        return tool_error("thread_mem is only available in gateway sessions")
    function_args = {
        **args,
        "_gateway_session_key": kwargs.get("_gateway_session_key", ""),
        "_user_id": kwargs.get("_user_id", ""),
    }
    return mgr.handle_tool_call("thread_mem.delete", function_args)


# ── Registration ─────────────────────────────────────────────────────

registry.register(
    name="thread_mem.set",
    toolset="thread_mem",
    schema=THREAD_MEM_SET_SCHEMA,
    handler=_thread_mem_set,
    check_fn=check_requirements,
    emoji="🧵",
)

registry.register(
    name="thread_mem.get",
    toolset="thread_mem",
    schema=THREAD_MEM_GET_SCHEMA,
    handler=_thread_mem_get,
    check_fn=check_requirements,
    emoji="🧵",
)

registry.register(
    name="thread_mem.delete",
    toolset="thread_mem",
    schema=THREAD_MEM_DELETE_SCHEMA,
    handler=_thread_mem_delete,
    check_fn=check_requirements,
    emoji="🧵",
)