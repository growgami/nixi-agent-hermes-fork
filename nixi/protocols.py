"""Communication protocol constants for nixi-adapter messages.

Protocols are injected as system-level context when forwarding messages to
the LLM, giving the model instructions on how to handle specific message
patterns without requiring the classifier to produce response text directly.
"""

NOHELLO_PROTOCOL: str = (
    "When a user sends a greeting-only message (just 'hi', 'hey', "
    "'hello', or similar with no substantive request), respond with "
    "only: https://nohello.net — no explanation, no follow-up question, "
    "no additional text."
)