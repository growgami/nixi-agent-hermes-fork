"""Communication protocol constants for nixi-adapter messages.

Protocols are injected as system-level context when forwarding messages to
the LLM, giving the model instructions on how to handle specific message
patterns without requiring the classifier to produce response text directly.
"""

NOHELLO_PROTOCOL: str = (
    "When a user sends a greeting-only message — meaning ONLY a greeting "
    "word with zero additional content (e.g. 'hi', 'hey', 'hello', 'nixi', "
    "'hey nixi') — respond with only: https://nohello.net — no explanation, "
    "no follow-up question, no additional text. "
    "Do NOT apply this to messages with conversational follow-up like "
    "'how are you', 'what's up', or 'how's it going' — those are social "
    "openers, not greeting-only messages. Respond to social openers normally."
)