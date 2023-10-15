"""AgentCache errors."""


class AgentCacheError(Exception):
    """Base class AgentCache errors."""


class MessageBundleClosedError(AgentCacheError):
    """Raised when a MessageBundle is closed and an attempt is made to add a message to it."""
