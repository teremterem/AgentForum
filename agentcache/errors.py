"""AgentCache errors."""


class AgentCacheError(Exception):
    """
    Base class AgentCache errors.
    """


class SendClosedError(AgentCacheError):
    """
    Raised when a Broadcastable is closed for sending.
    """
