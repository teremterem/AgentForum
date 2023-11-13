"""AgentCache errors."""


class AgentCacheError(Exception):
    """
    Base class AgentCache errors.
    """


class SendClosedError(AgentCacheError):
    """
    Raised when a AsyncStreamable is closed for sending.
    """
