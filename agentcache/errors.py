"""AgentCache errors."""


class AgentCacheError(Exception):
    """
    Base class AgentCache errors.
    """


class SendClosedError(AgentCacheError):
    """
    Raised when a AsyncStreamable is closed for sending.
    """


class AsyncNeededError(AgentCacheError):
    """
    Raised when a synchronous method is called on an asynchronous object.
    """
