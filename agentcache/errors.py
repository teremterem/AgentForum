"""AgentCache errors."""


class AgentCacheError(Exception):
    """
    Base class AgentCache errors.
    """


class SendClosedError(AgentCacheError):
    """
    Raised when a AsyncStreamable is closed for sending.
    """


class AsyncIterationError(AgentCacheError):
    """
    Raised when an error is encountered during iteration over an AsyncIterable.
    """
