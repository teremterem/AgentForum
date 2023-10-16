"""AgentCache errors."""


class AgentCacheError(Exception):
    """
    Base class AgentCache errors.
    """


class TokenStreamNotFinishedError(AgentCacheError):
    """
    Raised when a token stream is not finished and an attempt is made to get the full message.
    """


class MessageBundleNotFinishedError(AgentCacheError):
    """
    Raised when a MessageBundle is not finished fetching messages and an attempt is made to get all the messages from
    it using the synchronous method.
    """


class MessageBundleClosedError(AgentCacheError):
    """
    Raised when a MessageBundle is closed and an attempt is made to add a message to it.
    """
