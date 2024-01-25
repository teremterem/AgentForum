"""AgentForum errors."""


class AgentForumError(Exception):
    """
    Base class AgentForum errors.
    """


class SendClosedError(AgentForumError):
    """
    Raised when a AsyncStreamable is closed for sending.
    """


class ImmutableDoesNotExist(AgentForumError):
    """
    Raised when an Immutable object does not exist.
    """


class WrongImmutableTypeError(AgentForumError):
    """
    Raised when an Immutable object is of the wrong type.
    """


class EmptySequenceError(AgentForumError):
    """
    Raised when a sequence is empty.
    """
