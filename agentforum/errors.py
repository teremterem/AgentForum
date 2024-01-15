"""AgentForum errors."""


class AgentForumError(Exception):
    """
    Base class AgentForum errors.
    """


class SendClosedError(AgentForumError):
    """
    Raised when a AsyncStreamable is closed for sending.
    """


class WrongHashKeyError(AgentForumError):
    """
    Raised when a hash key is either not found in the storage or refers to an object of a wrong type.
    """
