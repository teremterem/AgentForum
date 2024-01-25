"""
AgentForum errors.
"""
import traceback
import typing

if typing.TYPE_CHECKING:
    from agentforum.forum import InteractionContext


class ForumErrorFormatter:
    """
    Mixin for errors that allows to format an error message before storing it as a Message in ForumTrees.
    """

    def what_to_raise(self) -> BaseException:
        """
        What should the framework actually raise. By default, it is assumed that the current object is the error itself
        (this class doesn't extend Exception because it is meant to either be used as a mixin on another class that
        actually extends BaseException or the implementation of `what_to_raise` should be overridden).
        """
        # noinspection PyTypeChecker
        return self

    async def agenerate_error_message_content(self, ctx: "InteractionContext") -> str:
        """
        Generate the content of the error message. The default implementation outputs the error with complete
        traceback.
        """
        # pylint: disable=unused-argument
        error = self.what_to_raise()
        # pylint: disable=no-member
        return "".join(traceback.format_exception(type(error), error, error.__traceback__))


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
