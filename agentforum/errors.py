"""
AgentForum errors.
"""

import traceback
import typing
from typing import Optional

if typing.TYPE_CHECKING:
    from agentforum.promises import MessagePromise


class AgentForumError(Exception):
    """
    Base class AgentForum errors.
    """


class FormattedForumError(AgentForumError):
    """
    Base class for all exceptions in the AgentForum project that can be formatted into a forum tree message.
    """

    def __init__(
        self, *args, original_error: Optional[BaseException] = None, include_stack_trace: bool = False, **metadata
    ):
        super().__init__(*args)
        self.original_error = original_error or self
        self.include_stack_trace = include_stack_trace
        self.metadata = metadata

    # noinspection PyUnusedLocal
    async def agenerate_error_message(
        self, previous_msg_promise: "MessagePromise", reply_to_msg_promise: "MessagePromise"
    ) -> str:
        """
        Generate the content of the error message. The default implementation outputs the error with complete
        traceback.
        """
        # pylint: disable=unused-argument
        if self.include_stack_trace:
            return "".join(
                traceback.format_exception(
                    type(self.original_error), self.original_error, self.original_error.__traceback__
                )
            )
        return "".join(traceback.format_exception_only(type(self.original_error), self.original_error)).strip()


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


class DetachedMessageError(AgentForumError):
    """
    Raised when a message is detached and an operation is attempted on it which is not allowed for detached messages.
    """


class NoAskingAgentError(AgentForumError):
    """
    Raised when no asking agent is found up the chain of parent InteractionContexts, or .respond() is called on a
    non-asking InteractionContext, or .response_sequence() is called on a non-asking AgentCall.
    """
