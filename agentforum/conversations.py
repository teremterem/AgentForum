# pylint: disable=protected-access
"""
A module that contains the HistoryTracker and ConversationTracker class. See the class docstrings for more details.
"""
import typing
from typing import Optional, AsyncIterator, Union

from agentforum.errors import FormattedForumError
from agentforum.models import Message
from agentforum.promises import MessagePromise, StreamedMessage, AsyncMessageSequence
from agentforum.utils import Sentinel

if typing.TYPE_CHECKING:
    from agentforum.forum import Forum
    from agentforum.typing import MessageType


class ConversationTracker:
    """
    An object that tracks the tip of a conversation (a chain of messages that are replies to each other).
    """

    def __init__(self, forum: "Forum", reply_to: Optional[Union[MessagePromise, AsyncMessageSequence]] = None) -> None:
        self.forum = forum
        self._latest_msg_promise = reply_to

    # noinspection PyProtectedMember
    async def aappend_zero_or_more_messages(
        self,
        content: "MessageType",
        default_sender_alias: str,
        history_tracker: "HistoryTracker",
        do_not_forward_if_possible: bool = True,
        **override_metadata,
    ) -> AsyncIterator[MessagePromise]:
        """
        Append zero or more messages to the conversation. Returns an async iterator that yields message promises.
        """
        # pylint: disable=too-many-branches,too-many-statements
        # TODO TODO TODO Oleksandr: is locking necessary in this method ?
        if isinstance(self._latest_msg_promise, AsyncMessageSequence):
            self._latest_msg_promise = await self._latest_msg_promise.aget_concluding_msg_promise(raise_if_none=False)
        if isinstance(history_tracker._latest_msg_promise, AsyncMessageSequence):
            history_tracker._latest_msg_promise = (
                await history_tracker._latest_msg_promise.aget_concluding_msg_promise(raise_if_none=False)
            )

        if isinstance(content, BaseException):
            if isinstance(content, FormattedForumError):
                formatted_error = content
            else:
                formatted_error = FormattedForumError(original_error=content)

            msg_promise = MessagePromise(
                forum=self.forum,
                content=await formatted_error.agenerate_error_message(
                    previous_msg_promise=history_tracker._latest_msg_promise,
                    reply_to_msg_promise=self._latest_msg_promise,
                ),
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=history_tracker._latest_msg_promise,
                reply_to=self._latest_msg_promise,
                is_error=True,
                error=content,
                **{
                    **formatted_error.metadata,
                    **override_metadata,
                },
            )
            history_tracker._latest_msg_promise = msg_promise
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, MessagePromise):
            msg_promise = MessagePromise(
                forum=self.forum,
                content=content,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=history_tracker._latest_msg_promise,
                reply_to=self._latest_msg_promise,
                is_error=content.is_error,
                error=content._error,
                **override_metadata,
            )
            history_tracker._latest_msg_promise = msg_promise
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, dict):
            msg_promise = MessagePromise(
                forum=self.forum,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=history_tracker._latest_msg_promise,
                reply_to=self._latest_msg_promise,
                **{
                    **content,
                    **override_metadata,
                },
            )
            history_tracker._latest_msg_promise = msg_promise
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, Message):
            if content.is_detached:
                msg_fields = content.as_dict()
                msg_fields.pop("reply_to_msg_hash_key", None)

                msg_promise = MessagePromise(
                    forum=self.forum,
                    default_sender_alias=default_sender_alias,
                    do_not_forward_if_possible=do_not_forward_if_possible,
                    branch_from=history_tracker._latest_msg_promise,
                    reply_to=self._latest_msg_promise,
                    **{
                        **msg_fields,
                        **override_metadata,
                    },
                )
            else:
                msg_promise = MessagePromise(
                    forum=self.forum,
                    content=content,
                    default_sender_alias=default_sender_alias,
                    do_not_forward_if_possible=do_not_forward_if_possible,
                    branch_from=history_tracker._latest_msg_promise,
                    reply_to=self._latest_msg_promise,
                    is_error=content.is_error,
                    error=content._error,
                    **override_metadata,
                )
            history_tracker._latest_msg_promise = msg_promise
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, (str, StreamedMessage)):
            msg_promise = MessagePromise(
                forum=self.forum,
                content=content,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=history_tracker._latest_msg_promise,
                reply_to=self._latest_msg_promise,
                **override_metadata,
            )
            history_tracker._latest_msg_promise = msg_promise
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif hasattr(content, "__iter__"):
            # this is not a single message, this is a collection of messages
            for sub_msg in content:
                async for msg_promise in self.aappend_zero_or_more_messages(
                    content=sub_msg,
                    default_sender_alias=default_sender_alias,
                    do_not_forward_if_possible=do_not_forward_if_possible,
                    history_tracker=history_tracker,
                    **override_metadata,
                ):
                    yield msg_promise
        elif hasattr(content, "__aiter__"):
            # this is not a single message, this is an asynchronous collection of messages
            async for sub_msg in content:
                async for msg_promise in self.aappend_zero_or_more_messages(
                    content=sub_msg,
                    default_sender_alias=default_sender_alias,
                    do_not_forward_if_possible=do_not_forward_if_possible,
                    history_tracker=history_tracker,
                    **override_metadata,
                ):
                    yield msg_promise
        else:
            raise ValueError(f"Unexpected message content type: {type(content)}")


class HistoryTracker:
    """
    An object that tracks the tip of a message tree branch (aka a history branch).

    If `branch_from` is set to NO_VALUE then it means that whether this conversation is branched off of an existing
    branch of messages or not will be determined by the messages that are passed into this conversation later.
    """

    def __init__(
        self, forum: "Forum", branch_from: Optional[Union[MessagePromise, AsyncMessageSequence, Sentinel]] = None
    ) -> None:
        self.forum = forum
        self._latest_msg_promise = branch_from
