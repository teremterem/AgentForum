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
    def __init__(self, forum: "Forum", reply_to: Optional[Union[MessagePromise, AsyncMessageSequence]] = None) -> None:
        self.forum = forum
        self._latest_msg_promise = reply_to


class HistoryTracker:
    """
    An object that tracks the tip of a message tree branch.

    If `branch_from` is set to NO_VALUE then it means that whether this conversation is branched off of an existing
    branch of messages or not will be determined by the messages that are passed into this conversation later.
    """

    # TODO TODO TODO TODO TODO Oleksandr: two modes ? `branch_from` and `reply_to` ? both modes work simultaneously ?

    def __init__(
        self, forum: "Forum", branch_from: Optional[Union[MessagePromise, AsyncMessageSequence, Sentinel]] = None
    ) -> None:
        self.forum = forum
        self._latest_msg_promise = branch_from

    # noinspection PyProtectedMember
    async def aappend_zero_or_more_messages(
        self,
        content: "MessageType",
        default_sender_alias: str,
        do_not_forward_if_possible: bool = True,
        conversation: Optional[Union[MessagePromise, AsyncMessageSequence]] = None,
        **override_metadata,
    ) -> AsyncIterator[MessagePromise]:
        """
        Append zero or more messages to the conversation. Returns an async iterator that yields message promises.
        """
        # pylint: disable=too-many-branches,too-many-statements
        # TODO TODO TODO Oleksandr: is locking necessary in this method ?
        if isinstance(self._latest_msg_promise, AsyncMessageSequence):
            self._latest_msg_promise = await self._latest_msg_promise.aget_concluding_msg_promise(raise_if_none=False)
        if conversation:
            if isinstance(conversation, AsyncMessageSequence):
                conversation._latest_msg_promise = await conversation._latest_msg_promise.aget_concluding_msg_promise(
                    raise_if_none=False
                )
            reply_to = conversation._latest_msg_promise
        else:
            reply_to = None

        if isinstance(content, BaseException):
            if isinstance(content, FormattedForumError):
                formatted_error = content
            else:
                formatted_error = FormattedForumError(original_error=content)

            msg_promise = MessagePromise(
                forum=self.forum,
                content=await formatted_error.agenerate_error_message(self._latest_msg_promise),
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                reply_to=reply_to,
                is_error=True,
                error=content,
                **{
                    **formatted_error.metadata,
                    **override_metadata,
                },
            )
            self._latest_msg_promise = msg_promise
            if conversation:
                conversation._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, MessagePromise):
            msg_promise = MessagePromise(
                forum=self.forum,
                content=content,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                reply_to=reply_to,
                is_error=content.is_error,
                error=content._error,
                **override_metadata,
            )
            self._latest_msg_promise = msg_promise
            if conversation:
                conversation._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, dict):
            msg_promise = MessagePromise(
                forum=self.forum,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                reply_to=reply_to,
                **{
                    **content,
                    **override_metadata,
                },
            )
            self._latest_msg_promise = msg_promise
            if conversation:
                conversation._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, Message):
            if content.is_detached:
                msg_fields = content.as_dict()
                msg_fields.pop("reply_to_msg_hash_key", None)

                msg_promise = MessagePromise(
                    forum=self.forum,
                    default_sender_alias=default_sender_alias,
                    do_not_forward_if_possible=do_not_forward_if_possible,
                    branch_from=self._latest_msg_promise,
                    reply_to=reply_to,
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
                    branch_from=self._latest_msg_promise,
                    reply_to=reply_to,
                    is_error=content.is_error,
                    error=content._error,
                    **override_metadata,
                )
            self._latest_msg_promise = msg_promise
            if conversation:
                conversation._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, (str, StreamedMessage)):
            msg_promise = MessagePromise(
                forum=self.forum,
                content=content,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                reply_to=reply_to,
                **override_metadata,
            )
            self._latest_msg_promise = msg_promise
            if conversation:
                conversation._latest_msg_promise = msg_promise
            yield msg_promise

        elif hasattr(content, "__iter__"):
            # this is not a single message, this is a collection of messages
            for sub_msg in content:
                async for msg_promise in self.aappend_zero_or_more_messages(
                    content=sub_msg,
                    default_sender_alias=default_sender_alias,
                    do_not_forward_if_possible=do_not_forward_if_possible,
                    conversation=conversation,
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
                    conversation=conversation,
                    **override_metadata,
                ):
                    yield msg_promise
        else:
            raise ValueError(f"Unexpected message content type: {type(content)}")
