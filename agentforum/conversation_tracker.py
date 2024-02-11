# pylint: disable=protected-access
"""
A module that contains the ConversationTracker class. See the class docstring for more details.
"""
import typing
from typing import Optional, AsyncIterator, Union

from agentforum.errors import FormattedForumError
from agentforum.models import Message
from agentforum.promises import MessagePromise, StreamedMessage
from agentforum.utils import Sentinel, NO_VALUE

if typing.TYPE_CHECKING:
    from agentforum.forum import Forum
    from agentforum.typing import MessageType


class ConversationTracker:
    """
    An object that tracks the tip of a conversation branch.

    If `branch_from` is set to NO_VALUE then it means that whether this conversation is branched off of an existing
    branch of messages or not will be determined by the messages that are passed into this conversation later.
    """

    def __init__(self, forum: "Forum", branch_from: Optional[Union[MessagePromise, Sentinel]] = None) -> None:
        self.forum = forum
        self._latest_msg_promise = branch_from

    @property
    def has_prior_history(self) -> bool:
        """
        Check if there is prior history in this conversation.
        """
        return self._latest_msg_promise and self._latest_msg_promise != NO_VALUE

    # noinspection PyProtectedMember
    async def aappend_zero_or_more_messages(
        self,
        content: "MessageType",
        default_sender_alias: str,
        do_not_forward_if_possible: bool = True,
        **override_metadata,
    ) -> AsyncIterator[MessagePromise]:
        """
        Append zero or more messages to the conversation. Returns an async iterator that yields message promises.
        """
        # pylint: disable=too-many-branches
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
                is_error=True,
                error=content,
                **{
                    **formatted_error.metadata,
                    **override_metadata,
                },
            )
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, MessagePromise):
            msg_promise = MessagePromise(
                forum=self.forum,
                content=content,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                is_error=content.is_error,
                error=content._error,
                **override_metadata,
            )
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, dict):
            msg_promise = MessagePromise(
                forum=self.forum,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                **{
                    **content,
                    **override_metadata,
                },
            )
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
                    branch_from=self._latest_msg_promise,
                    **{
                        **msg_fields,
                        **override_metadata,
                    },
                )
            else:
                if content.reply_to_msg_hash_key:
                    reply_to_msg = await self.forum.forum_trees.aretrieve_message(content.reply_to_msg_hash_key)
                    reply_to = MessagePromise(forum=self.forum, materialized_msg=reply_to_msg)
                else:
                    reply_to = None

                msg_promise = MessagePromise(
                    forum=self.forum,
                    content=content,
                    default_sender_alias=default_sender_alias,
                    do_not_forward_if_possible=do_not_forward_if_possible,
                    branch_from=self._latest_msg_promise,
                    reply_to=reply_to,  # TODO TODO TODO Oleksandr: should I really pass it here ?
                    is_error=content.is_error,
                    error=content._error,
                    **override_metadata,
                )
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, (str, StreamedMessage)):
            msg_promise = MessagePromise(
                forum=self.forum,
                content=content,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                **override_metadata,
            )
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif hasattr(content, "__iter__"):
            # this is not a single message, this is a collection of messages
            for sub_msg in content:
                async for msg_promise in self.aappend_zero_or_more_messages(
                    content=sub_msg,
                    default_sender_alias=default_sender_alias,
                    do_not_forward_if_possible=do_not_forward_if_possible,
                    **override_metadata,
                ):
                    self._latest_msg_promise = msg_promise
                    yield msg_promise
        elif hasattr(content, "__aiter__"):
            # this is not a single message, this is an asynchronous collection of messages
            async for sub_msg in content:
                async for msg_promise in self.aappend_zero_or_more_messages(
                    content=sub_msg,
                    default_sender_alias=default_sender_alias,
                    do_not_forward_if_possible=do_not_forward_if_possible,
                    **override_metadata,
                ):
                    self._latest_msg_promise = msg_promise
                    yield msg_promise
        else:
            raise ValueError(f"Unexpected message content type: {type(content)}")
