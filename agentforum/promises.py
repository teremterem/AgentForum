# pylint: disable=too-many-arguments,too-many-instance-attributes,protected-access
"""
This module contains wrappers for the pydantic models that turn those models into asynchronous promises.
"""
import asyncio
import typing
from typing import Optional, Any, AsyncIterator, Union

from pydantic import BaseModel, ConfigDict

from agentforum.errors import EmptySequenceError
from agentforum.models import Message, AgentCallMsg, ForwardedMessage, Freeform, ContentChunk
from agentforum.utils import AsyncStreamable, NO_VALUE, IN, SYSTEM_ALIAS

if typing.TYPE_CHECKING:
    from agentforum.conversations import ConversationTracker, HistoryTracker
    from agentforum.forum import Forum
    from agentforum.typing import MessageType, SingleMessageType


class AsyncMessageSequence(AsyncStreamable["_MessageTypeCarrier", "MessagePromise"]):
    """
    An asynchronous iterable over a sequence of messages that are being produced by an agent. Because the sequence is
    AsyncStreamable and relies on internal async queues, the speed at which messages are produced and sent to the
    sequence is independent of the speed at which consumers iterate over them.
    """

    def __init__(
        self,
        conversation_tracker: "ConversationTracker",
        *args,
        default_sender_alias: str,
        do_not_forward_if_possible: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_tracker = conversation_tracker
        self._default_sender_alias = default_sender_alias
        self._do_not_forward_if_possible = do_not_forward_if_possible

    async def acontains_errors(self) -> bool:
        """
        Check if any of the messages in the sequence is an error message.
        """
        async for msg_promise in self:
            if msg_promise.is_error:
                return True
        return False

    async def araise_if_error(self) -> None:
        """
        Raise an error if any of the messages in the sequence is an error message.
        """
        async for msg_promise in self:
            msg_promise.raise_if_error()

    async def aget_concluding_msg_promise(self, raise_if_none: bool = True) -> Optional["MessagePromise"]:
        """
        Get the last message promise in the sequence.
        """
        concluding_message = None
        async for concluding_message in self:
            pass
        if not concluding_message and raise_if_none:
            raise EmptySequenceError("AsyncMessageSequence is empty")
        return concluding_message

    async def amaterialize_concluding_message(self, raise_if_none: bool = True) -> Message:
        """
        Get the last message in the sequence, but return a Message object instead of a MessagePromise object.
        """
        return await (await self.aget_concluding_msg_promise(raise_if_none=raise_if_none)).amaterialize()

    async def amaterialize_concluding_content(self, raise_if_none: bool = True) -> str:
        """
        Get the content of the last message in the sequence as a string.
        """
        return (await self.amaterialize_concluding_message(raise_if_none=raise_if_none)).content

    async def amaterialize_as_list(self) -> list["Message"]:
        """
        Get all the messages in the sequence, but return a list of Message objects instead of MessagePromise objects.
        TODO Oleksandr: emphasize the difference between this method and amaterialize_full_history (maybe
         amaterialize_sequence vs amaterialize_sequence_with_history ?)
        """
        return [await msg.amaterialize() async for msg in self]

    async def aget_full_history(
        self, include_this_message: bool = True, follow_replies: bool = False
    ) -> list["MessagePromise"]:
        """
        Get the full chat history of the conversation branch up to the last message in the sequence.
        """
        concluding_msg_promise = await self.aget_concluding_msg_promise(raise_if_none=False)
        if concluding_msg_promise:
            return await concluding_msg_promise.aget_full_history(
                include_this_message=include_this_message, follow_replies=follow_replies
            )
        return []

    # TODO Oleksandr: also introduce a method that returns full history as an AsyncMessageSequence instead
    #  of a ready-to-use list of MessagePromise objects ?

    async def amaterialize_full_history(
        self, include_this_message: bool = True, follow_replies: bool = False
    ) -> list["Message"]:
        """
        Get the full chat history of the conversation branch up to the last message in the sequence, but return a list
        of Message objects instead of MessagePromise objects.
        """
        return [
            await msg_promise.amaterialize()
            for msg_promise in await self.aget_full_history(
                include_this_message=include_this_message, follow_replies=follow_replies
            )
        ]

    async def _aconvert_incoming_item(
        self, incoming_item: Union["_MessageTypeCarrier", BaseException]
    ) -> AsyncIterator["MessagePromise"]:
        if isinstance(incoming_item, BaseException):
            # This code branch is for unusual exceptions only (for ex. framework level exceptions).
            # Agent level exceptions will be processed by the `else` part, because they will be wrapped into
            # `_MessageTypeCarrier` (they will go through `_MessageProducer.send_zero_or_more_messages` method).
            content = incoming_item
            override_metadata = {}

            # For the unusual exceptions we create a blank history tracker because we don't have access to the proper
            # one.
            from agentforum.conversations import HistoryTracker  # pylint: disable=import-outside-toplevel

            history_tracker = HistoryTracker(self._conversation_tracker.forum)
        else:
            content = incoming_item.zero_or_more_messages
            override_metadata = incoming_item.override_metadata.as_dict()
            history_tracker = incoming_item.history_tracker

        async for msg_promise in self._conversation_tracker.aappend_zero_or_more_messages(
            content=content,
            history_tracker=history_tracker,
            default_sender_alias=self._default_sender_alias,
            do_not_forward_if_possible=self._do_not_forward_if_possible,
            **override_metadata,
        ):
            yield msg_promise

    class _MessageProducer(AsyncStreamable._Producer):
        """
        A context manager that allows sending messages to AsyncMessageSequence.
        """

        def send_zero_or_more_messages(
            self, content: "MessageType", history_tracker: "HistoryTracker", **metadata
        ) -> None:
            """
            Send a message or messages to the sequence this producer is attached to.
            """
            if isinstance(content, dict):
                content = Message(**content)
            elif hasattr(content, "__iter__") and not isinstance(content, (str, tuple, BaseModel)):
                # we are dealing with a "synchronous" collection of messages here - let's freeze it just in case
                # TODO Oleksandr: some sort of "deep freeze" is needed here - items can be mutable dicts or lists
                content = tuple(content)
            # TODO Oleksandr: validate `content` type manually, because in Pydantic it's just Any
            self.send(
                _MessageTypeCarrier(
                    zero_or_more_messages=content,
                    history_tracker=history_tracker,
                    override_metadata=Freeform(**metadata),
                )
            )


class StreamedMessage(AsyncStreamable[IN, ContentChunk]):
    """
    A message that is streamed token by token instead of being returned all at once. StreamedMessage only maintains
    content (as a stream of tokens) and metadata. It does not maintain final_sender_alias, prev_msg_hash_key, etc.
    """

    def __init__(self, *args, override_metadata: Optional[dict[str, Any]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._metadata = {}
        self._override_metadata = override_metadata or {}

        self._aggregated_content: Optional[str] = None
        self._aggregated_metadata: Optional[Freeform] = None

    async def amaterialize_content(self) -> str:
        """
        Get the full content of the message as a string.
        """
        if self._aggregated_content is None:
            # asyncio.Lock could have been used here, but there is not much harm in running it twice in a rare case
            self._aggregated_content = "".join([token.text async for token in self])
        return self._aggregated_content

    async def amaterialize_metadata(self) -> Freeform:
        """
        Build metadata from the metadata provided to the constructor and the metadata collected during streaming.
        Metadata provided to the constructor (override_metadata) takes precedence over the metadata collected during
        streaming.
        """
        if self._aggregated_metadata is None:
            # asyncio.Lock could have been used here, but there is not much harm in running it twice in a rare case
            await self.amaterialize_content()  # make sure all the tokens are collected
            self._aggregated_metadata = Freeform(**self._metadata, **self._override_metadata)
        return self._aggregated_metadata


# noinspection PyProtectedMember
class MessagePromise:
    """
    A promise to materialize a message.
    """

    def __init__(
        self,
        forum: "Forum",
        content: Optional["SingleMessageType"] = None,
        default_sender_alias: Optional[str] = None,
        do_not_forward_if_possible: bool = True,
        branch_from: Optional["MessagePromise"] = None,
        reply_to: Optional["MessagePromise"] = None,
        materialized_msg: Optional[Message] = None,
        is_error: bool = False,
        error: Optional[BaseException] = None,
        **override_metadata,
    ) -> None:
        # pylint: disable=too-many-boolean-expressions
        if materialized_msg and (
            content is not None
            or default_sender_alias
            or branch_from
            or reply_to
            or is_error
            or error
            or override_metadata
        ):
            raise ValueError(
                "If `materialized_msg` is provided, `content`, `default_sender_alias`, `branch_from`, `reply_to`, "
                "`is_error`, `error` and `override_metadata` must not be provided."
            )
        if not materialized_msg and (content is None or not default_sender_alias):
            raise ValueError(
                "If `materialized_msg` is not provided, `content` and `default_sender_alias` must be provided."
            )

        self.forum = forum

        self._content = content
        self._default_sender_alias = default_sender_alias
        self._do_not_forward_if_possible = do_not_forward_if_possible
        self._branch_from = branch_from
        self._reply_to = reply_to
        self._override_metadata = override_metadata

        if materialized_msg:
            self.is_error = materialized_msg.is_error
            self._error = materialized_msg._error
        else:
            self.is_error = is_error
            self._error = error

        self._materialized_msg: Optional[Message] = materialized_msg
        self._lock = asyncio.Lock()

    def __aiter__(self) -> AsyncIterator[ContentChunk]:
        if isinstance(self._content, (StreamedMessage, MessagePromise)):
            return self._content.__aiter__()

        async def _aiter() -> AsyncIterator[ContentChunk]:
            """
            Return only one element - the whole message.
            """
            if self._materialized_msg:
                yield ContentChunk(text=self._materialized_msg.content)
            elif isinstance(self._content, Message):
                yield ContentChunk(text=self._content.content)
            else:
                yield ContentChunk(text=self._content)

        return _aiter()

    def raise_if_error(self) -> None:
        """
        Raise an error if this message is an error message.
        """
        if self.is_error:
            raise self._error

    @property
    def is_agent_call(self) -> bool:
        """
        Check if this message is a call to an agent.
        """
        if self._materialized_msg:
            return isinstance(self._materialized_msg, AgentCallMsg)
        return False  # this will be overridden in AgentCallMsgPromise

    async def amaterialize(self) -> Message:
        """
        Get the full message. This method will "await" until all the tokens are received (or whatever else needs to be
        waited for before the actual message can be constructed and stored in the storage) and then return the message.
        """
        if not self._materialized_msg:
            async with self._lock:
                if not self._materialized_msg:
                    self._materialized_msg = await self._amaterialize_impl()
                    await self.forum.forum_trees.astore_immutable(self._materialized_msg)

                    # from now on the source of truth is self._materialized_msg
                    self._content = None
                    self._default_sender_alias = None
                    self._branch_from = None
                    self._reply_to = None
                    self._override_metadata = None

        return self._materialized_msg

    async def amaterialize_content(self) -> str:
        """
        Get the full content of the message as a string.
        """
        return (await self.amaterialize()).content

    async def aget_previous_msg_promise(self) -> Optional["MessagePromise"]:
        """
        Get the previous MessagePromise in this conversation branch.
        """
        if self._materialized_msg:
            if self._materialized_msg.prev_msg_hash_key:
                message = await self.forum.forum_trees.aretrieve_message(self._materialized_msg.prev_msg_hash_key)
                return MessagePromise(forum=self.forum, materialized_msg=message)
            return None
        return await self._aget_previous_msg_promise_impl()

    async def aget_reply_to_msg_promise(self) -> Optional["MessagePromise"]:
        """
        Get the MessagePromise that this MessagePromise is a reply to.
        """
        if self._materialized_msg:
            if self._materialized_msg.reply_to_msg_hash_key:
                message = await self.forum.forum_trees.aretrieve_message(self._materialized_msg.reply_to_msg_hash_key)
                return MessagePromise(forum=self.forum, materialized_msg=message)
            return None
        return self._reply_to

    async def _amaterialize_impl(self) -> Message:
        if self._branch_from and self._branch_from is not NO_VALUE:
            prev_msg_hash_key = (await self._branch_from.amaterialize()).hash_key
        else:
            prev_msg_hash_key = None

        if self._reply_to:
            reply_to_msg_hash_key = (await self._reply_to.amaterialize()).hash_key
        else:
            reply_to_msg_hash_key = None

        override_metadata = dict(self._override_metadata)
        override_sender_alias = override_metadata.pop("final_sender_alias", None)

        if isinstance(self._content, (str, StreamedMessage)):
            if isinstance(self._content, StreamedMessage):
                msg_content = await self._content.amaterialize_content()
                materialized_metadata = (await self._content.amaterialize_metadata()).as_dict()
                final_sender_alias = override_sender_alias or materialized_metadata.pop("final_sender_alias", None)
                metadata = {
                    **materialized_metadata,
                    **override_metadata,
                }
            else:
                # string content
                msg_content = self._content
                final_sender_alias = override_sender_alias
                metadata = override_metadata

            msg = Message(
                forum_trees=self.forum.forum_trees,
                final_sender_alias=final_sender_alias or self._default_sender_alias,
                content=msg_content,
                prev_msg_hash_key=prev_msg_hash_key,
                reply_to_msg_hash_key=reply_to_msg_hash_key,
                is_error=self.is_error,
                is_detached=False,
                **metadata,
            )
            msg._error = self._error
            return msg

        if isinstance(self._content, (Message, MessagePromise)):
            if isinstance(self._content, MessagePromise):
                msg_before_forward = await self._content.amaterialize()
            else:
                msg_before_forward = self._content

            if (
                (not self._do_not_forward_if_possible)
                or self._override_metadata
                or (self._branch_from is not NO_VALUE and prev_msg_hash_key != msg_before_forward.prev_msg_hash_key)
                or reply_to_msg_hash_key != msg_before_forward.reply_to_msg_hash_key
            ):
                # the message must be forwarded because either we are not actively trying to avoid forwarding
                # (do_not_forward_if_possible is False), or additional metadata was provided (message forwarding is
                # the only way to attach metadata to a message), or the original message is branched from a different
                # message than this message promise (which also means that message forwarding is the only way)
                forwarded_msg = ForwardedMessage(
                    forum_trees=self.forum.forum_trees,
                    final_sender_alias=override_sender_alias or self._default_sender_alias,
                    msg_before_forward_hash_key=msg_before_forward.hash_key,
                    prev_msg_hash_key=prev_msg_hash_key,
                    reply_to_msg_hash_key=reply_to_msg_hash_key,
                    is_error=self.is_error,
                    **{
                        **msg_before_forward.metadata_as_dict(),
                        **override_metadata,
                    },
                )
                forwarded_msg._error = self._error
                forwarded_msg._set_msg_before_forward(msg_before_forward)
                return forwarded_msg

            # TODO Oleksandr: this message is stored in the storage twice, because it is "materialized" twice
            return msg_before_forward

        raise ValueError(f"Unexpected message content type: {type(self._content)}")

    async def _aget_previous_msg_promise_impl(self) -> Optional["MessagePromise"]:
        if self._do_not_forward_if_possible and self._branch_from is NO_VALUE:
            # this message promise doesn't have a previous message promise of its own but there may be an "original"
            # message inside self._content which is not going to be forwarded (do_not_forward_if_possible is True),
            # hence we should try to work with the "original" message's branch instead of starting a new branch (which
            # would have been the case if we just returned self._branch_from as it's value is being None)
            if isinstance(self._content, MessagePromise):
                return await self._content.aget_previous_msg_promise()

            if isinstance(self._content, Message):
                message = await self._content.aget_previous_msg()
                if not message:
                    return None
                return MessagePromise(forum=self.forum, materialized_msg=message)

        return None if self._branch_from is NO_VALUE else self._branch_from

    async def aget_full_history(
        self, include_this_message: bool = True, follow_replies: bool = False
    ) -> list["MessagePromise"]:
        """
        Get the full chat history of the conversation branch up to this message. Returns a list of MessagePromise
        objects.
        """
        # TODO Oleksandr: introduce a limit on the number of messages to fetch ?
        msg_promise = self
        result = [msg_promise] if include_this_message else []
        while msg_promise := (
            # TODO TODO TODO Oleksandr: split into two separate methods ?
            await msg_promise.aget_reply_to_msg_promise()
            if follow_replies
            else await msg_promise.aget_previous_msg_promise()
        ):
            result.append(msg_promise)
        result.reverse()
        return result

    async def amaterialize_full_history(
        self, include_this_message: bool = True, follow_replies: bool = False
    ) -> list[Message]:
        """
        Get the full chat history of the conversation branch up to this message, but return a list of Message objects
        instead of MessagePromise objects.
        """
        return [
            await msg_promise.amaterialize()
            for msg_promise in await self.aget_full_history(
                include_this_message=include_this_message, follow_replies=follow_replies
            )
        ]


class AgentCallMsgPromise(MessagePromise):
    """
    A promise to materialize an agent call message. Agent call messages are special because they are not produced by
    any agent, but rather by the forum itself. They are used to capture parameters of agent calls (a sequence of
    request messages and agent function kwargs) so the results of those calls can be cached later.
    """

    def __init__(
        self, forum: "Forum", request_messages: AsyncMessageSequence, receiving_agent_alias: str, **function_kwargs
    ) -> None:
        super().__init__(
            forum=forum, content=receiving_agent_alias, default_sender_alias=SYSTEM_ALIAS, **function_kwargs
        )
        self._request_messages = request_messages

    @property
    def is_agent_call(self) -> bool:
        return True

    async def _amaterialize_impl(self) -> Message:
        messages = await self._request_messages.amaterialize_as_list()
        if messages:
            msg_seq_start_hash_key = messages[0].hash_key
            msg_seq_end_hash_key = messages[-1].hash_key
        else:
            msg_seq_start_hash_key = None
            msg_seq_end_hash_key = None

        return AgentCallMsg(
            forum_trees=self.forum.forum_trees,
            receiver_alias=self._content,  # receiving_agent_alias
            final_sender_alias=SYSTEM_ALIAS,  # agent calls should be cacheable and reuseable by other agents
            function_kwargs=self._override_metadata,  # function_kwargs from the constructor
            prev_msg_hash_key=msg_seq_end_hash_key,  # agent call gets attached to the end of the request messages
            msg_seq_start_hash_key=msg_seq_start_hash_key,
        )

    async def _aget_previous_msg_promise_impl(self) -> Optional[MessagePromise]:
        msg_promise = None
        async for msg_promise in self._request_messages:
            pass
        return msg_promise


class _MessageTypeCarrier(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    zero_or_more_messages: Any  # should be `MessageType` but Pydantic v2 seems to be confused by it
    history_tracker: Any  # TODO TODO TODO Oleksandr: should be `HistoryTracker` but there are circular dependencies
    override_metadata: Freeform = Freeform()
