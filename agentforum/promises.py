"""This module contains wrappers for the pydantic models that turn those models into asynchronous promises."""
import asyncio
import typing
from typing import Optional, List, Dict, Any, AsyncIterator, Union

from agentforum.models import Message, AgentCallMsg, ForwardedMessage, Freeform, MessageParameters, ContentChunk
from agentforum.typing import IN, MessageType, SingleMessageType
from agentforum.utils import AsyncStreamable

if typing.TYPE_CHECKING:
    from agentforum.forum import Forum, ConversationTracker


class MessageSequence(AsyncStreamable[MessageParameters, "MessagePromise"]):
    # TODO Oleksandr: rename to AsyncMessageSequence
    """
    An asynchronous iterable over a sequence of messages that are being produced by an agent. Because the sequence is
    AsyncStreamable and relies on internal async queues, the speed at which messages are produced and sent to the
    sequence is independent of the speed at which consumers iterate over them.
    """

    def __init__(
        self,
        conversation: "ConversationTracker",
        *args,
        default_sender_alias: str,
        do_not_forward_if_possible: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._conversation = conversation
        self._default_sender_alias = default_sender_alias
        self._do_not_forward_if_possible = do_not_forward_if_possible

    async def aget_concluding_message(self, raise_if_none: bool = True) -> Optional["MessagePromise"]:
        """Get the last message in the sequence."""
        concluding_message = None
        async for concluding_message in self:
            pass
        if not concluding_message and raise_if_none:
            # TODO Oleksandr: introduce a custom exception for this case
            raise ValueError("MessageSequence is empty")
        return concluding_message

    async def amaterialize_concluding_message(self, raise_if_none: bool = True) -> Message:
        """Get the last message in the sequence, but return a Message object instead of a MessagePromise object."""
        return await (await self.aget_concluding_message(raise_if_none=raise_if_none)).amaterialize()

    async def amaterialize_all(self) -> List["Message"]:
        """
        Get all the messages in the sequence, but return a list of Message objects instead of MessagePromise objects.
        """
        return [await msg.amaterialize() async for msg in self]

    async def aget_full_history(
        self, skip_agent_calls: bool = True, include_this_message: bool = True
    ) -> List["MessagePromise"]:
        """Get the full chat history of the conversation branch up to the last message in the sequence."""
        return await (await self.aget_concluding_message()).aget_history(
            skip_agent_calls=skip_agent_calls, include_this_message=include_this_message
        )

    async def amaterialize_full_history(
        self, skip_agent_calls: bool = True, include_this_message: bool = True
    ) -> List["Message"]:
        """
        Get the full chat history of the conversation branch up to the last message in the sequence, but return a list
        of Message objects instead of MessagePromise objects.
        """
        return await (await self.aget_concluding_message()).amaterialize_history(
            skip_agent_calls=skip_agent_calls, include_this_message=include_this_message
        )

    async def _aconvert_incoming_item(
        self, incoming_item: MessageParameters
    ) -> AsyncIterator[Union["MessagePromise", BaseException]]:
        sender_alias = incoming_item.override_sender_alias or self._default_sender_alias
        try:
            if isinstance(incoming_item.content, BaseException):
                raise incoming_item.content

            async for msg_promise in self._conversation.aappend_zero_or_more_messages(
                content=incoming_item.content,
                sender_alias=sender_alias,
                do_not_forward_if_possible=self._do_not_forward_if_possible,
                **incoming_item.metadata,
            ):
                yield msg_promise

        except BaseException as exc:  # pylint: disable=broad-except
            # TODO Oleksandr: introduce the concept of ErrorMessage
            yield exc

    class _MessageProducer(AsyncStreamable._Producer):  # pylint: disable=protected-access
        """A context manager that allows sending messages to MessageSequence."""

        def send_zero_or_more_messages(
            self, content: MessageType, override_sender_alias: Optional[str] = None, **metadata
        ) -> None:
            """Send a message or messages to the sequence this producer is attached to."""
            if not isinstance(content, (str, tuple)) and hasattr(content, "__iter__"):
                content = tuple(content)
            self.send(
                MessageParameters(content=content, override_sender_alias=override_sender_alias, metadata=metadata)
            )


class StreamedMessage(AsyncStreamable[IN, ContentChunk]):
    """
    A message that is streamed token by token instead of being returned all at once. StreamedMessage only maintains
    content (as a stream of tokens) and metadata. It does not maintain sender_alias, prev_msg_hash_key, etc.
    """

    def __init__(self, *args, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metadata = dict(metadata or {})


class MessagePromise:
    """A promise to materialize a message."""

    def __init__(
        self,
        forum: "Forum",
        content: Optional[SingleMessageType] = None,
        default_sender_alias: Optional[str] = None,
        override_sender_alias: Optional[str] = None,
        do_not_forward_if_possible: bool = True,
        branch_from: Optional["MessagePromise"] = None,
        materialized_msg: Optional[Message] = None,
        **metadata,
    ) -> None:
        if materialized_msg and (
            content is not None or default_sender_alias or override_sender_alias or branch_from or metadata
        ):
            raise ValueError(
                "If materialized_msg is provided, content, default_sender_alias, override_sender_alias, "
                "branch_from and metadata must not be provided."
            )

        self.forum = forum
        self._content = content
        self._default_sender_alias = default_sender_alias
        self._override_sender_alias = override_sender_alias
        self._do_not_forward_if_possible = do_not_forward_if_possible
        self._branch_from = branch_from
        self._metadata = metadata

        self._materialized_msg: Optional[Message] = materialized_msg
        self._lock = asyncio.Lock()

    def __aiter__(self) -> AsyncIterator[ContentChunk]:
        if isinstance(self._content, StreamedMessage):
            return self._content.__aiter__()
        return self

    async def __anext__(self) -> ContentChunk:
        if isinstance(self._content, StreamedMessage):
            raise RuntimeError("You need to call __aiter__() and iterate over that object instead.")

        # this message is not streamed, so we need to materialize it and return the whole content
        message = await self.amaterialize()
        return ContentChunk(text=message.content)

    @property
    def is_agent_call(self) -> bool:
        """Check if this message is a call to an agent."""
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
                    await self.forum.immutable_storage.astore_immutable(self._materialized_msg)

                    # from now on the source of truth is self._materialized_msg
                    self._content = None
                    self._default_sender_alias = None
                    self._override_sender_alias = None
                    self._branch_from = None
                    self._metadata = None

        return self._materialized_msg

    async def aget_previous_msg_promise(self, skip_agent_calls: bool = True) -> Optional["MessagePromise"]:
        """Get the previous MessagePromise in this conversation branch."""
        prev_msg_promise = await self._aget_previous_msg_promise_impl()

        if skip_agent_calls:
            while prev_msg_promise.is_agent_call:
                prev_msg_promise = await prev_msg_promise._aget_previous_msg_promise_impl()

        return prev_msg_promise

    async def _amaterialize_impl(self) -> Message:
        sender_alias = self._override_sender_alias or self._default_sender_alias
        prev_msg_hash_key = (await self._branch_from.amaterialize()).hash_key if self._branch_from else None

        if isinstance(self._content, (str, StreamedMessage)):
            if isinstance(self._content, StreamedMessage):
                msg_content = "".join([token.text async for token in self._content])
                # let's merge the metadata from the stream with the metadata provided to the constructor
                metadata = Freeform(**self._content.metadata, **self._metadata)
            else:
                msg_content = self._content
                metadata = Freeform(**self._metadata)

            return Message(
                content=msg_content,
                sender_alias=sender_alias,
                metadata=metadata,
                prev_msg_hash_key=prev_msg_hash_key,
            )

        if isinstance(self._content, (Message, MessagePromise)):
            if isinstance(self._content, MessagePromise):
                original_msg = await self._content.amaterialize()
            else:
                original_msg = self._content

            if (
                (not self._do_not_forward_if_possible)
                or self._metadata
                or prev_msg_hash_key != original_msg.prev_msg_hash_key
            ):
                # the message must be forwarded because either we are not actively trying to avoid forwarding
                # (do_not_forward_if_possible is False), or additional metadata was provided (message forwarding is
                # the only way to attach metadata to a message), or the original message is branched from a different
                # message than this message promise (which also means that message forwarding is the only way)
                forwarded_msg = ForwardedMessage(
                    original_msg_hash_key=original_msg.hash_key,
                    sender_alias=sender_alias,
                    # let's merge the metadata from the original message with the metadata provided to the constructor
                    metadata=Freeform(**original_msg.metadata.as_kwargs, **self._metadata),
                    prev_msg_hash_key=prev_msg_hash_key,
                )
                forwarded_msg._original_msg = original_msg  # pylint: disable=protected-access
                return forwarded_msg

            return original_msg

        raise ValueError(f"Unexpected message content type: {type(self._content)}")

    async def _aget_previous_msg_promise_impl(self) -> Optional["MessagePromise"]:
        if self._materialized_msg:
            if self._materialized_msg.prev_msg_hash_key:
                return await self.forum.afind_message_promise(self._materialized_msg.prev_msg_hash_key)
            return None
        return self._branch_from

    # TODO TODO TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO TODO TODO

    async def aget_history(
        self, skip_agent_calls: bool = True, include_this_message: bool = True
    ) -> List["MessagePromise"]:
        """Get the full chat history of the conversation branch up to this message."""
        # TODO Oleksandr: introduce a limit on the number of messages to fetch
        msg = self
        result = [msg] if include_this_message else []
        while msg := await msg.aget_previous_message(skip_agent_calls=skip_agent_calls):
            result.append(msg)
        result.reverse()
        return result

    async def amaterialize_history(
        self, skip_agent_calls: bool = True, include_this_message: bool = True
    ) -> List[Message]:
        """
        Get the full chat history of the conversation branch up to this message, but return a list of Message objects
        instead of MessagePromise objects.
        """
        return [
            await msg.amaterialize()
            for msg in await self.aget_history(
                skip_agent_calls=skip_agent_calls, include_this_message=include_this_message
            )
        ]
