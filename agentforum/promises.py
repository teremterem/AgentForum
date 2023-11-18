"""This module contains wrappers for the pydantic models that turn those models into asynchronous promises."""
import asyncio
import typing
from typing import Optional, Type, List, Dict, Any, AsyncIterator, Union

from agentforum._internals.internal_promises import MessagePromiseImpl
from agentforum.models import Message, AgentCallMsg, ForwardedMessage, Freeform, MessageParameters, ContentChunk
from agentforum.typing import IN, MessageType, SingleMessageType
from agentforum.utils import AsyncStreamable, async_cached_method

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
    # pylint: disable=protected-access
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
                    self._prev_msg_promise = None
                    self._metadata = None

        return self._materialized_msg

    async def _amaterialize_impl(self) -> Message:
        # TODO Oleksandr: get rid of code duplications within this method as much as possible
        if isinstance(self._content, str):
            return Message(
                content=self._content,
                sender_alias=self._override_sender_alias or self._default_sender_alias,
                metadata=Freeform(**self._metadata),
                prev_msg_hash_key=(await self._branch_from.amaterialize()).hash_key if self._branch_from else None,
            )
        elif isinstance(self._content, StreamedMessage):
            return Message(
                content="".join([token.text async for token in self._content]),
                sender_alias=self._override_sender_alias or self._default_sender_alias,
                metadata=Freeform(**self._metadata),
                prev_msg_hash_key=(await self._branch_from.amaterialize()).hash_key if self._branch_from else None,
            )

        elif isinstance(self._content, (Message, MessagePromise)):
            if isinstance(self._content, MessagePromise):
                original_msg = await self._content.amaterialize()
            else:
                original_msg = self._content

            should_be_forwarded = True
            if self._do_not_forward_if_possible and not self._metadata:
                prev_msg_hash_key = (await self._branch_from.amaterialize()).hash_key if self._branch_from else None
                if prev_msg_hash_key == original_msg.prev_msg_hash_key:
                    should_be_forwarded = False

            if should_be_forwarded:
                forwarded_msg = ForwardedMessage(
                    original_msg_hash_key=original_msg.hash_key,
                    sender_alias=self._override_sender_alias or self._default_sender_alias,
                    metadata=Freeform(**self._metadata),
                    prev_msg_hash_key=(await self._branch_from.amaterialize()).hash_key if self._branch_from else None,
                )
                forwarded_msg._original_msg = original_msg  # pylint: disable=protected-access
                return forwarded_msg

            return original_msg

    # TODO TODO TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO TODO TODO

    def _materialize_msg_promise_impl(self) -> MessagePromiseImpl:
        if isinstance(content, str):
            forward_of = None
        elif isinstance(content, Message):
            forward_of = MessagePromise(forum=self.forum, materialized_msg=content)
            # TODO Oleksandr: should we store the materialized_msg ?
            #  (the promise will not store it since it is already "materialized")
            #  or do we trust that something else already stored it ?
            content = ""  # this is a hack (the content will actually be taken from the forwarded message)
        elif isinstance(content, MessagePromise):
            forward_of = content
            content = ""  # this is a hack (the content will actually be taken from the forwarded message)
        else:
            raise ValueError(f"Unexpected message content type: {type(content)}")

        msg_promise = DetachedMsgPromise(
            forum=self.forum,
            branch_from=self._latest_msg_promise,
            forward_of=forward_of,
            detached_msg=Message(
                content=content,
                sender_alias=sender_alias,
                metadata=Freeform(**metadata),
            ),
        )
        self._latest_msg_promise = msg_promise
        return msg_promise

    @property
    def real_msg_class(self) -> Type[Message]:
        """Return the type of the real message that this promise represents."""
        if self._materialized_msg:
            return type(self._materialized_msg)
        return self._foresee_real_msg_class()

    async def aget_previous_message(self, skip_agent_calls: bool = True) -> Optional["MessagePromise"]:
        """Get the previous message in this conversation branch."""
        prev_msg = await self._aget_previous_message_cached()

        if skip_agent_calls:
            while prev_msg:
                if not issubclass(prev_msg.real_msg_class, AgentCallMsg):
                    break
                prev_msg = await prev_msg._aget_previous_message_cached()

        return prev_msg

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

    async def aget_original_message(self, return_self_if_none: bool = True) -> Optional["MessagePromise"]:
        """
        Get the original message for this forwarded message. Return self or None if the original message is not found
        (depending on whether return_self_if_none is True or False).
        """
        original_msg = self._aget_original_message_cached()
        if return_self_if_none:
            original_msg = original_msg or self
        return original_msg

    @async_cached_method
    async def _aget_previous_message_cached(self) -> Optional["MessagePromise"]:
        return await self._aget_previous_message_impl()

    @async_cached_method
    async def _aget_original_message_cached(self) -> Optional["MessagePromise"]:
        return await self._aget_original_message_impl()

    # noinspection PyMethodMayBeStatic
    def _foresee_real_msg_class(self) -> Type[Message]:
        """This method foresees what the real message type will be when it is "materialized"."""
        return Message

    async def _aget_previous_message_impl(self) -> Optional["MessagePromise"]:
        msg = await self.amaterialize()
        if msg.prev_msg_hash_key:
            return await self.forum.afind_message_promise(msg.prev_msg_hash_key)
        return None

    async def _aget_original_message_impl(self) -> Optional["MessagePromise"]:
        if isinstance(self.real_msg_class, ForwardedMessage):
            # noinspection PyUnresolvedReferences
            return await self.forum.afind_message_promise(self.amaterialize().original_msg_hash_key)
        return None
