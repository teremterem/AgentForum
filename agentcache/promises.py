"""This module contains wrappers for the pydantic models that turn those models into asynchronous promises."""
import typing
from typing import Optional, Type, List, Dict, Any, AsyncIterator

from agentcache.models import Token, Message, AgentCall, ForwardedMessage, Freeform
from agentcache.typing import IN
from agentcache.utils import Broadcastable

if typing.TYPE_CHECKING:
    from agentcache.forum import Forum


class MessageSequence(Broadcastable["MessagePromise", "MessagePromise"]):
    """
    An asynchronous iterable over a sequence of messages that are being produced by an agent. Because the sequence is
    Broadcastable and relies on an internal async queue, the speed at which messages are produced and sent to the
    sequence is independent of the speed at which consumers iterate over them.
    """

    # TODO Oleksandr: throw an error if the sequence is being iterated over within the same agent that is producing it
    #  to prevent deadlocks

    def __init__(
        self,
        forum: "Forum",
        in_reply_to: Optional["MessagePromise"] = None,
    ) -> None:
        super().__init__()
        self.forum = forum
        self._in_reply_to = in_reply_to

    async def aget_concluding_message(self, raise_if_none: bool = True) -> Optional["MessagePromise"]:
        """Get the last message in the sequence."""
        messages = await self.aget_all()
        if messages:
            return messages[-1]
        if raise_if_none:
            # TODO Oleksandr: introduce a custom exception for this case
            raise ValueError("MessageSequence is empty")
        return None


class MessagePromise(Broadcastable[IN, Token]):
    """A promise to materialize a message."""

    def __init__(
        self,
        forum: "Forum",
        materialized_msg: Optional[Message] = None,
        materialized_msg_content: Optional[str] = None,
    ) -> None:
        """
        TODO Oleksandr: better docstring ?
        NOTE: The `materialized_msg_content` parameter is used to initialize the MessagePromise with content from a
        "detached" message (see DetachedMsgPromise class).
        """
        if materialized_msg:
            # if there is a materialized message, then override whatever was passed in as materialized_msg_content -
            # materialized_msg.content is the source of truth
            materialized_msg_content = materialized_msg.content

        super().__init__(
            items_so_far=None if materialized_msg_content is None else [Token(text=materialized_msg_content)],
            completed=materialized_msg_content is not None,
        )
        self.forum = forum
        self._materialized_msg = materialized_msg

    @property
    def real_msg_class(self) -> Type[Message]:
        """Return the type of the real message that this promise represents."""
        if self._materialized_msg:
            return type(self._materialized_msg)
        return self._foresee_real_msg_class()

    async def amaterialize(self) -> Message:
        """
        Get the full message. This method will "await" until all the tokens are received (or whatever else needs to be
        waited for before the actual message can be constructed and stored in the storage) and then return the message.
        """
        if not self._materialized_msg:
            self._materialized_msg = await self._amaterialize()
            await self.forum.immutable_storage.astore_immutable(self._materialized_msg)

        return self._materialized_msg

    async def aget_previous_message(self, skip_agent_calls: bool = True) -> Optional["MessagePromise"]:
        """Get the previous message in this conversation branch."""
        if not hasattr(self, "_prev_msg"):
            prev_msg = await self._aget_previous_message()

            if skip_agent_calls:
                while prev_msg:
                    if not issubclass(prev_msg.real_msg_class, AgentCall):
                        break
                    # noinspection PyUnresolvedReferences
                    prev_msg = await prev_msg._aget_previous_message()  # pylint: disable=protected-access

            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self._prev_msg = prev_msg
        return self._prev_msg

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
        if not hasattr(self, "_original_msg"):
            original_msg = await self._aget_original_message()
            if return_self_if_none:
                original_msg = original_msg or self

            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self._original_msg = original_msg
        return self._original_msg

    # noinspection PyMethodMayBeStatic
    def _foresee_real_msg_class(self) -> Type[Message]:
        """This method foresees what the real message type will be when it is "materialized"."""
        return Message

    async def _amaterialize(self) -> Message:
        """Non-cached part of amaterialize()."""
        raise NotImplementedError(
            "Either create a MessagePromise that is materialized from the start or use a subclass that implements "
            "this method."
        )

    async def _aget_previous_message(self) -> Optional["MessagePromise"]:
        """Non-cached part of aget_previous_message()."""
        msg = await self.amaterialize()
        if msg.prev_msg_hash_key:
            return await self.forum.afind_message_promise(msg.prev_msg_hash_key)
        return None

    async def _aget_original_message(self) -> Optional["MessagePromise"]:
        """Non-cached part of aget_original_message()."""
        if isinstance(self.real_msg_class, ForwardedMessage):
            # noinspection PyUnresolvedReferences
            return await self.forum.afind_message_promise(self.amaterialize().original_msg_hash_key)
        return None


class StreamedMsgPromise(MessagePromise):
    """A message that is streamed token by token instead of being returned all at once."""

    def __init__(
        self,
        forum: "Forum",
        sender_alias: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        in_reply_to: Optional[MessagePromise] = None,
    ) -> None:
        super().__init__(forum=forum)
        self._sender_alias = sender_alias
        self._metadata = dict(metadata or {})
        self._in_reply_to = in_reply_to

    async def _amaterialize(self) -> Message:
        return Message(
            content="".join([token.text async for token in self]),
            sender_alias=self._sender_alias,
            metadata=Freeform(**self._metadata),
            prev_msg_hash_key=(await self._in_reply_to.amaterialize()).hash_key if self._in_reply_to else None,
        )

    async def _aget_previous_message(self) -> Optional["MessagePromise"]:
        return self._in_reply_to

    async def _aget_original_message(self) -> Optional["MessagePromise"]:
        return None


class DetachedMsgPromise(MessagePromise):
    """
    This is a detached message promise. A detached message is on one hand is complete, but on the other hand doesn't
    reference the previous message in the conversation yet (neither it references its original message, in case it's
    a forward). This is why in_reply_to and a_forward_of are not specified as standalone properties in the promise
    constructor - those relation will become part of the underlying Message upon its "materialization".
    """

    def __init__(
        self,
        forum: "Forum",
        detached_msg: Message,
        in_reply_to: Optional["MessagePromise"] = None,
        a_forward_of: Optional["MessagePromise"] = None,
    ) -> None:
        super().__init__(forum=forum, materialized_msg_content=detached_msg.content)
        self.forum = forum
        self._detached_msg = detached_msg
        self._in_reply_to = in_reply_to
        self._a_forward_of = a_forward_of

    def __aiter__(self) -> AsyncIterator[Token]:
        if self._a_forward_of:
            return self._a_forward_of.__aiter__()
        return super().__aiter__()

    @property
    def completed(self) -> bool:
        if self._a_forward_of:
            return self._a_forward_of.completed
        return super().completed

    async def _amaterialize(self) -> Message:
        """
        Get the full message. This method will "await" until all the tokens are received and then return the complete
        message.
        """
        prev_msg_hash_key = (await self._in_reply_to.amaterialize()).hash_key if self._in_reply_to else None
        original_msg = None

        if self._a_forward_of:
            original_msg = await self._a_forward_of.amaterialize()

            metadata_dict = original_msg.metadata.model_dump(exclude={"ac_model_"})
            # let's merge the metadata from the original message with the metadata from the detached message
            # (detached message metadata overrides the original message metadata in case of conflicts; also
            # keep in mind that it is a shallow merge - nested objects are not merged)
            metadata_dict.update(self._detached_msg.metadata.model_dump(exclude={"ac_model_"}))

            content = original_msg.content
            metadata = Freeform(**metadata_dict)
            extra_kwargs = {"original_msg_hash_key": original_msg.hash_key}
        else:
            content = self._detached_msg.content
            metadata = self._detached_msg.metadata
            extra_kwargs = {}

        self._materialized_msg = self.real_msg_class(
            **self._detached_msg.model_dump(
                exclude={"ac_model_", "content", "metadata", "prev_msg_hash_key", "original_msg_hash_key"}
            ),
            content=content,
            metadata=metadata,
            prev_msg_hash_key=prev_msg_hash_key,
            **extra_kwargs,
        )

        self._materialized_msg._original_msg = original_msg  # pylint: disable=protected-access
        return self._materialized_msg

    def _foresee_real_msg_class(self) -> Type[Message]:
        if self._a_forward_of:
            return ForwardedMessage
        return type(self._detached_msg)

    async def _aget_previous_message(self) -> Optional["MessagePromise"]:
        return self._in_reply_to

    async def _aget_original_message(self) -> Optional["MessagePromise"]:
        return self._a_forward_of
