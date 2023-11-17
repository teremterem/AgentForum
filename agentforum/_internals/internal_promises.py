"""
TODO Oleksandr: the classes defined in this module are temporary, merge their function into other related classes.
"""
import typing
from typing import Optional, Type, List, Dict, Any, AsyncIterator

from agentforum.models import Token, Message, AgentCallMsg, ForwardedMessage, Freeform
from agentforum.typing import IN
from agentforum.utils import AsyncStreamable, async_cached_method

if typing.TYPE_CHECKING:
    from agentforum.forum import Forum
    from agentforum.promises import MessageSequence


class MessagePromiseImpl(AsyncStreamable[IN, Token]):
    # pylint: disable=protected-access
    """A promise to materialize a message."""

    def __init__(
        self,
        forum: "Forum",
        materialized_msg: Optional[Message] = None,
        materialized_msg_content: Optional[str] = None,
    ) -> None:
        """
        TODO Oleksandr: better docstring ?
        NOTE: The `materialized_msg_content` parameter is used to initialize the MessagePromiseImpl with content from a
        "detached" message (see DetachedMsgPromiseImpl class).
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
        if not self._materialized_msg:  # TODO Oleksandr: apply asyncio.Lock
            self._materialized_msg = await self._amaterialize_impl()
            await self.forum.immutable_storage.astore_immutable(self._materialized_msg)

        return self._materialized_msg

    async def aget_previous_message(self, skip_agent_calls: bool = True) -> Optional["MessagePromiseImpl"]:
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
    ) -> List["MessagePromiseImpl"]:
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
        instead of MessagePromiseImpl objects.
        """
        return [
            await msg.amaterialize()
            for msg in await self.aget_history(
                skip_agent_calls=skip_agent_calls, include_this_message=include_this_message
            )
        ]

    async def aget_original_message(self, return_self_if_none: bool = True) -> Optional["MessagePromiseImpl"]:
        """
        Get the original message for this forwarded message. Return self or None if the original message is not found
        (depending on whether return_self_if_none is True or False).
        """
        original_msg = self._aget_original_message_cached()
        if return_self_if_none:
            original_msg = original_msg or self
        return original_msg

    @async_cached_method
    async def _aget_previous_message_cached(self) -> Optional["MessagePromiseImpl"]:
        return await self._aget_previous_message_impl()

    @async_cached_method
    async def _aget_original_message_cached(self) -> Optional["MessagePromiseImpl"]:
        return await self._aget_original_message_impl()

    # noinspection PyMethodMayBeStatic
    def _foresee_real_msg_class(self) -> Type[Message]:
        """This method foresees what the real message type will be when it is "materialized"."""
        return Message

    async def _amaterialize_impl(self) -> Message:
        raise NotImplementedError(
            "Either create a MessagePromiseImpl that is materialized from the start or use a subclass that implements "
            "this method."
        )

    async def _aget_previous_message_impl(self) -> Optional["MessagePromiseImpl"]:
        msg = await self.amaterialize()
        if msg.prev_msg_hash_key:
            return await self.forum.afind_message_promise(msg.prev_msg_hash_key)
        return None

    async def _aget_original_message_impl(self) -> Optional["MessagePromiseImpl"]:
        if isinstance(self.real_msg_class, ForwardedMessage):
            # noinspection PyUnresolvedReferences
            return await self.forum.afind_message_promise(self.amaterialize().original_msg_hash_key)
        return None


class StreamedMsgPromiseImpl(MessagePromiseImpl[IN]):
    """A message that is streamed token by token instead of being returned all at once."""

    def __init__(
        self,
        forum: "Forum",
        sender_alias: str,
        metadata: Optional[Dict[str, Any]] = None,
        branch_from: Optional[MessagePromiseImpl] = None,
    ) -> None:
        super().__init__(forum=forum)
        self._sender_alias = sender_alias
        self._metadata = dict(metadata or {})
        self._branch_from = branch_from

    async def _amaterialize_impl(self) -> Message:
        return Message(
            content="".join([token.text async for token in self]),
            sender_alias=self._sender_alias,
            metadata=Freeform(**self._metadata),
            prev_msg_hash_key=(await self._branch_from.amaterialize()).hash_key if self._branch_from else None,
        )

    async def _aget_previous_message_impl(self) -> Optional["MessagePromiseImpl"]:
        return self._branch_from

    async def _aget_original_message_impl(self) -> Optional["MessagePromiseImpl"]:
        return None


class DetachedMsgPromiseImpl(MessagePromiseImpl):
    # pylint: disable=protected-access
    """
    This is a detached message promise. A detached message is on one hand complete, but on the other hand doesn't
    reference the previous message in the conversation yet (neither it references its original message, in case it's
    a forward). This is why branch_from and forward_of are specified as standalone properties in the promise
    constructor - those relations will become part of the underlying Message upon its "materialization".
    """

    def __init__(
        self,
        forum: "Forum",
        detached_msg: Message,
        branch_from: Optional[MessagePromiseImpl] = None,
        forward_of: Optional[MessagePromiseImpl] = None,
    ) -> None:
        super().__init__(forum=forum, materialized_msg_content=detached_msg.content)
        self.forum = forum
        self._detached_msg = detached_msg
        self._branch_from = branch_from
        self._forward_of = forward_of

    def __aiter__(self) -> AsyncIterator[Token]:
        if self._forward_of:
            return self._forward_of.__aiter__()
        return super().__aiter__()

    @property
    def completed(self) -> bool:
        if self._forward_of:
            return self._forward_of.completed
        return super().completed

    async def _amaterialize_impl(self) -> Message:
        """
        Get the full message. This method will "await" until all the tokens are received and then return the complete
        message.
        """
        prev_msg_hash_key = (await self._branch_from.amaterialize()).hash_key if self._branch_from else None
        original_msg = None

        if self._forward_of:
            original_msg = await self._forward_of.amaterialize()

            metadata_dict = original_msg.metadata.model_dump(exclude={"af_model_"})
            # let's merge the metadata from the original message with the metadata from the detached message
            # (detached message metadata overrides the original message metadata in case of conflicts; also
            # keep in mind that it is a shallow merge - nested objects are not merged)
            metadata_dict.update(self._detached_msg.metadata.model_dump(exclude={"af_model_"}))

            content = original_msg.content
            metadata = Freeform(**metadata_dict)
            extra_kwargs = {"original_msg_hash_key": original_msg.hash_key}
        else:
            content = self._detached_msg.content
            metadata = self._detached_msg.metadata
            extra_kwargs = {}

        materialized_msg = self.real_msg_class(
            **self._detached_msg.model_dump(
                exclude={"af_model_", "content", "metadata", "prev_msg_hash_key", "original_msg_hash_key"}
            ),
            content=content,
            metadata=metadata,
            prev_msg_hash_key=prev_msg_hash_key,
            **extra_kwargs,
        )

        if original_msg:
            materialized_msg._original_msg = original_msg
        return materialized_msg

    def _foresee_real_msg_class(self) -> Type[Message]:
        if self._forward_of:
            return ForwardedMessage
        return type(self._detached_msg)

    async def _aget_previous_message_impl(self) -> Optional[MessagePromiseImpl]:
        return self._branch_from

    async def _aget_original_message_impl(self) -> Optional[MessagePromiseImpl]:
        return self._forward_of


class DetachedAgentCallMsgPromiseImpl(MessagePromiseImpl):
    """
    DetachedAgentCallMsgPromiseImpl is a subtype of MessagePromiseImpl that represents a promise that can be
    materialized into an AgentCallMsg which, in turn, is a subtype of Message that represents a call to an agent.
    """

    def __init__(
        self,
        forum: "Forum",
        request_messages: "MessageSequence",
        detached_agent_call_msg: AgentCallMsg,
    ) -> None:
        super().__init__(forum=forum, materialized_msg_content=detached_agent_call_msg.content)
        self.forum = forum
        self._request_messages = request_messages
        self._detached_agent_call_msg = detached_agent_call_msg

    async def _amaterialize_impl(self) -> Message:
        messages = await self._request_messages.amaterialize_all()
        if messages:
            msg_seq_start_hash_key = messages[0].hash_key
            msg_seq_end_hash_key = messages[-1].hash_key
        else:
            msg_seq_start_hash_key = None
            msg_seq_end_hash_key = None

        return self.real_msg_class(
            **self._detached_agent_call_msg.model_dump(
                exclude={"af_model_", "metadata", "prev_msg_hash_key", "msg_seq_start_hash_key"}
            ),
            metadata=self._detached_agent_call_msg.metadata,
            prev_msg_hash_key=msg_seq_end_hash_key,  # agent calls get attached to the end of the message sequence
            msg_seq_start_hash_key=msg_seq_start_hash_key,
        )

    def _foresee_real_msg_class(self) -> Type[Message]:
        return type(self._detached_agent_call_msg)

    async def _aget_previous_message_impl(self) -> Optional[MessagePromiseImpl]:
        msg_promises = [msg_promise async for msg_promise in self._request_messages]
        return msg_promises[-1] if msg_promises else None
