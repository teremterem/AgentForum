"""
This module contains wrappers for the models defined in agentcache.models. These wrappers are used to add additional
functionality to the models without modifying the models themselves.
"""
import asyncio
import contextvars
from abc import abstractmethod, ABC
from contextvars import ContextVar
from typing import Dict, Any, Optional, List, Type, AsyncIterator

from pydantic import BaseModel, ConfigDict

from agentcache.models import Message, Freeform, Token, AgentCall, ForwardedMessage
from agentcache.storage import ImmutableStorage
from agentcache.typing import IN, AgentFunction
from agentcache.utils import Broadcastable

DEFAULT_AGENT_ALIAS = "USER"


class Forum(BaseModel):
    """A forum for agents to communicate. Messages in the forum assemble in a tree-like structure."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    immutable_storage: ImmutableStorage

    def agent(self, func: AgentFunction) -> "Agent":
        """A decorator that registers an agent function in the forum."""
        return Agent(self, func)

    async def afind_message_promise(self, hash_key: str) -> "MessagePromise":
        """Find a message in the forum."""
        message = await self.immutable_storage.aretrieve_immutable(hash_key)
        if not isinstance(message, Message):
            # TODO Oleksandr: introduce a custom exception for this case ?
            raise ValueError(f"Expected a Message, got a {type(message)}")
        return MessagePromise(forum=self, materialized_msg=message)

    def new_message_promise(
        self,
        content: str,
        sender_alias: Optional[str] = None,
        in_reply_to: Optional["MessagePromise"] = None,
        **metadata,
    ) -> "MessagePromise":
        """
        Create a new, detached message promise in the forum. "Detached message promise" means that this message
        promise is a reply to another message promise that may or may not be "materialized" yet.
        """
        return MessagePromise(
            forum=self,
            in_reply_to=in_reply_to,
            detached_msg=Message(
                content=content,
                sender_alias=self.resolve_sender_alias(sender_alias),
                metadata=Freeform(**metadata),
            ),
        )

    @staticmethod
    def resolve_sender_alias(sender_alias: Optional[str]) -> str:
        """
        Resolve the sender alias to use in a message. If sender_alias is not None, it is returned. Otherwise, the
        current AgentContext is used to get the agent alias, and if there is no current AgentContext, then
        DEFAULT_AGENT_ALIAS (which translates to "USER") is used.
        """
        if not sender_alias:
            agent_context = AgentContext.get_current_context()
            if agent_context:
                sender_alias = agent_context.agent_alias
        return sender_alias or DEFAULT_AGENT_ALIAS


class Agent:
    """A wrapper for an agent function that allows calling the agent."""

    def __init__(self, forum: Forum, func: AgentFunction) -> None:
        self.forum = forum
        self.agent_alias = func.__name__
        self._func = func

    def call(self, request: "MessagePromise", sender_alias: Optional[str] = None, **kwargs) -> "MessageSequence":
        """Call the agent."""
        agent_call = MessagePromise(
            forum=self.forum,
            in_reply_to=request,
            detached_msg=AgentCall(
                content=self.agent_alias,  # the recipient of the call is this agent
                sender_alias=self.forum.resolve_sender_alias(sender_alias),
                metadata=Freeform(**kwargs),
            ),
        )
        responses = MessageSequence(self.forum, in_reply_to=agent_call)
        asyncio.create_task(self._acall_agent_func(agent_call=agent_call, responses=responses, **kwargs))
        return responses

    async def _acall_agent_func(self, agent_call: "MessagePromise", responses: "MessageSequence", **kwargs) -> None:
        with responses:
            try:
                request = await agent_call.aget_previous_message()
                with AgentContext(agent_alias=self.agent_alias):
                    await self._func(request, responses, **kwargs)
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                # catch all exceptions, including KeyboardInterrupt
                responses.send(exc)  # TODO Oleksandr: introduce ErrorMessage


class AgentContext:
    """
    A context within which an agent is called. This is needed for things like looking up a sender alias for a message
    that is being created by the agent, so it can be populated in the message automatically (and other similar things).
    """

    _current_context: ContextVar[Optional["AgentContext"]] = ContextVar("_current_context", default=None)

    def __init__(self, agent_alias: str):
        self.agent_alias = agent_alias
        self._previous_ctx_token: Optional[contextvars.Token] = None

    @classmethod
    def get_current_context(cls) -> Optional["AgentContext"]:
        """Get the current AgentContext object."""
        return cls._current_context.get()

    def __enter__(self) -> "AgentContext":
        """Set this context as the current context."""
        if self._previous_ctx_token:
            raise RuntimeError("AgentContext is not reentrant")
        self._previous_ctx_token = self._current_context.set(self)  # <- this is the context switch
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore the context that was current before this one."""
        self._current_context.reset(self._previous_ctx_token)
        self._previous_ctx_token = None


class MessagePromise(Broadcastable[IN, Token], ABC):
    def __init__(self, forum: Forum, materialized_msg: Optional[Message] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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

    @abstractmethod
    async def _amaterialize(self) -> Message:
        """Non-cached part of amaterialize()."""

    async def _aget_previous_message(self) -> Optional["MessagePromise"]:
        """Non-cached part of aget_previous_message()."""
        msg = await self.amaterialize()
        if msg.prev_msg_hash_key:
            return await self.forum.afind_message_promise(msg.prev_msg_hash_key)
        return None

    async def _aget_original_message(self) -> Optional["MessagePromise"]:
        """Non-cached part of aget_original_message()."""
        if isinstance(self.real_msg_class, ForwardedMessage):
            return await self.forum.afind_message_promise(self.amaterialize().original_msg_hash_key)
        return None


class StreamedMsgPromise(MessagePromise):
    """A message that is streamed token by token instead of being returned all at once."""

    def __init__(
        self,
        forum: Forum,
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
    This is a detached message promise. A detached is on one hand is complete, but on the other hand doesn't reference
    the previous message in the conversation yet (neither it references its original message, in case it's a forward).
    This is why in_reply_to and a_forward_of are not specified as standalone properties in the promise constructor -
    those relation will become part of the underlying Message upon its "materialization".
    """

    def __init__(
        self,
        forum: Forum,
        detached_msg: Message,
        in_reply_to: Optional["MessagePromise"] = None,
        a_forward_of: Optional["MessagePromise"] = None,
    ) -> None:
        super().__init__(forum=forum, items_so_far=[Token(text=detached_msg.content)], completed=True)
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

        return self._materialized_msg

    def _foresee_real_msg_class(self) -> Type[Message]:
        if self._a_forward_of:
            return ForwardedMessage
        return type(self._detached_msg)

    async def _aget_previous_message(self) -> Optional["MessagePromise"]:
        return self._in_reply_to

    async def _aget_original_message(self) -> Optional["MessagePromise"]:
        return self._a_forward_of


class MessageSequence(Broadcastable[MessagePromise, MessagePromise]):
    """
    An asynchronous iterable over a sequence of messages that are being produced by an agent. Because the sequence is
    Broadcastable and relies on an internal async queue, the speed at which messages are produced and sent to the
    sequence is independent of the speed at which consumers iterate over them.
    """

    # TODO Oleksandr: throw an error if the sequence is being iterated over within the same agent that is producing it
    #  to prevent deadlocks

    def __init__(
        self,
        forum: Forum,
        in_reply_to: Optional["MessagePromise"] = None,
    ) -> None:
        super().__init__()
        self.forum = forum
        self._in_reply_to = in_reply_to

    def send(self, content: str, sender_alias: Optional[str] = None, **metadata) -> None:
        """Send a message to the end of a sequence."""
        if isinstance(content, BaseException):
            # TODO Oleksandr: replace with ErrorMessage when it is introduced
            super().send(content)
            return

        if isinstance(content, MessagePromise):
            # TODO Oleksandr: update method signature to support this (or create a separate method ?)
            # TODO Oleksandr: turn this into a forum method akin to forum.anew_message_promise() ?
            msg_promise = MessagePromise(
                forum=self.forum,
                in_reply_to=self._in_reply_to,
                a_forward_of=content,
                detached_msg=Message(
                    content="",  # TODO Oleksandr: get rid of this hack
                    sender_alias=self.forum.resolve_sender_alias(sender_alias),
                    metadata=Freeform(**metadata),
                ),
            )
        else:
            msg_promise = self.forum.new_message_promise(
                content=content, sender_alias=sender_alias, in_reply_to=self._in_reply_to, **metadata
            )

        self._in_reply_to = msg_promise
        super().send(msg_promise)

    async def aget_concluding_message(self, raise_if_none: bool = True) -> Optional[MessagePromise]:
        """Get the last message in the sequence."""
        messages = await self.aget_all()
        if messages:
            return messages[-1]
        if raise_if_none:
            # TODO Oleksandr: introduce a custom exception for this case
            raise ValueError("MessageSequence is empty")
        return None
