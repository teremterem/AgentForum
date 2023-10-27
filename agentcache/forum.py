"""
This module contains wrappers for the models defined in agentcache.models. These wrappers are used to add additional
functionality to the models without modifying the models themselves.
"""
import asyncio
import contextvars
from contextvars import ContextVar
from functools import wraps
from typing import Dict, Any, Optional, List, Union

from pydantic import BaseModel, ConfigDict

from agentcache.models import Message, Freeform, Token, AgentCall
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
        # TODO Oleksandr: are you sure about `wraps` ? Agents don't implement `__call__`
        return wraps(func)(Agent(self, func))

    async def anew_message(
        self,
        content: str,
        sender_alias: Optional[str] = None,
        reply_to: Union["MessagePromise", Message, str] = None,
        **metadata,
    ) -> "MessagePromise":
        """
        Create a MessagePromise object that represents a message and store the underlying Message in ImmutableStorage.
        """
        if isinstance(reply_to, MessagePromise):
            reply_to = await reply_to.aget_hash_key()
        elif isinstance(reply_to, Message):
            reply_to = reply_to.hash_key
        # if reply_to is a string, we assume it's already a hash key
        # TODO Oleksandr: assert somehow that reply_to is a valid hash key when it's a string

        message = Message(
            content=content,
            sender_alias=self.resolve_sender_alias(sender_alias),
            metadata=Freeform(**metadata),
            prev_msg_hash_key=reply_to,
        )
        await self.immutable_storage.astore_immutable(message)
        return MessagePromise(forum=self, full_message=message)

    async def afind_message(self, hash_key: str) -> "MessagePromise":
        """Find a message in the forum."""
        message = await self.immutable_storage.aretrieve_immutable(hash_key)
        if not isinstance(message, Message):
            # TODO Oleksandr: introduce a custom exception for this case ?
            raise ValueError(f"Expected a Message, got a {type(message)}")
        return MessagePromise(forum=self, full_message=message)

    async def _anew_agent_call(
        self,
        agent_alias: str,
        request: "MessagePromise",
        sender_alias: Optional[str] = None,
        **kwargs,
    ) -> "MessagePromise":
        """Create a MessagePromise object that represents a call to an agent (AgentCall)."""
        agent_call = AgentCall(
            content=agent_alias,
            sender_alias=self.resolve_sender_alias(sender_alias),
            metadata=Freeform(**kwargs),
            prev_msg_hash_key=await request.aget_hash_key(),
        )
        await self.immutable_storage.astore_immutable(agent_call)
        return MessagePromise(forum=self, full_message=agent_call)

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


class MessagePromise(Broadcastable[IN, Token]):
    """A message that is streamed token by token instead of being returned all at once."""

    def __init__(
        self, forum: Forum, full_message: Message = None, sender_alias: str = None, reply_to: "MessagePromise" = None
    ) -> None:
        if full_message and reply_to:
            raise ValueError("Only one of `full_message` and `reply_to` should be specified")
        if full_message and sender_alias:
            raise ValueError("Only one of `full_message` and `sender_alias` should be specified")

        super().__init__(
            items_so_far=[Token(text=full_message.content)] if full_message else None,
            completed=bool(full_message),
        )
        self.forum = forum
        self._full_message = full_message
        self._sender_alias = forum.resolve_sender_alias(sender_alias)
        self._reply_to = reply_to
        self._metadata: Dict[str, Any] = {}

    async def aget_full_message(self) -> Message:  # TODO Oleksandr: rename to amaterialize
        """
        Get the full message. This method will "await" until all the tokens are received and then return the complete
        message.
        """
        if not self._full_message:
            # TODO Oleksandr: offload most of this logic to the Forum class ?
            tokens = await self.aget_all()
            self._full_message = Message(
                content="".join([token.text for token in tokens]),
                sender_alias=self._sender_alias,
                metadata=Freeform(**self._metadata),  # TODO Oleksandr: create a separate function that does this ?
                prev_msg_hash_key=await self._reply_to.aget_hash_key() if self._reply_to else None,
            )
            await self.forum.immutable_storage.astore_immutable(self._full_message)
        return self._full_message

    async def aget_content(self) -> str:
        """Get the content of the full message."""
        return (await self.aget_full_message()).content

    @property
    def sender_alias(self) -> str:
        """Get the sender alias of the full message."""
        return self._full_message.sender_alias if self._full_message else self._sender_alias

    async def aget_metadata(self) -> Freeform:
        """Get the metadata of the full message."""
        return (await self.aget_full_message()).metadata

    async def aget_hash_key(self) -> str:
        """Get the hash key of the full message."""
        return (await self.aget_full_message()).hash_key

    async def aget_previous_message(self) -> Optional["MessagePromise"]:
        """Get the previous message in the conversation."""
        full_message = await self.aget_full_message()
        if not full_message.prev_msg_hash_key:
            return None

        if not hasattr(self, "_prev_msg"):
            # TODO Oleksandr: offload most of this logic to the Forum class ?
            prev_msg_hash_key = full_message.prev_msg_hash_key
            while isinstance(
                prev_msg := await self.forum.immutable_storage.aretrieve_immutable(prev_msg_hash_key), AgentCall
            ):
                # skip agent calls
                prev_msg_hash_key = prev_msg.request_hash_key
            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit,PyTypeChecker
            self._prev_msg = MessagePromise(forum=self.forum, full_message=prev_msg)
        return self._prev_msg

    async def aget_full_chat(self) -> List["MessagePromise"]:
        """Get the full chat history for this message (including this message)."""
        # TODO Oleksandr: introduce a limit on the number of messages to fetch
        msg = self
        result = [msg]
        while msg := await msg.aget_previous_message():
            result.append(msg)
        result.reverse()
        return result


class MessageSequence(Broadcastable[MessagePromise, MessagePromise]):
    """
    An asynchronous iterable over a sequence of messages that are being produced by an agent. Because the sequence is
    Broadcastable and relies on an internal async queue, the speed at which messages are produced and sent to the
    sequence is independent of the speed at which consumers iterate over them.
    """

    # TODO Oleksandr: throw an error if the sequence is being iterated over within the same agent that is producing it
    #  to prevent deadlocks

    async def aget_concluding_message(self, raise_if_none: bool = True) -> Optional[MessagePromise]:
        """Get the last message in the sequence."""
        messages = await self.aget_all()
        if messages:
            return messages[-1]
        if raise_if_none:
            # TODO Oleksandr: introduce a custom exception for this case
            raise ValueError("MessageSequence is empty")
        return None


class Agent:
    """A wrapper for an agent function that allows calling the agent."""

    def __init__(self, forum: Forum, func: AgentFunction) -> None:
        self._forum = forum
        self._func = func
        self.agent_alias = func.__name__

    def call(self, request: MessagePromise, **kwargs) -> MessageSequence:
        """Call the agent."""
        response = MessageSequence()
        asyncio.create_task(self._asubmit_agent_call(request, response, **kwargs))
        return response

    async def _asubmit_agent_call(self, request: MessagePromise, responses: MessageSequence, **kwargs) -> None:
        # noinspection PyProtectedMember
        agent_call = await request.forum._anew_agent_call(  # pylint: disable=protected-access
            agent_alias=self.agent_alias,
            request=request,
            **kwargs,
        )
        await self._acall_agent_func(agent_call, responses)

    async def _acall_agent_func(self, agent_call: MessagePromise, responses: MessageSequence) -> None:
        request = await agent_call.aget_previous_message()
        with responses, AgentContext(agent_alias=self.agent_alias):
            await self._func(request, responses, **(await agent_call.aget_metadata()).as_kwargs)


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
