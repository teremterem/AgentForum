"""
This module contains wrappers for the models defined in agentcache.models. These wrappers are used to add additional
functionality to the models without modifying the models themselves.
"""
import asyncio
import contextvars
from contextvars import ContextVar
from typing import Dict, Any, Optional, List, Type

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
        return Agent(self, func)

    async def anew_message(
        self,
        content: str,
        sender_alias: Optional[str] = None,
        in_reply_to: Optional["MessagePromise"] = None,
        **metadata,
    ) -> "MessagePromise":
        """Create a new message in the forum."""
        return MessagePromise(
            forum=self,
            in_reply_to=in_reply_to,
            detached_msg=Message(
                content=content,
                sender_alias=self.resolve_sender_alias(sender_alias),
                metadata=Freeform(**metadata),
            ),
        )

    async def afind_message(self, hash_key: str) -> "MessagePromise":
        """Find a message in the forum."""
        message = await self.immutable_storage.aretrieve_immutable(hash_key)
        if not isinstance(message, Message):
            # TODO Oleksandr: introduce a custom exception for this case ?
            raise ValueError(f"Expected a Message, got a {type(message)}")
        return MessagePromise(forum=self, materialized_msg=message)

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

    # TODO Oleksandr: split all the logic in this class into three subclasses:
    #  StreamedMsgPromise, DetachedMsgPromise and MaterializedMsgPromise

    def __init__(  # pylint: disable=too-many-arguments
        self,
        forum: Forum,
        sender_alias: Optional[str] = None,
        in_reply_to: Optional["MessagePromise"] = None,
        materialized_msg: Optional[Message] = None,
        detached_msg: Optional[Message] = None,
    ) -> None:
        if materialized_msg and detached_msg:
            raise ValueError("materialized_msg and detached_msg cannot be specified at the same time")
        if materialized_msg and (sender_alias or in_reply_to):
            raise ValueError("If materialized_msg is specified, sender_alias and in_reply_to must be None")
        if detached_msg and sender_alias:
            raise ValueError("If detached_msg is specified, sender_alias must be None")
        if not (materialized_msg or detached_msg) and not sender_alias:
            raise ValueError("sender_alias must be specified if neither materialized_msg nor detached_msg is given")

        msg_content = None
        if materialized_msg:
            msg_content = materialized_msg.content
        elif detached_msg:
            msg_content = detached_msg.content
        super().__init__(
            items_so_far=[Token(text=msg_content)] if msg_content else None,
            completed=bool(materialized_msg or detached_msg),
        )
        self.forum = forum
        self._sender_alias = sender_alias
        self._in_reply_to = in_reply_to
        self._materialized_msg = materialized_msg
        self._detached_msg = detached_msg
        self._metadata: Dict[str, Any] = {}

    async def amaterialize(self) -> Message:
        """
        Get the full message. This method will "await" until all the tokens are received and then return the complete
        message.
        """
        if not self._materialized_msg:
            prev_msg_hash_key = (await self._in_reply_to.amaterialize()).hash_key if self._in_reply_to else None

            if self._detached_msg:
                # This is a detached message - a message that on one hand is complete, but on the other hand doesn't
                # reference the previous message in the conversation yet. This is why we are cloning it here and
                # adding the prev_msg_hash_key value to the clone.
                self._materialized_msg = self._detached_msg.model_copy(update={"prev_msg_hash_key": prev_msg_hash_key})
            else:
                # It is a streamed message, so we need to assemble it from the tokens.
                tokens = await self.aget_all()
                self._materialized_msg = Message(
                    content="".join([token.text for token in tokens]),
                    sender_alias=self._sender_alias,
                    metadata=Freeform(**self._metadata),
                    prev_msg_hash_key=prev_msg_hash_key,
                )
            await self.forum.immutable_storage.astore_immutable(self._materialized_msg)

        return self._materialized_msg

    def _real_msg_class(self) -> Type[Message]:
        if self._materialized_msg:
            return type(self._materialized_msg)
        if self._detached_msg:
            return type(self._detached_msg)
        return Message

    async def _aget_previous_message(self) -> Optional["MessagePromise"]:
        if self._materialized_msg:
            if self._materialized_msg.prev_msg_hash_key:
                return await self.forum.afind_message(self._materialized_msg.prev_msg_hash_key)
            return None
        return self._in_reply_to  # this is the source of truth in case of detached and streamed messages

    async def aget_previous_message(self, skip_agent_calls: bool = True) -> Optional["MessagePromise"]:
        """Get the previous message in the conversation."""
        if not hasattr(self, "_prev_msg"):
            prev_msg = await self._aget_previous_message()

            if skip_agent_calls:
                while prev_msg:
                    if not issubclass(type(prev_msg), AgentCall):
                        break
                    prev_msg = await prev_msg._aget_previous_message()  # pylint: disable=protected-access

            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self._prev_msg = prev_msg
        return self._prev_msg

    async def aget_history(self, skip_agent_calls: bool = True, include_this_message=True) -> List["MessagePromise"]:
        """Get the full chat history for this message."""
        # TODO Oleksandr: introduce a limit on the number of messages to fetch
        msg = self
        result = [msg] if include_this_message else []
        while msg := await msg.aget_previous_message(skip_agent_calls=skip_agent_calls):
            result.append(msg)
        result.reverse()
        return result

    async def amaterialize_history(self, skip_agent_calls: bool = True) -> List[Message]:
        """
        Get the full chat history of this message, but return a list Message objects instead of MessagePromise objects.
        """
        return [await msg.amaterialize() for msg in await self.aget_history(skip_agent_calls=skip_agent_calls)]


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

    def call(self, request: MessagePromise, sender_alias: Optional[str] = None, **kwargs) -> MessageSequence:
        """Call the agent."""
        sender_alias = self._forum.resolve_sender_alias(sender_alias)
        # TODO Oleksandr: make sure that responses are attached to the AgentCall in the message tree
        responses = MessageSequence()
        asyncio.create_task(
            self._asubmit_agent_call(request=request, sender_alias=sender_alias, responses=responses, **kwargs)
        )
        return responses

    async def _asubmit_agent_call(
        self, request: MessagePromise, sender_alias: str, responses: MessageSequence, **kwargs
    ) -> None:
        with responses:
            try:
                agent_call = MessagePromise(
                    forum=self._forum,
                    in_reply_to=request,
                    detached_msg=AgentCall(
                        content=self.agent_alias,  # the recipient of the call is this agent
                        sender_alias=sender_alias,
                        metadata=Freeform(**kwargs),
                    ),
                )
                await self._acall_agent_func(agent_call, responses, **kwargs)
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                # catch all exceptions, including KeyboardInterrupt
                responses.send(exc)  # TODO Oleksandr: introduce ErrorMessage

    async def _acall_agent_func(self, agent_call: MessagePromise, responses: MessageSequence, **kwargs) -> None:
        request = await agent_call.aget_previous_message()
        with AgentContext(agent_alias=self.agent_alias):
            await self._func(request, responses, **kwargs)


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
