# pylint: disable=protected-access
"""
The Forum class is the main entry point for the agentcache library. It is used to create a forum, register agents in
it, and call agents. The Forum class is also responsible for storing messages in the forum (it uses ImmutableStorage
for that).
"""
import asyncio
import contextvars
from contextvars import ContextVar
from typing import Optional, List

from pydantic import BaseModel, ConfigDict

from agentcache.models import Message, Freeform, AgentCallMsg
from agentcache.promises import MessagePromise, DetachedMsgPromise, MessageSequence, DetachedAgentCallMsgPromise
from agentcache.storage import ImmutableStorage
from agentcache.typing import AgentFunction, MessageType, SingleMessageType

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

    def _new_message_promise(
        self,
        content: Optional[SingleMessageType] = None,
        sender_alias: Optional[str] = None,
        branch_from: Optional["MessagePromise"] = None,
        **metadata,
    ) -> "MessagePromise":
        """
        Create a new, detached message promise in the forum. "Detached message promise" means that this message
        promise may be a reply to another message promise that may or may not be "materialized" yet.
        """
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

        return DetachedMsgPromise(
            forum=self,
            branch_from=branch_from,
            forward_of=forward_of,
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
        current InteractionContext is used to get the agent alias, and if there is no current InteractionContext, then
        DEFAULT_AGENT_ALIAS (which translates to "USER") is used.
        """
        if not sender_alias:
            ctx = InteractionContext.get_current_context()
            if ctx:
                sender_alias = ctx.this_agent.agent_alias
        return sender_alias or DEFAULT_AGENT_ALIAS


# noinspection PyProtectedMember
class Agent:
    """A wrapper for an agent function that allows calling the agent."""

    def __init__(self, forum: Forum, func: AgentFunction) -> None:
        self.forum = forum
        self.agent_alias = func.__name__
        self._func = func

    def quick_call(
        self,
        content: Optional[MessageType],
        sender_alias: Optional[str] = None,
        branch_from: Optional["MessagePromise"] = None,
        **function_kwargs,
    ) -> "MessageSequence":
        agent_call = self.call(sender_alias=sender_alias, branch_from=branch_from, **function_kwargs)
        if content is not None:
            agent_call.send_request(content, sender_alias=sender_alias)
        return agent_call.finish()

    async def aquick_call(
        self,
        content: Optional[MessageType],
        sender_alias: Optional[str] = None,
        branch_from: Optional["MessagePromise"] = None,
        **function_kwargs,
    ) -> "MessageSequence":
        agent_call = self.call(sender_alias=sender_alias, branch_from=branch_from, **function_kwargs)
        if content is not None:
            await agent_call.asend_request(content, sender_alias=sender_alias)
        return agent_call.finish()

    def call(
        self, sender_alias: Optional[str] = None, branch_from: Optional["MessagePromise"] = None, **function_kwargs
    ) -> "AgentCall":
        agent_call = AgentCall(
            forum=self.forum,
            receiving_agent=self,
            sender_alias=sender_alias,
            branch_from=branch_from,
            **function_kwargs,
        )

        parent_ctx = InteractionContext.get_current_context()
        # TODO Oleksandr: get rid of this if by making Forum a context manager too and making sure all the
        #  "seed" agent calls are done within the context of the forum ?
        if parent_ctx:
            parent_ctx._child_agent_calls.append(agent_call)

        asyncio.create_task(self._acall_non_cached_agent_func(agent_call=agent_call, **function_kwargs))
        return agent_call

    async def _acall_non_cached_agent_func(self, agent_call: "AgentCall", **function_kwargs) -> None:
        with agent_call._responses:
            with InteractionContext(forum=self.forum, agent=self, responses=agent_call._responses) as ctx:
                try:
                    await self._func(agent_call._requests, ctx, **function_kwargs)
                except BaseException as exc:  # pylint: disable=broad-exception-caught
                    # catch all exceptions, including KeyboardInterrupt
                    ctx.respond(exc)


# noinspection PyProtectedMember
class InteractionContext:
    """
    A context within which an agent is called. This is needed for things like looking up a sender alias for a message
    that is being created by the agent, so it can be populated in the message automatically (and other similar things).
    """

    _current_context: ContextVar[Optional["InteractionContext"]] = ContextVar("_current_context", default=None)

    def __init__(self, forum: Forum, agent: Agent, responses: "MessageSequence") -> None:
        self.forum = forum
        self.this_agent = agent
        # TODO Oleksandr: self.parent_context: Optional["InteractionContext"] ?
        self._responses = responses
        self._child_agent_calls: List[AgentCall] = []
        self._previous_ctx_token: Optional[contextvars.Token] = None

    def respond(self, content: MessageType, sender_alias: Optional[str] = None, **metadata) -> None:
        """Respond with a message or a sequence of messages."""
        self._responses._send_msg(content, sender_alias=sender_alias, **metadata)

    async def arespond(self, content: MessageType, sender_alias: Optional[str] = None, **metadata) -> None:
        """Respond with a message or a sequence of messages (async version)."""
        await self._responses._asend_msg(content, sender_alias=sender_alias, **metadata)

    @classmethod
    def get_current_context(cls) -> Optional["InteractionContext"]:
        """Get the current InteractionContext object."""
        return cls._current_context.get()

    def __enter__(self) -> "InteractionContext":
        """Set this context as the current context."""
        if self._previous_ctx_token:
            raise RuntimeError("InteractionContext is not reentrant")
        self._previous_ctx_token = self._current_context.set(self)  # <- this is the context switch
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore the context that was current before this one."""
        for child_agent_call in self._child_agent_calls:
            # just in case any of the child agent calls weren't explicitly finished, finish them now
            child_agent_call.finish()
        self._current_context.reset(self._previous_ctx_token)
        self._previous_ctx_token = None


# noinspection PyProtectedMember
class AgentCall:
    def __init__(
        self,
        forum: Forum,
        receiving_agent: Agent,
        sender_alias: Optional[str] = None,
        branch_from: Optional["MessagePromise"] = None,
        **kwargs,
    ) -> None:
        self.forum = forum
        self.receiving_agent = receiving_agent

        self._requests = MessageSequence(self.forum, branch_from=branch_from)
        agent_call_msg_promise = DetachedAgentCallMsgPromise(
            forum=self.forum,
            message_sequence=self._requests,
            detached_agent_call_msg=AgentCallMsg(
                content=self.receiving_agent.agent_alias,
                sender_alias=self.forum.resolve_sender_alias(sender_alias),
                metadata=Freeform(**kwargs),
            ),
        )
        self._responses = MessageSequence(self.forum, branch_from=agent_call_msg_promise)

    def send_request(self, content: MessageType, sender_alias: Optional[str] = None, **metadata) -> "AgentCall":
        self._requests._send_msg(content, sender_alias=sender_alias, **metadata)
        return self

    async def asend_request(self, content: MessageType, sender_alias: Optional[str] = None, **metadata) -> "AgentCall":
        await self._requests._asend_msg(content, sender_alias=sender_alias, **metadata)
        return self

    def finish(self) -> "MessageSequence":
        self._requests._close()
        return self._responses
