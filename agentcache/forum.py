"""
The Forum class is the main entry point for the agentcache library. It is used to create a forum, register agents in
it, and call agents. The Forum class is also responsible for storing messages in the forum (it uses ImmutableStorage
for that).
"""
import asyncio
import contextvars
from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel, ConfigDict

from agentcache.models import Message, Freeform, AgentCall
from agentcache.promises import MessagePromise, DetachedMsgPromise, MessageSequence, DetachedAgentCallPromise
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

    def new_message_promise(
        self,
        content: Optional[SingleMessageType] = None,
        sender_alias: Optional[str] = None,
        in_reply_to: Optional["MessagePromise"] = None,
        **metadata,
    ) -> "MessagePromise":
        """
        Create a new, detached message promise in the forum. "Detached message promise" means that this message
        promise may be a reply to another message promise that may or may not be "materialized" yet.
        """
        if isinstance(content, str):
            a_forward_of = None
        elif isinstance(content, Message):
            a_forward_of = MessagePromise(forum=self.forum, materialized_msg=content)
            # TODO Oleksandr: should we store the materialized_msg ?
            #  (the promise will not store it since it is already "materialized")
            #  or do we trust that something else already stored it ?
            content = ""  # this is a hack (the content will actually be taken from the forwarded message)
        elif isinstance(content, MessagePromise):
            a_forward_of = content
            content = ""  # this is a hack (the content will actually be taken from the forwarded message)
        else:
            raise ValueError(f"Unexpected message content type: {type(content)}")

        return DetachedMsgPromise(
            forum=self,
            in_reply_to=in_reply_to,
            a_forward_of=a_forward_of,
            detached_msg=Message(  # TODO Oleksandr: introduce a concept of PartialMessage to make this cleaner ?
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


class Agent:
    """A wrapper for an agent function that allows calling the agent."""

    def __init__(self, forum: Forum, func: AgentFunction) -> None:
        self.forum = forum
        self.agent_alias = func.__name__
        self._func = func

    def call(self, request: "MessagePromise", sender_alias: Optional[str] = None, **kwargs) -> "MessageSequence":
        """Call the agent."""
        agent_call = DetachedAgentCallPromise(
            forum=self.forum,
            message_sequence=MessageSequence(items_so_far=[request], completed=True),
            detached_agent_call=AgentCall(
                content=self.agent_alias,  # the recipient of the call is this agent
                sender_alias=self.forum.resolve_sender_alias(sender_alias),
                metadata=Freeform(**kwargs),
            ),
        )
        responses = MessageSequence()
        asyncio.create_task(self._acall_agent_func(agent_call=agent_call, responses=responses, **kwargs))
        return responses

    async def _acall_agent_func(self, agent_call: "MessagePromise", responses: "MessageSequence", **kwargs) -> None:
        with responses:
            ctx = InteractionContext(forum=self.forum, agent=self, responses=responses, latest_message=agent_call)
            try:
                request = await agent_call.aget_previous_message()
                with ctx:
                    await self._func(request, ctx, **kwargs)
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                # catch all exceptions, including KeyboardInterrupt
                ctx.respond(exc)


class InteractionContext:
    """
    A context within which an agent is called. This is needed for things like looking up a sender alias for a message
    that is being created by the agent, so it can be populated in the message automatically (and other similar things).
    """

    _current_context: ContextVar[Optional["InteractionContext"]] = ContextVar("_current_context", default=None)

    def __init__(
        self, forum: Forum, agent: Agent, responses: "MessageSequence", latest_message: Optional["MessagePromise"]
    ) -> None:
        self.forum = forum
        self.this_agent = agent
        # TODO Oleksandr: self.parent_context: Optional["InteractionContext"] ?
        self._responses = responses
        self._latest_message = latest_message
        self._previous_ctx_token: Optional[contextvars.Token] = None

    def respond(self, content: MessageType, sender_alias: Optional[str] = None, **metadata) -> None:
        """Respond with a message or a sequence of messages."""
        if isinstance(content, BaseException):
            # TODO Oleksandr: introduce the concept of ErrorMessage and move this if into Forum.new_message_promise()
            self._responses.send(content)
            return

        if isinstance(content, (str, Message, MessagePromise)):
            msg_promise = self.forum.new_message_promise(
                content=content, sender_alias=sender_alias, in_reply_to=self._latest_message
            )
        else:
            if hasattr(content, "__iter__"):
                for item in content:
                    self.respond(item, sender_alias=sender_alias, **metadata)
            elif hasattr(content, "__aiter__"):
                raise ValueError("Use `await ctx.arespond(...)` for async iterables")
            else:
                raise ValueError(f"Unexpected message content type: {type(content)}")
            return

        self._latest_message = msg_promise
        self._responses.send(msg_promise)

    async def arespond(self, content: MessageType, sender_alias: Optional[str] = None, **metadata) -> None:
        """Respond with a message or a sequence of messages (async version)."""
        if isinstance(content, BaseException):
            # TODO Oleksandr: introduce the concept of ErrorMessage and move this if into Forum.new_message_promise()
            self._responses.send(content)
            return

        if isinstance(content, (str, Message, MessagePromise)):
            msg_promise = self.forum.new_message_promise(
                content=content, sender_alias=sender_alias, in_reply_to=self._latest_message
            )
        else:
            if hasattr(content, "__iter__"):
                for item in content:
                    self.respond(item, sender_alias=sender_alias, **metadata)
            elif hasattr(content, "__aiter__"):
                async for item in content:
                    self.respond(item, sender_alias=sender_alias, **metadata)
            else:
                raise ValueError(f"Unexpected message content type: {type(content)}")
            return

        self._latest_message = msg_promise
        self._responses.send(msg_promise)

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
        self._current_context.reset(self._previous_ctx_token)
        self._previous_ctx_token = None
