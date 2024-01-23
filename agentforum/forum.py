"""
The Forum class is the main entry point for the agentforum library. It is used to create a forum, register agents in
it, and call agents. The Forum class is also responsible for storing messages in the forum (it uses ForumTrees
for that).
"""
import asyncio
import contextvars
import typing
from contextvars import ContextVar
from typing import Optional, AsyncIterator, Union, Callable

from pydantic import BaseModel, ConfigDict, PrivateAttr, Field

from agentforum.models import Message, Immutable
from agentforum.promises import MessagePromise, AsyncMessageSequence, StreamedMessage, AgentCallMsgPromise
from agentforum.storage.trees import ForumTrees
from agentforum.storage.trees_impl import InMemoryTrees
from agentforum.utils import Sentinel, NO_VALUE

if typing.TYPE_CHECKING:
    from agentforum.typing import AgentFunction, MessageType

USER_ALIAS = "USER"


class ConversationTracker:
    """
    An object that tracks the tip of a conversation branch.

    If `branch_from` is set to NO_VALUE then it means that whether this conversation is branched off of an existing
    branch of messages or not will be determined by the messages that are passed into this conversation later.
    """

    def __init__(self, forum: "Forum", branch_from: Optional[Union[MessagePromise, Sentinel]] = None) -> None:
        self.forum = forum
        self._latest_msg_promise = branch_from

    @property
    def has_prior_history(self) -> bool:
        """Check if there is prior history in this conversation."""
        return self._latest_msg_promise and self._latest_msg_promise != NO_VALUE

    async def aappend_zero_or_more_messages(
        self,
        content: "MessageType",
        default_sender_alias: str,
        do_not_forward_if_possible: bool = True,
        **override_metadata,
    ) -> AsyncIterator[MessagePromise]:
        """
        Append zero or more messages to the conversation. Returns an async iterator that yields message promises.
        """
        if isinstance(content, (str, Message, StreamedMessage, MessagePromise)):
            msg_promise = MessagePromise(
                forum=self.forum,
                content=content,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                **override_metadata,
            )
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, dict):
            msg_promise = MessagePromise(
                forum=self.forum,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                **{
                    **content,
                    **override_metadata,
                },
            )
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif hasattr(content, "__iter__"):
            # this is not a single message, this is a collection of messages
            for sub_msg in content:
                async for msg_promise in self.aappend_zero_or_more_messages(
                    content=sub_msg,
                    default_sender_alias=default_sender_alias,
                    do_not_forward_if_possible=do_not_forward_if_possible,
                    **override_metadata,
                ):
                    self._latest_msg_promise = msg_promise
                    yield msg_promise
        elif hasattr(content, "__aiter__"):
            # this is not a single message, this is an asynchronous collection of messages
            async for sub_msg in content:
                async for msg_promise in self.aappend_zero_or_more_messages(
                    content=sub_msg,
                    default_sender_alias=default_sender_alias,
                    do_not_forward_if_possible=do_not_forward_if_possible,
                    **override_metadata,
                ):
                    self._latest_msg_promise = msg_promise
                    yield msg_promise
        else:
            raise ValueError(f"Unexpected message content type: {type(content)}")


class Forum(BaseModel):
    """A forum for agents to communicate. Messages in the forum assemble in a tree-like structure."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    forum_trees: ForumTrees = Field(default_factory=InMemoryTrees)
    _conversations: dict[str, ConversationTracker] = PrivateAttr(default_factory=dict)

    def agent(  # pylint: disable=too-many-arguments
        self,
        func: Optional["AgentFunction"] = None,
        alias: Optional[str] = None,
        description: Optional[str] = None,
        uppercase_func_name: bool = True,
        normalize_spaces_in_docstring: bool = True,
    ) -> Union["Agent", Callable[["AgentFunction"], "Agent"]]:
        """A decorator that registers an agent function in the forum."""
        if func is None:
            # the decorator `@forum.agent(...)` was used with arguments
            def _decorator(f: "AgentFunction") -> "Agent":
                return Agent(
                    self,
                    f,
                    alias=alias,
                    description=description,
                    uppercase_func_name=uppercase_func_name,
                    normalize_spaces_in_docstring=normalize_spaces_in_docstring,
                )

            return _decorator

        # the decorator `@forum.agent` was used either without arguments or as a direct function call
        return Agent(
            self,
            func,
            alias=alias,
            description=description,
            uppercase_func_name=uppercase_func_name,
            normalize_spaces_in_docstring=normalize_spaces_in_docstring,
        )

    def get_conversation(
        self, descriptor: Immutable, branch_from_if_new: Optional[Union[MessagePromise, Sentinel]] = None
    ) -> ConversationTracker:
        """
        Get a ConversationTracker object that tracks the tip of a conversation branch. If the conversation doesn't
        exist yet, it will be created. If branch_from_if_new is specified, the conversation will be branched off of
        that message promise (as long as the conversation doesn't exist yet). Descriptor is used to uniquely identify
        the conversation. It can be an arbitrary Immutable object - its hash_key will be used to identify the
        conversation.
        """
        conversation = self._conversations.get(descriptor.hash_key)
        if not conversation:
            conversation = ConversationTracker(forum=self, branch_from=branch_from_if_new)
            self._conversations[descriptor.hash_key] = conversation

        return conversation


# noinspection PyProtectedMember
class Agent:
    """A wrapper for an agent function that allows calling the agent."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        forum: Forum,
        func: "AgentFunction",
        alias: Optional[str] = None,
        description: Optional[str] = None,
        uppercase_func_name: bool = True,
        normalize_spaces_in_docstring: bool = True,
    ) -> None:
        self.forum = forum
        self._func = func

        self.alias = alias
        if self.alias is None:
            self.alias = func.__name__
            if uppercase_func_name:
                self.alias = self.alias.upper()

        self.description = description
        if self.description is None:
            self.description = func.__doc__
            if self.description and normalize_spaces_in_docstring:
                self.description = " ".join(self.description.split())
        if self.description:
            # replace all {AGENT_ALIAS} entries in the description with the actual agent alias
            self.description = self.description.format(AGENT_ALIAS=self.alias)

        self.__name__ = self.alias
        self.__doc__ = self.description

    def quick_call(  # pylint: disable=too-many-arguments
        self,
        content: Optional["MessageType"] = None,
        override_sender_alias: Optional[str] = None,
        branch_from: Optional[MessagePromise] = None,
        conversation: Optional[ConversationTracker] = None,
        force_new_conversation: bool = False,
        **function_kwargs,
    ) -> "AsyncMessageSequence":
        """
        Call the agent and immediately finish the call. Returns a AsyncMessageSequence object that contains the agent's
        response(s). If force_new_conversation is False and conversation is not specified and pre-existing messages are
        passed as requests (for ex. messages that came from other agents), then this agent call will be automatically
        branched off of the conversation branch those pre-existing messages belong to (the history will be inherited
        from those messages, in other words).
        """
        agent_call = self.call(
            branch_from=branch_from,
            conversation=conversation,
            force_new_conversation=force_new_conversation,
            **function_kwargs,
        )
        if content is not None:
            if override_sender_alias:
                agent_call.send_request(content, sender_alias=override_sender_alias)
            else:
                agent_call.send_request(content)
        return agent_call.response_sequence()

    def call(
        self,
        branch_from: Optional[MessagePromise] = None,
        conversation: Optional[ConversationTracker] = None,
        force_new_conversation: bool = False,
        **function_kwargs,
    ) -> "AgentCall":
        """
        Call the agent. Returns an AgentCall object that can be used to send requests to the agent and receive its
        responses. If force_new_conversation is False and conversation is not specified and pre-existing messages are
        passed as requests (for ex. messages that came from other agents), then this agent call will be automatically
        branched off of the conversation branch those pre-existing messages belong to (the history will be inherited
        from those messages, in other words).
        """
        if branch_from and conversation:
            raise ValueError("Cannot specify both conversation and branch_from in Agent.call() or Agent.quick_call()")
        if branch_from:
            conversation = ConversationTracker(self.forum, branch_from=branch_from)

        if conversation:
            if conversation.has_prior_history and force_new_conversation:
                raise ValueError("Cannot force a new conversation when there is prior history in ConversationTracker")
        else:
            conversation = ConversationTracker(self.forum, branch_from=NO_VALUE)

        agent_call = AgentCall(
            self.forum, conversation, self, do_not_forward_if_possible=not force_new_conversation, **function_kwargs
        )

        parent_ctx = InteractionContext.get_current_context()
        # TODO Oleksandr: get rid of this if-statement by making Forum a context manager too and making sure all the
        #  "seed" agent calls are done within the context of the forum ?
        if parent_ctx:
            parent_ctx._child_agent_calls.append(agent_call)  # pylint: disable=protected-access

        asyncio.create_task(self._acall_non_cached_agent_func(agent_call=agent_call, **function_kwargs))
        return agent_call

    async def _acall_non_cached_agent_func(self, agent_call: "AgentCall", **function_kwargs) -> None:
        # pylint: disable=protected-access
        with agent_call._response_producer:
            with InteractionContext(
                forum=self.forum,
                agent=self,
                request_messages=agent_call._request_messages,
                response_producer=agent_call._response_producer,
            ) as ctx:
                try:
                    await self._func(ctx, **function_kwargs)
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

    def __init__(
        self,
        forum: Forum,
        agent: Agent,
        request_messages: AsyncMessageSequence,
        response_producer: "AsyncMessageSequence._MessageProducer",
    ) -> None:
        self.forum = forum
        self.this_agent = agent
        self.request_messages = request_messages
        self._response_producer = response_producer
        self._child_agent_calls: list[AgentCall] = []
        self._previous_ctx_token: Optional[contextvars.Token] = None
        # TODO Oleksandr: self.parent_context: Optional["InteractionContext"] ?

    def respond(self, content: "MessageType", **metadata) -> None:
        """Respond with a message or a sequence of messages."""
        self._response_producer.send_zero_or_more_messages(content, **metadata)

    @classmethod
    def get_current_context(cls) -> Optional["InteractionContext"]:
        """Get the current InteractionContext object."""
        return cls._current_context.get()

    @classmethod
    def get_current_sender_alias(cls) -> str:
        """Get the sender alias from the current InteractionContext object."""
        ctx = cls.get_current_context()
        if ctx:
            return ctx.this_agent.alias
        return USER_ALIAS

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
            child_agent_call.response_sequence()
        self._current_context.reset(self._previous_ctx_token)
        self._previous_ctx_token = None


# noinspection PyProtectedMember
class AgentCall:
    """
    A call to an agent. This object is returned by Agent.call() method. It is used to send requests to the agent and
    receive its responses.
    """

    def __init__(
        self,
        forum: Forum,
        conversation: ConversationTracker,
        receiving_agent: Agent,
        do_not_forward_if_possible: bool = True,
        **function_kwargs,
    ) -> None:
        self.forum = forum
        self.receiving_agent = receiving_agent

        # TODO Oleksandr: either explain this temporary_sub_conversation in a comment or refactor it completely when
        #  you get to implementing cached agent calls
        temporary_sub_conversation = ConversationTracker(forum=forum, branch_from=conversation._latest_msg_promise)

        self._request_messages = AsyncMessageSequence(
            temporary_sub_conversation,
            default_sender_alias=InteractionContext.get_current_sender_alias(),
            do_not_forward_if_possible=do_not_forward_if_possible,
        )
        self._request_producer = AsyncMessageSequence._MessageProducer(self._request_messages)

        agent_call_msg_promise = AgentCallMsgPromise(
            forum=self.forum,
            request_messages=self._request_messages,
            receiving_agent_alias=self.receiving_agent.alias,
            **function_kwargs,
        )
        conversation._latest_msg_promise = agent_call_msg_promise

        self._response_messages = AsyncMessageSequence(conversation, default_sender_alias=self.receiving_agent.alias)
        self._response_producer = AsyncMessageSequence._MessageProducer(self._response_messages)

    def send_request(self, content: "MessageType", **metadata) -> "AgentCall":
        """Send a request to the agent."""
        self._request_producer.send_zero_or_more_messages(content, **metadata)
        return self

    def response_sequence(self) -> "AsyncMessageSequence":
        """
        Finish the agent call and return the agent's response(s).

        NOTE: After this method is called it is not possible to send any more requests to this AgentCall object.
        """
        self._request_producer.close()
        return self._response_messages
