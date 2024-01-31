# pylint: disable=too-many-arguments
"""
The Forum class is the main entry point for the agentforum library. It is used to create a forum, register agents in
it, and call agents. The Forum class is also responsible for storing messages in the forum (it uses ForumTrees
for that).
"""
import asyncio

# noinspection PyPackageRequirements
import contextvars
import typing

# noinspection PyPackageRequirements
from contextvars import ContextVar
from typing import Optional, AsyncIterator, Union, Callable

from pydantic import BaseModel, ConfigDict, PrivateAttr, Field

from agentforum.errors import ForumErrorFormatter, NoAskingAgentError
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

    # noinspection PyProtectedMember
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
        # pylint: disable=protected-access
        if isinstance(content, BaseException):
            if isinstance(content, ForumErrorFormatter):
                error_formatter = content
            else:
                error_formatter = ForumErrorFormatter(original_error=content)
            msg_promise = MessagePromise(
                forum=self.forum,
                content=await error_formatter.agenerate_error_message(self._latest_msg_promise),
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                is_error=True,
                error=content,
                **override_metadata,
            )
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, MessagePromise):
            msg_promise = MessagePromise(
                forum=self.forum,
                content=content,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                is_error=content.is_error,
                error=content._error,
                **override_metadata,
            )
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, Message):
            msg_promise = MessagePromise(
                forum=self.forum,
                content=content,
                default_sender_alias=default_sender_alias,
                do_not_forward_if_possible=do_not_forward_if_possible,
                branch_from=self._latest_msg_promise,
                is_error=content.is_error,
                error=content._error,
                **override_metadata,
            )
            self._latest_msg_promise = msg_promise
            yield msg_promise

        elif isinstance(content, (str, StreamedMessage)):
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

    def ask(
        self,
        content: Optional["MessageType"] = None,
        override_sender_alias: Optional[str] = None,
        branch_from: Optional[MessagePromise] = None,
        conversation: Optional[ConversationTracker] = None,
        force_new_conversation: bool = False,
        **function_kwargs,
    ) -> "AsyncMessageSequence":
        """
        "Ask" the agent and immediately receive an AsyncMessageSequence object that can be used to obtain the agent's
        response(s). If force_new_conversation is False and conversation is not specified and pre-existing messages are
        passed as requests (for ex. messages that came from other agents), then this agent call will be automatically
        branched off of the conversation branch those pre-existing messages belong to (the history will be inherited
        from those messages, in other words).
        """
        return self._quick_call(
            is_asking=True,
            content=content,
            override_sender_alias=override_sender_alias,
            branch_from=branch_from,
            conversation=conversation,
            force_new_conversation=force_new_conversation,
            **function_kwargs,
        )

    def start_asking(
        self,
        branch_from: Optional[MessagePromise] = None,
        conversation: Optional[ConversationTracker] = None,
        force_new_conversation: bool = False,
        **function_kwargs,
    ) -> "AgentCall":
        """
        Initiate the process of "asking" the agent. Returns an AgentCall object that can be used to send requests to
        the agent by calling `send_request()` zero or more times and receive its responses by calling
        `response_sequence()` at the end. If force_new_conversation is False and conversation is not specified and
        pre-existing messages are passed as requests (for ex. messages that came from other agents), then this agent
        call will be automatically branched off of the conversation branch those pre-existing messages belong to (the
        history will be inherited from those messages, in other words).
        """
        return self._call(
            is_asking=True,
            branch_from=branch_from,
            conversation=conversation,
            force_new_conversation=force_new_conversation,
            **function_kwargs,
        )

    def tell(
        self,
        content: Optional["MessageType"] = None,
        override_sender_alias: Optional[str] = None,
        branch_from: Optional[MessagePromise] = None,
        conversation: Optional[ConversationTracker] = None,
        force_new_conversation: bool = False,
        **function_kwargs,
    ) -> None:
        """
        "Tell" the agent. Does not return anything, because it's a one-way communication. If force_new_conversation
        is False and conversation is not specified and pre-existing messages are passed as requests (for ex. messages
        that came from other agents), then this agent call will be automatically branched off of the conversation
        branch those pre-existing messages belong to (the history will be inherited from those messages, in other
        words).
        """
        self._quick_call(
            is_asking=False,
            content=content,
            override_sender_alias=override_sender_alias,
            branch_from=branch_from,
            conversation=conversation,
            force_new_conversation=force_new_conversation,
            **function_kwargs,
        )

    def start_telling(
        self,
        branch_from: Optional[MessagePromise] = None,
        conversation: Optional[ConversationTracker] = None,
        force_new_conversation: bool = False,
        **function_kwargs,
    ) -> "AgentCall":
        """
        Initiate the process of "telling" the agent. Returns an AgentCall object that can be used to send requests to
        the agent by calling `send_request()` zero or more times and calling `finish()` at the end. If
        force_new_conversation is False and conversation is not specified and pre-existing messages are passed as
        requests (for ex. messages that came from other agents), then this agent call will be automatically branched
        off of the conversation branch those pre-existing messages belong to (the history will be inherited from those
        messages, in other words).
        """
        return self._call(
            is_asking=False,
            branch_from=branch_from,
            conversation=conversation,
            force_new_conversation=force_new_conversation,
            **function_kwargs,
        )

    def _quick_call(  # pylint: disable=too-many-arguments
        self,
        is_asking: bool,
        content: Optional["MessageType"] = None,
        override_sender_alias: Optional[str] = None,
        branch_from: Optional[MessagePromise] = None,
        conversation: Optional[ConversationTracker] = None,
        force_new_conversation: bool = False,
        **function_kwargs,
    ) -> Optional["AsyncMessageSequence"]:
        agent_call = self._call(
            is_asking=is_asking,
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
        return agent_call.response_sequence() if is_asking else None

    def _call(
        self,
        is_asking: bool,
        branch_from: Optional[MessagePromise] = None,
        conversation: Optional[ConversationTracker] = None,
        force_new_conversation: bool = False,
        **function_kwargs,
    ) -> "AgentCall":
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
            forum=self.forum,
            conversation=conversation,
            receiving_agent=self,
            is_asking=is_asking,
            do_not_forward_if_possible=not force_new_conversation,
            **function_kwargs,
        )

        parent_ctx = InteractionContext.get_current_context()
        # TODO Oleksandr: if there is no active agent call, InteractionContext.get_current_context() should return a
        #  default, "USER" context and not just None
        if parent_ctx:
            parent_ctx._child_agent_calls.append(agent_call)  # pylint: disable=protected-access

        asyncio.create_task(self._acall_non_cached_agent_func(agent_call=agent_call, **function_kwargs))
        return agent_call

    async def _acall_non_cached_agent_func(self, agent_call: "AgentCall", **function_kwargs) -> None:
        # pylint: disable=protected-access,broad-except
        with agent_call._response_producer:
            with InteractionContext(
                forum=self.forum,
                agent=self,
                request_messages=agent_call._request_messages,
                is_asker_context=agent_call.is_asking,
                response_producer=agent_call._response_producer,
            ) as ctx:
                try:
                    await self._func(ctx, **function_kwargs)
                except BaseException as exc:
                    ctx.get_asker_context().respond(exc)


# noinspection PyProtectedMember
class InteractionContext:  # pylint: disable=too-many-instance-attributes
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
        is_asker_context: bool,
        response_producer: "AsyncMessageSequence._MessageProducer",
    ) -> None:
        self.forum = forum
        self.this_agent = agent
        self.request_messages = request_messages
        self.is_asker_context = is_asker_context
        self.parent_context: Optional["InteractionContext"] = self.get_current_context()

        self._response_producer = response_producer
        self._child_agent_calls: list[AgentCall] = []
        self._previous_ctx_token: Optional[contextvars.Token] = None

    def respond(self, content: "MessageType", **metadata) -> None:
        """Respond with a message or a sequence of messages."""
        if not self.is_asker_context:
            raise NoAskingAgentError("Cannot respond in a context that is not asking")
        self._response_producer.send_zero_or_more_messages(content, **metadata)

    @classmethod
    def get_current_context(cls) -> Optional["InteractionContext"]:
        """Get the current InteractionContext object."""
        return cls._current_context.get()

    def get_asker_context(self) -> "InteractionContext":
        """
        Get the InteractionContext object of the agent that is currently asking. If there is no agent asking, raise
        NoAskingAgentError.
        """
        ctx = self
        while ctx:
            if ctx.is_asker_context:
                return ctx
            ctx = ctx.parent_context
        raise NoAskingAgentError("There is no agent up the chain of parent contexts that is currently asking")

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
        is_asking: bool,
        do_not_forward_if_possible: bool = True,
        **function_kwargs,
    ) -> None:
        self.forum = forum
        self.receiving_agent = receiving_agent
        self.is_asking = is_asking

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

        if is_asking:
            # TODO TODO TODO Oleksandr: switch to branch_from=NO_VALUE and employ reply_to (reply to AgentCallMsg)
            self._response_messages = AsyncMessageSequence(
                conversation, default_sender_alias=self.receiving_agent.alias
            )
            self._response_producer = AsyncMessageSequence._MessageProducer(self._response_messages)
        else:
            self._response_messages = None
            self._response_producer = None

    def send_request(self, content: "MessageType", **metadata) -> "AgentCall":
        """Send a request to the agent."""
        self._request_producer.send_zero_or_more_messages(content, **metadata)
        return self

    def response_sequence(self) -> "AsyncMessageSequence":
        """
        Finish the agent call and return the agent's response(s).

        NOTE: After this method is called it is not possible to send any more requests to this AgentCall object.
        """
        if not self.is_asking:
            raise NoAskingAgentError(
                "Cannot get response sequence for an agent call that is not asking, "
                "use ask()/start_asking() instead of tell()/start_telling()"
            )
        self.finish()
        return self._response_messages

    def finish(self) -> None:
        """
        Finish the agent call.

        NOTE: After this method is called it is not possible to send any more requests to this AgentCall object.
        """
        self._request_producer.close()
