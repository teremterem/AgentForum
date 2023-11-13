# pylint: disable=protected-access
"""
The Forum class is the main entry point for the agentcache library. It is used to create a forum, register agents in
it, and call agents. The Forum class is also responsible for storing messages in the forum (it uses ImmutableStorage
for that).
"""
import asyncio
import contextvars
from contextvars import ContextVar
from typing import Optional, List, Dict, AsyncIterator

from pydantic import BaseModel, ConfigDict, PrivateAttr

from agentcache.models import Message, Freeform, AgentCallMsg, Immutable
from agentcache.promises import MessagePromise, MessageSequence, DetachedAgentCallMsgPromise, DetachedMsgPromise
from agentcache.storage import ImmutableStorage
from agentcache.typing import AgentFunction, MessageType, SingleMessageType

USER_ALIAS = "USER"


class ConversationTracker:
    """An object that tracks the tip of a conversation branch."""

    # TODO Oleksandr: Introduce the concept of MessagePlaceholder instead of MessagePromise
    #  (composition instead of polymorphism) and move most of the logic from ConversationTracker to MessagePlaceholder.
    #  This way it will become possible to delay the decision of whether any given message is eligible for
    #  "do_not_forward_if_possible".

    def __init__(self, forum: "Forum", branch_from: Optional[MessagePromise] = None) -> None:
        self.forum = forum
        self._latest_msg_promise = branch_from

    @property
    def has_prior_history(self) -> bool:
        """Check if there is prior history in this conversation."""
        return bool(self._latest_msg_promise)

    def new_message_promise(self, content: SingleMessageType, sender_alias: str, **metadata) -> "MessagePromise":
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

    async def aappend_zero_or_more_messages(
        self, content: MessageType, sender_alias: str, do_not_forward_if_possible: bool = True, **metadata
    ) -> AsyncIterator[MessagePromise]:
        """
        Append zero or more messages to the conversation. Returns an async iterator that yields message promises.
        """
        if do_not_forward_if_possible and not self.has_prior_history and not metadata:
            # if there is no prior history (and no extra metadata) we can just append the original message
            # (or sequence of messages) instead of creating message forwards
            # TODO Oleksandr: move this logic into the future MessagePlaceholder class, because right now it works in
            #  quite an unpredictable and hard to comprehend way

            if isinstance(content, MessageSequence):
                async for msg_promise in content:
                    self._latest_msg_promise = msg_promise
                    # TODO Oleksandr: other parallel tasks may submit messages to the same conversation which will
                    #  mess it up (because we are not doing forwards here) - how to protect from this ?
                    yield msg_promise
                return
            if isinstance(content, MessagePromise):
                self._latest_msg_promise = content
                yield content
                return
            if isinstance(content, Message):
                self._latest_msg_promise = MessagePromise(forum=self.forum, materialized_msg=content)
                yield self._latest_msg_promise
                return

        # if it's not a plain string then it should be forwarded (either because prior history in this conversation
        # should be maintained or because there is extra metadata)

        if isinstance(content, (str, Message, MessagePromise)):
            yield self.new_message_promise(
                content=content,
                sender_alias=sender_alias,
                **metadata,
            )

        elif hasattr(content, "__iter__"):
            for msg in content:
                async for msg_promise in self.aappend_zero_or_more_messages(
                    content=msg,
                    sender_alias=sender_alias,
                    **metadata,
                ):
                    yield msg_promise
        elif hasattr(content, "__aiter__"):
            async for msg in content:
                async for msg_promise in self.aappend_zero_or_more_messages(
                    content=msg,
                    sender_alias=sender_alias,
                    **metadata,
                ):
                    yield msg_promise
        else:
            raise ValueError(f"Unexpected message content type: {type(content)}")


class Forum(BaseModel):
    """A forum for agents to communicate. Messages in the forum assemble in a tree-like structure."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    immutable_storage: ImmutableStorage
    _conversations: Dict[str, ConversationTracker] = PrivateAttr(default_factory=dict)

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

    def get_conversation(
        self, descriptor: Immutable, branch_from_if_new: Optional["MessagePromise"] = None
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

    def __init__(self, forum: Forum, func: AgentFunction) -> None:
        self.forum = forum
        self.agent_alias = func.__name__
        self._func = func

    def quick_call(
        self,
        content: Optional[MessageType],
        override_sender_alias: Optional[str] = None,
        conversation: Optional[ConversationTracker] = None,
        force_new_conversation: bool = False,
        **function_kwargs,
    ) -> "MessageSequence":
        """
        Call the agent and immediately finish the call. Returns a MessageSequence object that contains the agent's
        response(s). If force_new_conversation is False and conversation is not specified and pre-existing messages are
        passed as requests (for ex. messages that came from other agents), then this agent call will be automatically
        branched off of the conversation branch those pre-existing messages belong to (the history will be inherited
        from those messages, in other words).
        """
        agent_call = self.call(
            conversation=conversation,
            force_new_conversation=force_new_conversation,
            **function_kwargs,
        )
        if content is not None:
            agent_call.send_request(content, override_sender_alias=override_sender_alias)
        return agent_call.finish()

    def call(
        self,
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
        if conversation:
            if conversation.has_prior_history and force_new_conversation:
                raise ValueError("Cannot force a new conversation when there is prior history in ConversationTracker")
        else:
            conversation = ConversationTracker(self.forum)

        agent_call = AgentCall(
            self.forum, conversation, self, do_not_forward_if_possible=not force_new_conversation, **function_kwargs
        )

        parent_ctx = InteractionContext.get_current_context()
        # TODO Oleksandr: get rid of this if-statement by making Forum a context manager too and making sure all the
        #  "seed" agent calls are done within the context of the forum ?
        if parent_ctx:
            parent_ctx._child_agent_calls.append(agent_call)

        asyncio.create_task(self._acall_non_cached_agent_func(agent_call=agent_call, **function_kwargs))
        return agent_call

    async def _acall_non_cached_agent_func(self, agent_call: "AgentCall", **function_kwargs) -> None:
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
        request_messages: MessageSequence,
        response_producer: "MessageSequence._MessageProducer",
    ) -> None:
        self.forum = forum
        self.this_agent = agent
        self.request_messages = request_messages
        self._response_producer = response_producer
        self._child_agent_calls: List[AgentCall] = []
        self._previous_ctx_token: Optional[contextvars.Token] = None
        # TODO Oleksandr: self.parent_context: Optional["InteractionContext"] ?

    def respond(self, content: MessageType, override_sender_alias: Optional[str] = None, **metadata) -> None:
        """Respond with a message or a sequence of messages."""
        self._response_producer.send_zero_or_more_messages(
            content, override_sender_alias=override_sender_alias, **metadata
        )

    @classmethod
    def get_current_context(cls) -> Optional["InteractionContext"]:
        """Get the current InteractionContext object."""
        return cls._current_context.get()

    @classmethod
    def get_current_sender_alias(cls) -> str:
        """Get the sender alias from the current InteractionContext object."""
        ctx = cls.get_current_context()
        if ctx:
            return ctx.this_agent.agent_alias
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
            child_agent_call.finish()
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
        **kwargs,
    ) -> None:
        self.forum = forum
        self.receiving_agent = receiving_agent

        # TODO Oleksandr: either explain this temporary_sub_conversation in a comment or refactor it completely when
        #  you get to implementing cached agent calls
        temporary_sub_conversation = ConversationTracker(forum=forum, branch_from=conversation._latest_msg_promise)

        self._request_messages = MessageSequence(
            temporary_sub_conversation,
            default_sender_alias=InteractionContext.get_current_sender_alias(),
            do_not_forward_if_possible=do_not_forward_if_possible,
        )
        self._request_producer = MessageSequence._MessageProducer(self._request_messages)

        # TODO Oleksandr: extract this into a separate method of ConversationTracker
        agent_call_msg_promise = DetachedAgentCallMsgPromise(
            forum=self.forum,
            request_messages=self._request_messages,
            detached_agent_call_msg=AgentCallMsg(
                content=self.receiving_agent.agent_alias,
                # we keep agent calls anonymous to be able to cache call results for multiple caller agents to reuse
                sender_alias="",
                metadata=Freeform(**kwargs),
            ),
        )
        conversation._latest_msg_promise = agent_call_msg_promise

        self._response_messages = MessageSequence(conversation, default_sender_alias=self.receiving_agent.agent_alias)
        self._response_producer = MessageSequence._MessageProducer(self._response_messages)

    def send_request(
        self, content: MessageType, override_sender_alias: Optional[str] = None, **metadata
    ) -> "AgentCall":
        """Send a request to the agent."""
        self._request_producer.send_zero_or_more_messages(
            content, override_sender_alias=override_sender_alias, **metadata
        )
        return self

    def finish(self) -> "MessageSequence":
        """Finish the agent call and return the agent's response(s)."""
        self._request_producer.close()
        return self._response_messages
