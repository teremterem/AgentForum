"""
Typing definitions that involve imports from agentforum.
"""
from typing import Callable, Awaitable, Union, Iterable, AsyncIterable

from agentforum.forum import InteractionContext
from agentforum.models import Message
from agentforum.promises import StreamedMessage, MessagePromise

SingleMessageType = Union[str, StreamedMessage, Message, MessagePromise, BaseException]
AgentFunction = Callable[[InteractionContext, ...], Awaitable[None]]

# TODO Oleksandr: why not allow Iterable and AsyncIterable of MessageType itself ?
MessageType = Union[SingleMessageType, Iterable[SingleMessageType], AsyncIterable[SingleMessageType]]
