"""Typing definitions for AgentForum."""
import typing
from typing import TypeVar, Callable, Awaitable, Union, Iterable, Any, AsyncIterable

if typing.TYPE_CHECKING:
    from agentforum.forum import InteractionContext
    from agentforum.promises import StreamedMessage, MessagePromise
    from agentforum.models import Message

    SingleMessageType = Union[str, StreamedMessage, Message, MessagePromise, BaseException]
    AgentFunction = Callable[[InteractionContext, ...], Awaitable[None]]
else:
    SingleMessageType = Union[Any]
    AgentFunction = Callable

# TODO Oleksandr: why not allow Iterable and AsyncIterable of MessageType itself ?
MessageType = Union[SingleMessageType, Iterable[SingleMessageType], AsyncIterable[SingleMessageType]]

IN = TypeVar("IN")
OUT = TypeVar("OUT")
