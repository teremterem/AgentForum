"""Typing definitions for AgentCache."""
import typing
from typing import TypeVar, Callable, Awaitable, Union, Iterable, Any, AsyncIterable

if typing.TYPE_CHECKING:
    from agentcache.forum import MessagePromise, InteractionContext
    from agentcache.models import Message

    SingleMessageType = Union[str, Message, MessagePromise, BaseException]
    AgentFunction = Callable[[MessagePromise, InteractionContext, ...], Awaitable[None]]
else:
    SingleMessageType = Union[Any]
    AgentFunction = Callable

MessageType = Union[SingleMessageType, Iterable[SingleMessageType], AsyncIterable[SingleMessageType]]

IN = TypeVar("IN")
OUT = TypeVar("OUT")
