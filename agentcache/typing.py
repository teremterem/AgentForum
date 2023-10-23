"""Typing definitions for AgentCache."""
import typing
from typing import TypeVar, Callable, Awaitable

if typing.TYPE_CHECKING:
    from agentcache.forum import StreamedMessage, MessageSequence

    AgentFunction = Callable[[StreamedMessage, MessageSequence, ...], Awaitable[None]]
else:
    AgentFunction = Callable

IN = TypeVar("IN")
OUT = TypeVar("OUT")
