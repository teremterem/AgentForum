"""Typing definitions for AgentCache."""
import typing
from typing import TypeVar, Callable, Awaitable

if typing.TYPE_CHECKING:
    from agentcache.forum import MessagePromise, MessageSequence

    AgentFunction = Callable[[MessagePromise, MessageSequence, ...], Awaitable[None]]
else:
    AgentFunction = Callable

IN = TypeVar("IN")
OUT = TypeVar("OUT")
