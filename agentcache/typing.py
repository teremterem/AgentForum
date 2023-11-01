"""Typing definitions for AgentCache."""
import typing
from typing import TypeVar, Callable, Awaitable

if typing.TYPE_CHECKING:
    from agentcache.forum import MessagePromise, InteractionContext

    AgentFunction = Callable[[MessagePromise, InteractionContext, ...], Awaitable[None]]
else:
    AgentFunction = Callable  # TODO Oleksandr: doesn't it just hide the type information from the users, though ?

IN = TypeVar("IN")
OUT = TypeVar("OUT")
