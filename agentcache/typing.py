"""Typing definitions for AgentCache."""
import typing
from typing import TypeVar

if typing.TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from agentcache.models import Message, StreamedMessage

MessageType = typing.Union["Message", "StreamedMessage"]

IN = TypeVar("IN")
OUT = TypeVar("OUT")
