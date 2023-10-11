"""Typing definitions for AgentCache."""
import typing

if typing.TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from agentcache.models import Message, StreamedMessage

MessageType = typing.Union["Message", "StreamedMessage"]
