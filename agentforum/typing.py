"""
Typing definitions that involve imports from agentforum.
"""
from typing import Union, Iterable, AsyncIterable, Any, Protocol

from agentforum.forum import InteractionContext
from agentforum.models import Message
from agentforum.promises import StreamedMessage, MessagePromise


class AgentFunction(Protocol):
    """
    A protocol for agent functions.
    """

    async def __call__(self, ctx: InteractionContext, **kwargs) -> None:
        ...


# TODO Oleksandr: add documentation somewhere that explains what MessageType and SingleMessageType represent
SingleMessageType = Union[str, dict[str, Any], StreamedMessage, Message, MessagePromise, BaseException]
MessageType = Union[SingleMessageType, Iterable["MessageType"], AsyncIterable["MessageType"]]
