"""
Typing definitions that involve imports from agentforum.
"""
from typing import Callable, Awaitable, Union, Iterable, AsyncIterable, Dict, Any

from agentforum.forum import InteractionContext
from agentforum.models import Message
from agentforum.promises import StreamedMessage, MessagePromise

AgentFunction = Callable[[InteractionContext, ...], Awaitable[None]]

# TODO Oleksandr: add documentation somewhere that explains what MessageType and SingleMessageType represent
SingleMessageType = Union[str, Dict[str, Any], StreamedMessage, Message, MessagePromise, BaseException]
MessageType = Union[SingleMessageType, Iterable["MessageType"], AsyncIterable["MessageType"]]
