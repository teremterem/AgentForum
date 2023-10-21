"""TODO Oleksandr: add module docstring"""
from typing import Union

from agentcache.ext.llms.openai import aopenai_chat_completion
from agentcache.model_wrappers import MessageSequence, StreamedMessage
from agentcache.models import Message


async def acall_agent_draft(request: Union[StreamedMessage, Message], **kwargs) -> MessageSequence:
    """The very first agent."""
    # TODO Oleksandr: a bundle of messages that are not part of the same conversation doesn't make sense
    # TODO Oleksandr: if we want to cache the agent response then all incoming messages should be known before the
    #  agent function is entered
    message = (await request.aget_all())[-1]
    response = await aopenai_chat_completion(messages=await message.aget_full_chat(), kwargs=request.sequence_metadata)
    return MessageSequence(items_so_far=[response], completed=True)


async def aagent_caller_draft(call: AgentCall) -> MessageSequence:
    """TODO Oleksandr"""


async def aagent_draft(request: StreamedMessage, **kwargs) -> MessageSequence:
    """TODO Oleksandr"""
