"""TODO Oleksandr: add module docstring"""
from typing import Union

from agentcache.ext.llms.openai import aopenai_chat_completion
from agentcache.model_wrappers import MessageSequence, StreamedMessage
from agentcache.models import Message, Freeform


async def acall_agent_draft(request: Union[StreamedMessage, Message], **kwargs) -> MessageSequence:
    """The very first agent."""
    requests = MessageSequence(  # TODO Oleksandr: replace this with AgentCall
        sequence_metadata=Freeform(**kwargs),
        items_so_far=[request],
        completed=True,
    )
    message = (await request.aget_all())[-1]
    response = await aopenai_chat_completion(
        messages=await message.aget_full_chat(), kwargs=requests.sequence_metadata
    )
    return MessageSequence(items_so_far=[response], completed=True)


async def aagent_caller_draft(call: AgentCall) -> MessageSequence:
    """TODO Oleksandr"""


async def aagent_draft(request: StreamedMessage, **kwargs) -> MessageSequence:
    """TODO Oleksandr"""
