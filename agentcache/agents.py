"""TODO Oleksandr: add module docstring"""
from agentcache.ext.llms.openai import aopenai_chat_completion
from agentcache.models import AsyncMessageBundle


async def afirst_agent(incoming: AsyncMessageBundle) -> AsyncMessageBundle:
    """The very first agent."""
    # TODO Oleksandr: a bundle of messages that are not part of the same conversation doesn't make sense
    # TODO Oleksandr: if we want to cache the agent response then all incoming messages should be known before the
    #  agent function is entered
    message = (await incoming.aget_all())[-1]
    response = await aopenai_chat_completion(messages=await message.aget_full_chat(), kwargs=incoming.bundle_metadata)
    return AsyncMessageBundle(items_so_far=[response], completed=True)
