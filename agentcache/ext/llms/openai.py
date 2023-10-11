"""TODO Oleksandr"""
from typing import List

from agentcache.models import Message


async def achatgpt(messages: List[Message], stream: bool = False, **kwargs):
    """TODO Oleksandr"""
    import openai  # pylint: disable=import-outside-toplevel

    message_dicts = [{"role": message.role, "content": message.content} for message in messages]

    response = await openai.ChatCompletion.acreate(messages=message_dicts, stream=stream, **kwargs)
    if stream:
        return (token async for token in response)
    return response
