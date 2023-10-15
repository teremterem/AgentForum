from typing import List

from agentcache.ext.llms.openai import aopenai_chat_completion
from agentcache.models import Message
from agentcache.typing import MessageType


class AgentFirstDraft:
    def __init__(self) -> None:
        # TODO Oleksandr: make message history the responsibility of the AgentCache framework
        self._message_history: List[MessageType] = []

    async def arun(self, user_input: str, **kwargs) -> MessageType:
        self._message_history.append(Message(content=user_input))
        response = await aopenai_chat_completion(messages=self._message_history, **kwargs)
        self._message_history.append(response)
        return response
