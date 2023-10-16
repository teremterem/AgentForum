from typing import List

from agentcache.ext.llms.openai import aopenai_chat_completion
from agentcache.models import MessageBundle
from agentcache.typing import MessageType


class AgentFirstDraft:
    def __init__(self) -> None:
        # TODO Oleksandr: make message history the responsibility of the AgentCache framework
        self._message_history: List[MessageType] = []

    async def arun(self, incoming: MessageBundle) -> MessageBundle:
        async for message in incoming:
            self._message_history.append(message)
        response = await aopenai_chat_completion(messages=self._message_history, kwargs=incoming.bundle_metadata)
        self._message_history.append(response)
        return MessageBundle(messages_so_far=[response], complete=True)
