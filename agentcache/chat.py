from typing import List

from agentcache.typing import MessageType


class MessageBundle:
    def __init__(self) -> None:
        self.messages_so_far: List[MessageType] = []


class ChatManager:
    pass
