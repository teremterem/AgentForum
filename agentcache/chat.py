from asyncio import Queue
from typing import List, Optional, Union

from agentcache.typing import MessageType
from agentcache.utils import END_OF_QUEUE


class MessageBundle:
    def __init__(self) -> None:
        self.messages_so_far: List[MessageType] = []
        # pylint: disable=unsubscriptable-object
        self._message_queue: Optional[Queue[Union[MessageType, object]]] = Queue()

    async def asend_message(self, message: MessageType, close_bundle: bool) -> None:
        self._message_queue.put_nowait(message)
        if close_bundle:
            self._message_queue.put_nowait(END_OF_QUEUE)

    async def asend_interim_message(self, message: MessageType) -> None:
        await self.asend_message(message, close_bundle=False)

    async def asend_final_message(self, message: MessageType) -> None:
        await self.asend_message(message, close_bundle=True)


class ChatManager:
    pass
