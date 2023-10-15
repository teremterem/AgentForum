"""Classes for sending and receiving messages between agents."""
import asyncio
from typing import List, AsyncIterator

from agentcache.errors import MessageBundleClosedError
from agentcache.typing import MessageType
from agentcache.utils import END_OF_QUEUE


class MessageBundle:
    """
    A bundle of messages that can be iterated over asynchronously. The speed at which messages can be sent to this
    bundle is independent of the speed at which consumers iterate over them.
    """

    def __init__(self) -> None:
        self.closed = False
        self.messages_so_far: List[MessageType] = []

        self._message_queue = asyncio.Queue()
        self._lock = asyncio.Lock()

    def __aiter__(self) -> AsyncIterator[MessageType]:
        # noinspection PyTypeChecker
        return self._Iterator(self)

    async def aget_all_messages(self) -> List[MessageType]:
        """Get all the messages in the bundle (making sure that all the messages are fetched first)."""
        async for _ in self:
            pass
        return self.messages_so_far

    async def asend_message(self, message: MessageType, close_bundle: bool) -> None:
        """Send a message to the bundle."""
        if self.closed:
            raise MessageBundleClosedError("Cannot send messages to a closed bundle.")
        self._message_queue.put_nowait(message)
        if close_bundle:
            self._message_queue.put_nowait(END_OF_QUEUE)
            self.closed = True

    async def asend_interim_message(self, message: MessageType) -> None:
        """Send an interim message to the bundle."""
        await self.asend_message(message, close_bundle=False)

    async def asend_final_message(self, message: MessageType) -> None:
        """Send the final message to the bundle. The bundle will be closed after this."""
        await self.asend_message(message, close_bundle=True)

    async def _wait_for_next_message(self) -> MessageType:
        if not self._message_queue:
            raise StopAsyncIteration

        message = await self._message_queue.get()

        if message is END_OF_QUEUE:
            self._message_queue = None
            raise StopAsyncIteration

        self.messages_so_far.append(message)
        return message

    class _Iterator:
        def __init__(self, message_bundle: "MessageBundle") -> None:
            self._message_bundle = message_bundle
            self._index = 0

        async def __anext__(self) -> MessageType:
            if self._index < len(self._message_bundle.messages_so_far):
                message = self._message_bundle.messages_so_far[self._index]
            elif not self._message_bundle._message_queue:
                raise StopAsyncIteration
            else:
                async with self._message_bundle._lock:
                    if self._index < len(self._message_bundle.messages_so_far):
                        message = self._message_bundle.messages_so_far[self._index]
                    else:
                        message = await self._message_bundle._wait_for_next_message()

            self._index += 1
            return message
